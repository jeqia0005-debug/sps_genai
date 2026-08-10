"""GPT-2 question-answering model shared by the training scripts and the API.

Reference: Module 8 Class Activity - Fine-tuning an LLM and adding it to the
Text Generation API.

The pipeline has three stages, each one starting from the previous:

1. `base`  - pretrained `openai-community/gpt2` from HuggingFace.
2. `sft`   - stage 1 plus the LoRA adapter fine-tuned on SQuAD.
3. `rl`    - stage 2 plus a second LoRA adapter trained with policy gradient to
             answer in the reasoning/answer format.

`build_qa_model` loads any of the three, and both the API and the training
scripts generate through the same helpers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.lora import inject_lora, load_lora, merge_lora
from app.squad_data import build_prompt

BASE_MODEL_NAME = os.environ.get("GPT2_MODEL_NAME", "openai-community/gpt2")
SFT_ADAPTER_PATH = os.environ.get("GPT2_SFT_ADAPTER", "gpt2_squad_lora.pt")
RL_ADAPTER_PATH = os.environ.get("GPT2_RL_ADAPTER", "gpt2_rl_lora.pt")

# A well-formed response is around 52 tokens at the median, so a smaller
# budget truncates the closing tag on a sizeable fraction of completions.
DEFAULT_MAX_NEW_TOKENS = 96


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick the best available device, or honour an explicit request."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(model_name: str = BASE_MODEL_NAME):
    """GPT-2 ships without a padding token; reuse the end-of-text token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Batched generation needs the padding on the left so that every sequence
    # ends at its own last real token.
    tokenizer.padding_side = "left"
    return tokenizer


@dataclass
class GenerationOutput:
    """One decoded completion plus the bookkeeping the RL trainer needs."""

    text: str
    finished: bool


@dataclass
class BatchGeneration:
    """Raw tensors for a batch of sampled completions.

    `attention_mask` covers the left padding of the prompts and everything after
    the first end-of-text token, so the same mask can be reused for the second
    forward pass that recomputes log probabilities.
    """

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    prompt_len: int
    outputs: list[GenerationOutput]


class QuestionAnswerer:
    """Wraps a GPT-2 policy with the prompt layout and sampling used everywhere."""

    def __init__(self, model, tokenizer, device: torch.device, stage: str = "base"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.stage = stage

    def prompt_for(self, question: str, context: str) -> str:
        return build_prompt(question, context)

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> BatchGeneration:
        """Sample completions for a batch of prompts.

        `num_return_sequences` samples several completions per prompt; they come
        back grouped by prompt.
        """
        self.model.eval()
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        prompt_len = encoded["input_ids"].shape[1]

        sequences = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        eos_id = self.tokenizer.eos_token_id
        prompt_mask = encoded["attention_mask"].repeat_interleave(num_return_sequences, dim=0)
        attention_mask = torch.cat(
            [prompt_mask, completion_mask(sequences, prompt_len, eos_id).long()], dim=1
        )

        outputs = []
        for row in sequences[:, prompt_len:]:
            tokens = row.tolist()
            finished = eos_id in tokens
            if finished:
                tokens = tokens[: tokens.index(eos_id)]
            outputs.append(
                GenerationOutput(text=self.tokenizer.decode(tokens), finished=finished)
            )
        return BatchGeneration(
            sequences=sequences,
            attention_mask=attention_mask,
            prompt_len=prompt_len,
            outputs=outputs,
        )

    def answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.7,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> GenerationOutput:
        """Single-question convenience wrapper used by the API endpoints."""
        batch = self.generate(
            [self.prompt_for(question, context)],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
        )
        return batch.outputs[0]


def sequence_log_probs(
    model,
    sequences: torch.Tensor,
    prompt_len: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log probability of every generated token under `model`.

    Generation happens without gradients; the policy gradient is taken on a
    second forward pass over prompt and completion together, which is both
    cheaper and simpler than threading gradients through the sampling loop.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(sequences)

    # Prompts are left padded, so positions have to be counted from the first
    # real token; otherwise this pass would see different position embeddings
    # than the generation pass did.
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

    logits = model(
        input_ids=sequences, attention_mask=attention_mask, position_ids=position_ids
    ).logits
    # Position t predicts token t+1, so drop the last position and shift targets.
    logits = logits[:, prompt_len - 1 : -1, :]
    targets = sequences[:, prompt_len:]

    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def completion_mask(
    sequences: torch.Tensor, prompt_len: int, eos_token_id: int
) -> torch.Tensor:
    """Mask keeping the generated tokens up to and including the first EOS."""
    completions = sequences[:, prompt_len:]
    is_eos = completions == eos_token_id
    # cumsum marks everything strictly after the first EOS, which is padding.
    after_eos = is_eos.cumsum(dim=1) - is_eos.long()
    return (after_eos == 0).float()


def build_qa_model(
    stage: str = "rl",
    model_name: str = BASE_MODEL_NAME,
    device: str | torch.device = "auto",
    sft_adapter: str = SFT_ADAPTER_PATH,
    rl_adapter: str = RL_ADAPTER_PATH,
    lora_rank: int = 8,
    lora_alpha: int = 16,
) -> QuestionAnswerer:
    """Load one of the three pipeline stages.

    The RL adapter was trained on top of the *merged* supervised model, so the
    supervised adapter has to be folded into the weights before the RL adapter
    is attached; loading them in any other order would not reproduce the policy
    that was trained.
    """
    if stage not in {"base", "sft", "rl"}:
        raise ValueError(f"Unknown stage '{stage}', expected base, sft or rl.")

    device = resolve_device(device) if isinstance(device, str) else device
    tokenizer = load_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    if stage in {"sft", "rl"}:
        if not os.path.exists(sft_adapter):
            raise FileNotFoundError(
                f"Supervised adapter '{sft_adapter}' not found. Run train_llm.py first."
            )
        inject_lora(model, rank=lora_rank, alpha=lora_alpha)
        load_lora(model, sft_adapter, device=str(device))

    if stage == "rl":
        if not os.path.exists(rl_adapter):
            raise FileNotFoundError(
                f"RL adapter '{rl_adapter}' not found. Run train_rl.py first."
            )
        merge_lora(model)
        inject_lora(model, rank=lora_rank, alpha=lora_alpha)
        load_lora(model, rl_adapter, device=str(device))

    model.to(device)
    model.eval()
    return QuestionAnswerer(model, tokenizer, device, stage=stage)
