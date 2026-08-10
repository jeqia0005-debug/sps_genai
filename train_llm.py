"""Stage 1: fine-tune GPT-2 on SQuAD question answering.

Reference: Module 8 Class Activity - Fine-tuning an LLM and adding it to the
Text Generation API.

Only the LoRA adapter is trained, so the artefact this script writes is a couple
of MB instead of the ~500MB a full GPT-2 checkpoint would take.

    uv run python train_llm.py --examples 4000 --epochs 1
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM

from app.llm_model import BASE_MODEL_NAME, load_tokenizer, resolve_device
from app.lora import count_trainable, inject_lora, lora_parameters, save_lora
from app.squad_data import build_training_pairs, load_squad

IGNORE_INDEX = -100


class QADataset(Dataset):
    """Prompt/response pairs tokenised for causal-language-model training."""

    def __init__(self, pairs, tokenizer, max_length: int = 320):
        self.examples = []
        for prompt, response in pairs:
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
            response_ids = response_ids + [tokenizer.eos_token_id]

            input_ids = (prompt_ids + response_ids)[:max_length]
            # The prompt is context, not something the model should learn to
            # produce, so its positions are excluded from the loss.
            labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
            self.examples.append((input_ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def make_collate_fn(pad_token_id: int):
    """Right-pad a batch to its longest sequence."""

    def collate(batch):
        width = max(len(ids) for ids, _ in batch)
        input_ids, labels, attention = [], [], []
        for ids, lbl in batch:
            padding = width - len(ids)
            input_ids.append(ids + [pad_token_id] * padding)
            labels.append(lbl + [IGNORE_INDEX] * padding)
            attention.append([1] * len(ids) + [0] * padding)
        return (
            torch.tensor(input_ids),
            torch.tensor(labels),
            torch.tensor(attention),
        )

    return collate


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on SQuAD.")
    parser.add_argument("--model-name", default=BASE_MODEL_NAME)
    parser.add_argument("--examples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument(
        "--structured-ratio",
        type=float,
        default=0.25,
        help="Fraction of training responses using the reasoning/answer layout.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="gpt2_squad_lora.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    tokenizer = load_tokenizer(args.model_name)
    examples = load_squad("train", limit=args.examples, seed=args.seed)
    pairs = build_training_pairs(examples, structured_ratio=args.structured_ratio, seed=args.seed)
    print(f"Loaded {len(pairs)} SQuAD training pairs.")

    dataset = QADataset(pairs, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    model.to(device)
    trainable, total = count_trainable(model)
    print(f"Trainable parameters: {trainable:,} of {total:,} ({trainable / total:.2%})")

    optimizer = torch.optim.AdamW(lora_parameters(model), lr=args.lr)
    model.train()

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        running_loss, seen = 0.0, 0
        for step, (input_ids, labels, attention) in enumerate(loader, start=1):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention = attention.to(device)

            loss = model(input_ids=input_ids, attention_mask=attention, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
            optimizer.step()

            running_loss += loss.item()
            seen += 1
            if step % 50 == 0 or step == len(loader):
                print(
                    f"epoch {epoch}/{args.epochs} | step {step}/{len(loader)} "
                    f"| avg loss {running_loss / seen:.4f}"
                )

    save_lora(
        model,
        args.output,
        metadata={
            "stage": "sft",
            "base_model": args.model_name,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "examples": len(pairs),
            "epochs": args.epochs,
            "structured_ratio": args.structured_ratio,
            "final_loss": round(running_loss / max(seen, 1), 4),
            "minutes": round((time.time() - started) / 60, 2),
        },
    )
    print(f"Saved supervised adapter to {args.output}")


if __name__ == "__main__":
    main()
