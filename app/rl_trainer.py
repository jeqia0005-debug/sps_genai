"""Policy gradient post-training of the question-answering model.

Reference: Module 9 - Basics of Reinforcement Learning, Module 10 - Applying
Reinforcement Learning to Language.

Text generation is treated as a Markov decision process: the state is the prompt
plus the tokens generated so far, an action is the next token, an episode is one
completion, and the reward is the score `app.rl_reward` assigns to the finished
completion.

Each epoch samples a batch of episodes from the current policy and takes one
gradient step on

    loss = -(log pi(a_t | s_t) * R(tau)).mean()

where R(tau) is the return of the episode the action belongs to.

Episodes are sampled in one batched generation call, and log probabilities are
recomputed in a second forward pass over prompt and completion together so that
the gradient reaches the sampled tokens.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field

import torch

from app.llm_model import QuestionAnswerer, sequence_log_probs
from app.lora import adapters_disabled, lora_parameters
from app.rl_reward import compute_reward
from app.squad_data import QAExample


@dataclass
class PolicyGradientConfig:
    """Hyperparameters of the RL stage."""

    episodes_per_epoch: int = 16     # completions collected before each update
    max_new_tokens: int = 96
    temperature: float = 1.0
    top_k: int = 50
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    epochs: int = 200
    seed: int = 0
    log_every: int = 5

    # A non-zero value adds a KL penalty against the supervised policy, which
    # keeps the text from degenerating when the format reward dominates.
    kl_coefficient: float = 0.0


@dataclass
class EpochMetrics:
    """One policy gradient update, logged for the training-curve endpoint."""

    epoch: int
    mean_return: float
    max_return: float
    format_rate: float
    answer_f1: float
    kl: float
    loss: float
    completion_tokens: float


@dataclass
class TrainingLog:
    """Everything the API needs to report on how the RL stage went."""

    config: dict = field(default_factory=dict)
    epochs: list[dict] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)


class PolicyGradientTrainer:
    """Trains the LoRA adapter of a `QuestionAnswerer` with REINFORCE."""

    def __init__(
        self,
        policy: QuestionAnswerer,
        examples: list[QAExample],
        config: PolicyGradientConfig | None = None,
    ):
        self.policy = policy
        self.examples = examples
        self.config = config or PolicyGradientConfig()
        self.rng = random.Random(self.config.seed)

        trainable = lora_parameters(policy.model)
        if not trainable:
            raise ValueError("The policy has no LoRA parameters to train.")
        self.optimizer = torch.optim.Adam(trainable, lr=self.config.learning_rate)
        self.log = TrainingLog(config=asdict(self.config))

    def collect_episodes(self):
        """Act in the environment with the current policy and score the results.

        Every episode starts from a randomly drawn question.
        """
        batch = [self.rng.choice(self.examples) for _ in range(self.config.episodes_per_epoch)]
        prompts = [self.policy.prompt_for(ex.question, ex.context) for ex in batch]

        generation = self.policy.generate(
            prompts,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
        )
        breakdowns = [
            compute_reward(output.text, example.answers, finished=output.finished)
            for example, output in zip(batch, generation.outputs)
        ]
        return batch, generation, breakdowns

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """The loss whose gradient is the policy gradient.

        `weights` holds the return of the episode each action belongs to, so it
        is broadcast across every token of a completion.  The mask keeps padding
        and everything past the end-of-text token out of the average.
        """
        weighted = -(log_probs * weights.unsqueeze(1))
        return (weighted * mask).sum() / mask.sum().clamp(min=1.0)

    def train_one_epoch(self, epoch: int) -> EpochMetrics:
        batch, generation, breakdowns = self.collect_episodes()

        returns = torch.tensor(
            [b.total for b in breakdowns], dtype=torch.float32, device=self.policy.device
        )
        mask = generation.attention_mask[:, generation.prompt_len :].float()

        # Dropout stays off so the recomputed log probabilities match the ones
        # the sampled tokens were actually drawn from; gradients still flow.
        self.policy.model.eval()
        log_probs = sequence_log_probs(
            self.policy.model, generation.sequences, generation.prompt_len, generation.attention_mask
        )

        loss = self.compute_loss(log_probs, returns, mask)

        kl_value = 0.0
        if self.config.kl_coefficient > 0:
            with torch.no_grad(), adapters_disabled(self.policy.model):
                ref_log_probs = sequence_log_probs(
                    self.policy.model,
                    generation.sequences,
                    generation.prompt_len,
                    generation.attention_mask,
                )
            # Low-variance, always positive KL estimator (Schulman's k3).
            log_ratio = ref_log_probs - log_probs
            kl = torch.exp(log_ratio) - log_ratio - 1.0
            kl_term = (kl * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss + self.config.kl_coefficient * kl_term
            kl_value = float(kl_term.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            lora_parameters(self.policy.model), self.config.max_grad_norm
        )
        self.optimizer.step()

        metrics = EpochMetrics(
            epoch=epoch,
            mean_return=float(returns.mean()),
            max_return=float(returns.max()),
            format_rate=sum(b.format_compliant for b in breakdowns) / len(breakdowns),
            answer_f1=sum(b.answer_f1 for b in breakdowns) / len(breakdowns),
            kl=kl_value,
            loss=float(loss.detach()),
            completion_tokens=float(mask.sum(dim=1).mean()),
        )

        if epoch % self.config.log_every == 0:
            best = max(range(len(breakdowns)), key=lambda i: breakdowns[i].total)
            self.log.samples.append(
                {
                    "epoch": epoch,
                    "question": batch[best].question,
                    "completion": generation.outputs[best].text,
                    "return": round(breakdowns[best].total, 4),
                }
            )
        return metrics

    def train(self, verbose: bool = True) -> TrainingLog:
        started = time.time()
        for epoch in range(1, self.config.epochs + 1):
            metrics = self.train_one_epoch(epoch)
            self.log.epochs.append(asdict(metrics))
            if verbose and (epoch % self.config.log_every == 0 or epoch == 1):
                print(
                    f"epoch {epoch:4d} | return {metrics.mean_return:6.3f} "
                    f"| format {metrics.format_rate:5.1%} | f1 {metrics.answer_f1:5.3f} "
                    f"| loss {metrics.loss:8.4f}"
                )
        self.log.duration_seconds = time.time() - started
        return self.log


@torch.no_grad()
def evaluate(
    policy: QuestionAnswerer,
    examples: list[QAExample],
    max_new_tokens: int = 96,
    temperature: float = 0.7,
    batch_size: int = 8,
) -> dict:
    """Measure format compliance and answer quality on held-out questions."""
    returns, f1s, compliant = [], [], 0

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prompts = [policy.prompt_for(ex.question, ex.context) for ex in chunk]
        generation = policy.generate(
            prompts, max_new_tokens=max_new_tokens, temperature=temperature
        )
        for example, output in zip(chunk, generation.outputs):
            breakdown = compute_reward(output.text, example.answers, finished=output.finished)
            returns.append(breakdown.total)
            f1s.append(breakdown.answer_f1)
            compliant += int(breakdown.format_compliant)

    count = max(len(returns), 1)
    return {
        "examples": len(returns),
        "mean_return": round(sum(returns) / count, 4),
        "format_compliance": round(compliant / count, 4),
        "answer_f1": round(sum(f1s) / count, 4),
    }
