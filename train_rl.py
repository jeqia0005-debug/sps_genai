"""Stage 2: post-train the fine-tuned GPT-2 with reinforcement learning.

Reference: Module 9 - Basics of Reinforcement Learning, Module 10 - Applying
Reinforcement Learning to Language.

The supervised model from `train_llm.py` answers SQuAD questions but uses the
reasoning/answer layout only part of the time.  This stage trains it to always
use that layout from the reward in `app.rl_reward` alone, with no labelled
example of the layout.

    uv run python train_rl.py --epochs 200

The supervised adapter is merged into the frozen weights first, so the adapter
trained here sits on top of it.
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM

from app.llm_model import (
    BASE_MODEL_NAME,
    SFT_ADAPTER_PATH,
    QuestionAnswerer,
    load_tokenizer,
    resolve_device,
)
from app.lora import count_trainable, inject_lora, load_lora, merge_lora, save_lora
from app.rl_trainer import PolicyGradientConfig, PolicyGradientTrainer, evaluate
from app.squad_data import load_squad


def parse_args():
    parser = argparse.ArgumentParser(
        description="Policy gradient post-training of the SQuAD model."
    )
    parser.add_argument("--model-name", default=BASE_MODEL_NAME)
    parser.add_argument("--sft-adapter", default=SFT_ADAPTER_PATH)
    parser.add_argument("--output", default="gpt2_rl_lora.pt")
    parser.add_argument("--log", default="rl_training_log.json")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--episodes-per-epoch", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--kl-coefficient",
        type=float,
        default=0.0,
        help="KL penalty against the supervised policy, disabled at 0.0. "
        "Raise it (e.g. 0.02) if the policy drifts into degenerate text.",
    )

    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_policy(args, device) -> QuestionAnswerer:
    """Load the supervised model and attach a fresh adapter for the RL stage."""
    tokenizer = load_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    metadata = load_lora(model, args.sft_adapter, device=str(device))
    print(f"Loaded supervised adapter: {metadata}")

    merge_lora(model)
    inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    model.to(device)

    trainable, total = count_trainable(model)
    print(f"Trainable parameters: {trainable:,} of {total:,} ({trainable / total:.2%})")
    return QuestionAnswerer(model, tokenizer, device, stage="rl")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_examples = load_squad("train", limit=args.train_examples, seed=args.seed + 1)
    eval_examples = load_squad("dev", limit=args.eval_examples, seed=args.seed + 2)

    policy = build_policy(args, device)

    print("\nEvaluating the supervised policy before RL ...")
    before = evaluate(policy, eval_examples, max_new_tokens=args.max_new_tokens)
    print(json.dumps(before, indent=2))

    config = PolicyGradientConfig(
        episodes_per_epoch=args.episodes_per_epoch,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        learning_rate=args.lr,
        kl_coefficient=args.kl_coefficient,
        epochs=args.epochs,
        seed=args.seed,
    )
    trainer = PolicyGradientTrainer(policy, train_examples, config)

    print("\nRunning policy gradient training ...")
    log = trainer.train()

    print("\nEvaluating the RL policy ...")
    after = evaluate(policy, eval_examples, max_new_tokens=args.max_new_tokens)
    print(json.dumps(after, indent=2))

    log.config["evaluation"] = {"before_rl": before, "after_rl": after}
    log.save(args.log)
    save_lora(
        policy.model,
        args.output,
        metadata={
            "stage": "rl",
            "base_model": args.model_name,
            "algorithm": "REINFORCE",
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "epochs": args.epochs,
            "evaluation": {"before_rl": before, "after_rl": after},
        },
    )
    print(f"\nSaved RL adapter to {args.output} and training log to {args.log}")


if __name__ == "__main__":
    main()
