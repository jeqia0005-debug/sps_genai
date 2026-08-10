"""Tests for the reinforcement learning components.

None of these load GPT-2, so the whole file runs in a couple of seconds:

    uv run python tests/test_rl_pipeline.py
    uv run pytest tests/
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.lora import (  # noqa: E402
    adapters_disabled,
    inject_lora,
    load_lora,
    lora_parameters,
    merge_lora,
    save_lora,
)
from app.rl_reward import compute_reward, parse_response, token_f1  # noqa: E402
from app.rl_trainer import PolicyGradientConfig, PolicyGradientTrainer  # noqa: E402
from app.squad_data import build_structured_response  # noqa: E402

WELL_FORMED = build_structured_response(
    "Alexandre Dumas", "The Count of Monte Cristo was written by Alexandre Dumas."
)


class TinyAttention(nn.Module):
    """Stand-in for a GPT-2 block: LoRA targets modules by attribute name."""

    def __init__(self, width: int = 16):
        super().__init__()
        self.c_attn = nn.Linear(width, width)
        self.c_proj = nn.Linear(width, width)

    def forward(self, x):
        return self.c_proj(torch.relu(self.c_attn(x)))


def test_reward_is_maximal_for_a_well_formed_correct_answer():
    breakdown = compute_reward(WELL_FORMED, ["Alexandre Dumas"])
    assert breakdown.format_compliant
    assert breakdown.answer_f1 == 1.0
    assert breakdown.penalties == 0.0
    assert breakdown.total == 4.0


def test_correct_answer_without_the_format_scores_lower():
    formatted = compute_reward(WELL_FORMED, ["Alexandre Dumas"]).total
    plain = compute_reward("Alexandre Dumas", ["Alexandre Dumas"]).total
    assert plain < formatted
    # Answer content is scored independently of the tags.
    assert plain > 0


def test_partial_formatting_earns_partial_credit():
    """All three completions carry the same answer, isolating the tag credit."""
    none = compute_reward("Dumas", ["Dumas"])
    partial = compute_reward(
        "<reasoning> he wrote it </reasoning> <answer> Dumas", ["Dumas"]
    )
    full = compute_reward(build_structured_response("Dumas", "he wrote it"), ["Dumas"])

    assert none.tag_reward == 0.0
    assert 0 < partial.tag_reward < full.tag_reward
    assert not partial.format_compliant
    assert none.total < partial.total < full.total


def test_unfinished_completions_are_penalised():
    finished = compute_reward(WELL_FORMED, ["Alexandre Dumas"], finished=True)
    truncated = compute_reward(WELL_FORMED, ["Alexandre Dumas"], finished=False)
    assert truncated.total < finished.total


def test_token_f1_gives_partial_credit():
    assert token_f1("Alexandre Dumas", "Alexandre Dumas") == 1.0
    assert token_f1("the Alexandre Dumas.", "Alexandre Dumas") == 1.0  # normalised
    assert 0 < token_f1("Alexandre", "Alexandre Dumas") < 1.0
    assert token_f1("Victor Hugo", "Alexandre Dumas") == 0.0


def test_parsing_falls_back_when_tags_are_incomplete():
    lenient = parse_response("<answer> Paris")
    assert lenient.answer == "Paris"
    assert not lenient.well_formed


def test_adapter_starts_as_a_no_op_and_can_be_switched_off():
    torch.manual_seed(0)
    model = TinyAttention()
    x = torch.randn(4, 16)
    baseline = model(x).clone()

    inject_lora(model, rank=4, alpha=8)
    # B is initialised to zero, so the adapter changes nothing before training.
    assert torch.allclose(model(x), baseline, atol=1e-6)

    for param in lora_parameters(model):
        nn.init.normal_(param, std=0.1)
    assert not torch.allclose(model(x), baseline, atol=1e-4)

    with adapters_disabled(model):
        assert torch.allclose(model(x), baseline, atol=1e-6)


def test_merging_preserves_the_adapted_output():
    torch.manual_seed(0)
    model = TinyAttention()
    inject_lora(model, rank=4, alpha=8)
    for param in lora_parameters(model):
        nn.init.normal_(param, std=0.1)

    x = torch.randn(4, 16)
    adapted = model(x).clone()
    merge_lora(model)
    assert torch.allclose(model(x), adapted, atol=1e-5)
    assert not lora_parameters(model)


def test_adapter_round_trips_through_a_checkpoint():
    torch.manual_seed(0)
    model = TinyAttention()
    inject_lora(model, rank=4, alpha=8)
    for param in lora_parameters(model):
        nn.init.normal_(param, std=0.1)

    x = torch.randn(4, 16)
    expected = model(x).clone()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "adapter.pt")
        save_lora(model, path, metadata={"stage": "test"})

        # Same path the API takes: frozen weights from elsewhere, adapter from
        # the checkpoint. `.base.` is the prefix the LoRA wrapper introduces.
        frozen = {
            key.replace(".base.", "."): value
            for key, value in model.state_dict().items()
            if "lora_" not in key
        }
        reloaded = TinyAttention()
        reloaded.load_state_dict(frozen)
        inject_lora(reloaded, rank=4, alpha=8)
        metadata = load_lora(reloaded, path)

    assert metadata["stage"] == "test"
    assert torch.allclose(reloaded(x), expected, atol=1e-6)


def make_trainer() -> PolicyGradientTrainer:
    """Only `compute_loss` is exercised, so no model has to be built."""
    trainer = PolicyGradientTrainer.__new__(PolicyGradientTrainer)
    trainer.config = PolicyGradientConfig()
    return trainer


def test_loss_is_the_masked_return_weighted_log_probability():
    trainer = make_trainer()
    log_probs = torch.tensor([[-1.0, -2.0, -9.0], [-0.5, -1.5, -9.0]])
    returns = torch.tensor([2.0, -1.0])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])  # third token is padding

    loss = trainer.compute_loss(log_probs, returns, mask)

    # -(logp * R) averaged over the four unmasked tokens; the padded positions
    # must not contribute even though their log probability is large.
    expected = -(2.0 * (-1.0) + 2.0 * (-2.0) + (-1.0) * (-0.5) + (-1.0) * (-1.5)) / 4
    assert abs(float(loss) - expected) < 1e-6


def test_return_is_broadcast_to_every_action_of_an_episode():
    """Every action of an episode is weighted by that episode's return."""
    trainer = make_trainer()
    mask = torch.ones(1, 2)

    def gradient_for(episode_return: float) -> torch.Tensor:
        log_probs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        trainer.compute_loss(log_probs, torch.tensor([episode_return]), mask).backward()
        return log_probs.grad

    rewarded = gradient_for(3.0)
    punished = gradient_for(-3.0)

    # Descending this gradient raises the probability of the tokens of a
    # rewarded episode and lowers it for a punished one.
    assert (rewarded < 0).all()
    assert (punished > 0).all()
    # Both actions carry the same weight regardless of when they were taken.
    assert abs(float(rewarded[0, 0] - rewarded[0, 1])) < 1e-6


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
