"""Reward function for the reinforcement learning stage.

The target response format is

    <reasoning> ... </reasoning> <answer> ... </answer>

`compute_reward` returns the only signal the policy receives during training.
The score is shaped rather than binary: partial credit is given for each tag
that appears, so completions that are close to the format score above ones that
ignore it entirely.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import asdict, dataclass, field

from app.squad_data import ANSWER_CLOSE, ANSWER_OPEN, REASONING_CLOSE, REASONING_OPEN

STRUCTURE_PATTERN = re.compile(
    rf"^\s*{REASONING_OPEN}(?P<reasoning>.*?){REASONING_CLOSE}"
    rf"\s*{ANSWER_OPEN}(?P<answer>.*?){ANSWER_CLOSE}\s*$",
    re.DOTALL,
)

TAG_REWARD = 0.25          # per tag that appears exactly once
STRUCTURE_REWARD = 1.0     # the whole completion matches the layout
ANSWER_REWARD = 2.0        # scaled by the token F1 against the gold answer
BREVITY_PENALTY = 0.5      # answer span padded with commentary
EMPTY_REASONING_PENALTY = 0.5
UNFINISHED_PENALTY = 0.5   # ran into the token budget without stopping

MAX_ANSWER_WORDS = 12


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop articles and punctuation."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token overlap F1, the standard SQuAD partial-credit metric."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass
class ParsedResponse:
    """What the model actually produced, split into its two sections."""

    reasoning: str = ""
    answer: str = ""
    well_formed: bool = False


def parse_response(completion: str) -> ParsedResponse:
    """Extract the reasoning and answer sections from a completion."""
    match = STRUCTURE_PATTERN.match(completion.strip())
    if match:
        return ParsedResponse(
            reasoning=match.group("reasoning").strip(),
            answer=match.group("answer").strip(),
            well_formed=True,
        )

    # Fall back to a lenient extraction so a partially formatted completion can
    # still be scored on the content of its answer section.
    lenient = re.search(rf"{ANSWER_OPEN}(.*?)(?:{ANSWER_CLOSE}|$)", completion, re.DOTALL)
    reasoning = re.search(
        rf"{REASONING_OPEN}(.*?)(?:{REASONING_CLOSE}|$)", completion, re.DOTALL
    )
    return ParsedResponse(
        reasoning=reasoning.group(1).strip() if reasoning else "",
        answer=lenient.group(1).strip() if lenient else completion.strip(),
        well_formed=False,
    )


@dataclass
class RewardBreakdown:
    """Per-component reward, kept separate so the API can explain a score."""

    tag_reward: float = 0.0
    structure_reward: float = 0.0
    answer_reward: float = 0.0
    penalties: float = 0.0
    answer_f1: float = 0.0
    format_compliant: bool = False
    parsed: ParsedResponse = field(default_factory=ParsedResponse)

    @property
    def total(self) -> float:
        return self.tag_reward + self.structure_reward + self.answer_reward - self.penalties

    def as_dict(self) -> dict:
        data = asdict(self)
        data["total"] = round(self.total, 4)
        return data


def compute_reward(
    completion: str,
    gold_answers: list[str] | str,
    finished: bool = True,
) -> RewardBreakdown:
    """Score one completion against the target format and the gold answer."""
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    breakdown = RewardBreakdown()

    for tag in (REASONING_OPEN, REASONING_CLOSE, ANSWER_OPEN, ANSWER_CLOSE):
        if completion.count(tag) == 1:
            breakdown.tag_reward += TAG_REWARD

    parsed = parse_response(completion)
    breakdown.parsed = parsed
    breakdown.format_compliant = parsed.well_formed
    if parsed.well_formed:
        breakdown.structure_reward = STRUCTURE_REWARD

    breakdown.answer_f1 = max(
        (token_f1(parsed.answer, gold) for gold in gold_answers), default=0.0
    )
    breakdown.answer_reward = ANSWER_REWARD * breakdown.answer_f1

    if len(parsed.answer.split()) > MAX_ANSWER_WORDS:
        breakdown.penalties += BREVITY_PENALTY
    if not parsed.reasoning:
        breakdown.penalties += EMPTY_REASONING_PENALTY
    if not finished:
        breakdown.penalties += UNFINISHED_PENALTY

    return breakdown


def format_description() -> dict:
    """Machine-readable description of the target format, served by the API."""
    return {
        "template": f"{REASONING_OPEN} ... {REASONING_CLOSE} {ANSWER_OPEN} ... {ANSWER_CLOSE}",
        "components": {
            "tag_reward": f"{TAG_REWARD} per tag appearing exactly once (max {4 * TAG_REWARD})",
            "structure_reward": f"{STRUCTURE_REWARD} if the whole completion matches the layout",
            "answer_reward": f"{ANSWER_REWARD} x token F1 against the SQuAD gold answer",
            "penalties": (
                f"{BREVITY_PENALTY} if the answer exceeds {MAX_ANSWER_WORDS} words, "
                f"{EMPTY_REASONING_PENALTY} if the reasoning section is empty, "
                f"{UNFINISHED_PENALTY} if generation hit the token budget"
            ),
        },
        "max_reward": 4 * TAG_REWARD + STRUCTURE_REWARD + ANSWER_REWARD,
    }
