"""SQuAD question-answering data for the GPT-2 fine-tuning pipeline.

Reference: Module 8 Class Activity - Fine-tuning an LLM and adding it to the
Text Generation API.

SQuAD v1.1 is read from the raw JSON files, which are cached on disk after the
first download.

Two response layouts are produced from the same example: the plain layout used
by the supervised fine-tuning stage, and the structured layout that wraps the
response in reasoning/answer tags and is rewarded by the RL stage.
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass
from typing import Iterable

SQUAD_URLS = {
    "train": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json",
    "dev": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json",
}

DEFAULT_DATA_DIR = "data/squad"

REASONING_OPEN = "<reasoning>"
REASONING_CLOSE = "</reasoning>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"


@dataclass
class QAExample:
    """A single SQuAD question with its context and accepted answers."""

    question: str
    context: str
    answers: list[str]
    answer_start: int

    @property
    def gold_answer(self) -> str:
        return self.answers[0]

    def supporting_sentence(self) -> str:
        """Return the context sentence that contains the answer.

        Used as the reasoning text of the structured layout: it gives the model
        something concrete to put between the reasoning tags instead of an
        empty span.
        """
        sentences = re.split(r"(?<=[.!?])\s+", self.context)
        offset = 0
        for sentence in sentences:
            end = offset + len(sentence)
            if offset <= self.answer_start < end + 1:
                return sentence.strip()
            offset = end + 1
        return sentences[0].strip() if sentences else self.context.strip()


def download_squad(split: str = "train", data_dir: str = DEFAULT_DATA_DIR) -> str:
    """Download a SQuAD split if it is not on disk yet and return its path."""
    if split not in SQUAD_URLS:
        raise ValueError(f"Unknown split '{split}', expected one of {list(SQUAD_URLS)}.")

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{split}-v1.1.json")
    if not os.path.exists(path):
        urllib.request.urlretrieve(SQUAD_URLS[split], path)
    return path


def load_squad(
    split: str = "train",
    limit: int | None = None,
    data_dir: str = DEFAULT_DATA_DIR,
    max_context_chars: int = 700,
    seed: int = 42,
) -> list[QAExample]:
    """Load SQuAD examples, keeping only the ones with a short enough context.

    Long contexts blow up the sequence length and make fine-tuning on a CPU
    impractical, so paragraphs above `max_context_chars` are skipped rather than
    truncated: truncating risks cutting away the answer span.
    """
    path = download_squad(split, data_dir)
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)

    examples: list[QAExample] = []
    for article in raw["data"]:
        for paragraph in article["paragraphs"]:
            context = " ".join(paragraph["context"].split())
            if len(context) > max_context_chars:
                continue
            for qa in paragraph["qas"]:
                answers = [a["text"] for a in qa["answers"]]
                if not answers:
                    continue
                examples.append(
                    QAExample(
                        question=" ".join(qa["question"].split()),
                        context=context,
                        answers=answers,
                        answer_start=qa["answers"][0]["answer_start"],
                    )
                )

    random.Random(seed).shuffle(examples)
    return examples[:limit] if limit else examples


def build_prompt(question: str, context: str) -> str:
    """The part of the sequence the model conditions on but is never scored on."""
    return f"Context: {context}\nQuestion: {question}\nAnswer:"


def build_plain_response(answer: str) -> str:
    """Response layout used by the Module 8 question-answering fine-tune."""
    return f" {answer}"


def build_structured_response(answer: str, reasoning: str) -> str:
    """Response layout the reinforcement learning stage is asked to produce."""
    return (
        f" {REASONING_OPEN} {reasoning} {REASONING_CLOSE}"
        f" {ANSWER_OPEN} {answer} {ANSWER_CLOSE}"
    )


def build_training_pairs(
    examples: Iterable[QAExample],
    structured_ratio: float = 0.25,
    seed: int = 0,
) -> list[tuple[str, str]]:
    """Turn examples into (prompt, response) pairs for supervised fine-tuning.

    `structured_ratio` sets how many responses use the structured layout, which
    is what leaves the model sampling that layout occasionally when the RL stage
    starts.
    """
    rng = random.Random(seed)
    pairs = []
    for example in examples:
        prompt = build_prompt(example.question, example.context)
        if rng.random() < structured_ratio:
            response = build_structured_response(
                example.gold_answer, example.supporting_sentence()
            )
        else:
            response = build_plain_response(example.gold_answer)
        pairs.append((prompt, response))
    return pairs
