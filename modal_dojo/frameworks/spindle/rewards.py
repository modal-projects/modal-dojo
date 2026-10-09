"""Default reward for :class:`SpindleRecipe` runs without ``custom_rm_function``.

Mirrors the Miles math-RL default: the response's final ``\\boxed{...}`` (or
last number) is compared with the dataset label.
"""

from __future__ import annotations

import re
from typing import Any

_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def extract_answer(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    boxed = _BOXED.findall(text)
    if boxed:
        return normalize(boxed[-1])
    numbers = _NUMBER.findall(text)
    return normalize(numbers[-1]) if numbers else normalize(text.strip())


def normalize(answer: str) -> str:
    answer = answer.strip().strip("$").replace(",", "").replace(" ", "")
    answer = re.sub(r"\\(text|mathrm|mbox)\{([^}]*)\}", r"\2", answer)
    answer = answer.rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", answer):
        answer = answer.split(".")[0]
    return answer.lower()


def boxed_match_reward(args: Any, sample: Any) -> float:
    label = sample.label
    if isinstance(label, dict):
        label = label.get("answer", label.get("label", ""))
    if label is None:
        return 0.0
    return 1.0 if extract_answer(sample.response) == normalize(str(label)) else 0.0
