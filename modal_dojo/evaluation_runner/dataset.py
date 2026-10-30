"""A Modal Evals evaluation built from a Dojo dataset's rows.

Each task holds one row's chat messages and label. The trial sends the
messages to the target as one chat completion and scores the reply with the
scorer the task names.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal

import httpx
import modal
import modal_evals as evals
from pydantic import Field

Scorer = Literal["exact_match", "contains", "boxed_math"]

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def _final_answer(text: str) -> str:
    if boxed := _BOXED.findall(text):
        return boxed[-1]
    if numbers := _NUMBER.findall(text):
        return numbers[-1]
    return text


def _number(text: str) -> float | None:
    try:
        return float(text.replace(",", "").strip())
    except ValueError:
        return None


def _math(response: str, label: str) -> float:
    answer, expected = _final_answer(response), _final_answer(label)
    answer_number, expected_number = _number(answer), _number(expected)
    if answer_number is not None and expected_number is not None:
        return float(abs(answer_number - expected_number) < 1e-6)
    return float(_normalize(answer) == _normalize(expected))


SCORERS: dict[str, Callable[[str, str], float]] = {
    "exact_match": lambda response, label: float(
        _normalize(response) == _normalize(label)
    ),
    "contains": lambda response, label: float(
        _normalize(label) in _normalize(response)
    ),
    "boxed_math": _math,
}


class DatasetParams(evals.Params):
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)


class DatasetTask(evals.TaskData):
    messages: str
    label: str
    scorer: Scorer


class DatasetResult(evals.TrialResult):
    score: Annotated[float, evals.HigherIsBetter]
    latency: Annotated[evals.Milliseconds, evals.LowerIsBetter]
    response: str


async def run(
    ctx: evals.TrialContext[DatasetParams, DatasetTask],
) -> DatasetResult:
    base_url = ctx.target.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers=dict(ctx.target.headers),
            json={
                "model": ctx.target.model,
                "messages": json.loads(ctx.task.data.messages),
                "temperature": ctx.params.temperature,
                "max_tokens": ctx.params.max_tokens,
            },
        )
    if response.status_code >= 400:
        raise evals.TrialError(
            "TARGET_ERROR",
            f"The target answered HTTP {response.status_code}",
            retryable=response.status_code in (408, 429) or response.status_code >= 500,
        )
    content = response.json()["choices"][0]["message"].get("content") or ""
    return DatasetResult(
        score=SCORERS[ctx.task.data.scorer](content, ctx.task.data.label),
        latency=(time.monotonic() - started) * 1000,
        response=content,
    )


def build_evaluation(
    key: str,
    tasks: Sequence[evals.Task[DatasetTask]],
    *,
    display_name: str,
    description: str,
    max_concurrency: int,
) -> evals.Evaluation:
    """An evaluation of ``tasks`` whose trials run :func:`run`."""
    evaluation = evals.Evaluation(
        key, display_name=display_name, description=description
    )
    listed = list(tasks)

    def list_tasks() -> list[evals.Task[DatasetTask]]:
        return listed

    evaluation.tasks()(list_tasks)
    evaluation.trial(
        image=modal.Image.debian_slim(python_version="3.12").uv_pip_install(  # pyright: ignore[reportArgumentType]
            "httpx~=0.28"
        ),
        timeout=600,
        max_concurrency=max_concurrency,
        package=__name__.rpartition(".")[0],
    )(run)
    evaluation.report(
        evals.Card("score", label="Score"),
        evals.Table("tasks"),
    )
    return evaluation


def task_from_row(
    index: int,
    row: dict[str, Any],
    *,
    input_key: str,
    label_key: str,
    scorer: Scorer,
    key_column: str | None = None,
) -> evals.Task[DatasetTask]:
    prompt = row[input_key]
    messages = (
        [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    )
    key = str(row[key_column]) if key_column else f"row-{index:06d}"
    return evals.Task(
        key,
        DatasetTask(
            messages=json.dumps(messages), label=str(row[label_key]), scorer=scorer
        ),
    )
