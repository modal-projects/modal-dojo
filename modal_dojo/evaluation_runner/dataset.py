"""A Modal Evals evaluation built from a Dojo dataset's rows.

Each task holds one row's chat messages, its label, and the training run's
reward function, cloudpickled. The trial sends the messages to the target as
one chat completion and scores the reply with that reward function, called as
training calls it: ``reward(args, sample)`` with ``sample.response`` and
``sample.label``.
"""

from __future__ import annotations

import base64
import functools
import inspect
import json
import time
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, Any

import cloudpickle
import httpx
import modal
import modal_evals as evals
from pydantic import Field

RUNNER_DEPENDENCIES = (
    "click>=8.2,<9",
    "cloudpickle",
    "datasets",
    "httpx",
    "packaging",
    "pydantic>=2,<3",
    "fastapi",
    "randomname",
    "rich",
    "requests",
)


class DatasetParams(evals.Params):
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)


class DatasetTask(evals.TaskData):
    messages: str
    label: str
    reward: str


class DatasetResult(evals.TrialResult):
    score: Annotated[float, evals.HigherIsBetter]
    latency: Annotated[evals.Milliseconds, evals.LowerIsBetter]
    response: str


def pickle_reward(reward: Callable[..., Any]) -> str:
    return base64.b64encode(cloudpickle.dumps(reward)).decode("ascii")


@functools.cache
def _reward(payload: str) -> Callable[..., Any]:
    return cloudpickle.loads(base64.b64decode(payload))


async def score(
    reward: Callable[..., Any], messages: Any, response: str, label: str
) -> float:
    sample = SimpleNamespace(
        prompt=messages, response=response, label=label, metadata={}, status=None
    )
    value = reward(SimpleNamespace(), sample)
    if inspect.isawaitable(value):
        value = await value
    if isinstance(value, dict):
        value = value.get("reward", value.get("score"))
    if not isinstance(value, (bool, int, float)):
        raise TypeError(f"The reward function returned {value!r}, not a number")
    return float(value)


async def run(
    ctx: evals.TrialContext[DatasetParams, DatasetTask],
) -> DatasetResult:
    base_url = ctx.target.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    messages = json.loads(ctx.task.data.messages)
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers=dict(ctx.target.headers),
            json={
                "model": ctx.target.model,
                "messages": messages,
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
        score=await score(
            _reward(ctx.task.data.reward), messages, content, ctx.task.data.label
        ),
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
    """An evaluation of ``tasks`` whose trials run :func:`run`.

    The runner Image carries all of ``modal_dojo`` and its dependencies, so a
    reward function that uses them unpickles there.
    """
    evaluation = evals.Evaluation(
        key, display_name=display_name, description=description
    )
    listed = list(tasks)

    def list_tasks() -> list[evals.Task[DatasetTask]]:
        return listed

    evaluation.tasks()(list_tasks)
    evaluation.trial(
        image=modal.Image.debian_slim(python_version="3.12").uv_pip_install(  # pyright: ignore[reportArgumentType]
            *RUNNER_DEPENDENCIES
        ),
        timeout=600,
        max_concurrency=max_concurrency,
        package=__name__.partition(".")[0],
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
    reward: str,
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
            messages=json.dumps(messages), label=str(row[label_key]), reward=reward
        ),
    )
