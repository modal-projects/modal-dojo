"""A scoped tinker-cookbook rollout hook: capture complete groups before filtering."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import fields, is_dataclass
from functools import wraps
from pathlib import Path
from typing import Any


def _decode_turns(tokenizer, trajectory) -> list[dict[str, str]]:
    """Each turn's new prompt tokens and its response.

    A turn's prompt usually repeats the whole conversation so far; decoding
    only what follows it keeps long multi-turn episodes linear in size.
    """
    turns = []
    context: list[int] = []
    for transition in trajectory.transitions:
        prompt = transition.ob.to_ints()
        shared = 0
        for seen, token in zip(context, prompt):
            if seen != token:
                break
            shared += 1
        turns.append(
            {
                "prompt": tokenizer.decode(prompt[shared:]),
                "response": tokenizer.decode(transition.ac.tokens),
            }
        )
        context = prompt + list(transition.ac.tokens)
    return turns


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@asynccontextmanager
async def capture_trajectories(log_path: str, tokenizer, *, on_group=None):
    from tinker_cookbook.rl import rollouts

    original = rollouts.do_group_rollout
    pending = set()
    path = Path(log_path) / "dojo-trajectories.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:

        @wraps(original)
        async def capture(env_group_builder, *args, **kwargs):
            task = asyncio.current_task()
            pending.add(task)
            try:
                group = await original(env_group_builder, *args, **kwargs)
            finally:
                pending.discard(task)
            record = {
                "tags": env_group_builder.logging_tags(),
                "trajectory_group": _jsonable(group),
                "decoded": [
                    _decode_turns(tokenizer, trajectory)
                    for trajectory in group.trajectories_G
                ],
            }
            line = json.dumps(record, ensure_ascii=False)
            output.write(line + "\n")
            output.flush()
            if on_group is not None:
                await on_group(record)
            return group

        rollouts.do_group_rollout = capture
        try:
            yield path
        finally:
            rollouts.do_group_rollout = original
            # Drain sibling rollouts before closing the output file.
            tasks = list(pending)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
