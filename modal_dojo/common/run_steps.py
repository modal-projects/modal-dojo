"""Change how many rollouts a running slime/Miles run will train for.

``modal-dojo run set-steps`` writes a target into a Modal Dict keyed by training
run id. The patched driver loop (see ``budget_stop_patch``) reads it before each
rollout and moves ``args.num_rollout`` to match, so a run can be extended or cut
short without restarting. The LR schedule keeps the length it was launched with.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

RUN_STEPS_DICT_NAME = "modal-dojo-run-steps"


def _run_steps_dict() -> Any:
    import modal

    return modal.Dict.from_name(RUN_STEPS_DICT_NAME, create_if_missing=True)


def set_target_steps(training_run_id: str, steps: int) -> None:
    _run_steps_dict()[training_run_id] = steps


def get_target_steps(training_run_id: str) -> int | None:
    value = _run_steps_dict().get(training_run_id)
    return value if isinstance(value, int) else None


def _apply_target_steps(args: Any, rollout_id: int, run_id: str) -> None:
    try:
        target = get_target_steps(run_id)
    except Exception as exc:  # a lookup failure must never stop training
        print(f"[modal-dojo] WARNING: could not read the step target: {exc}")
        return
    if target is None:
        return
    # The rollout about to start is already committed to (async runs have
    # generated it), so it is the earliest the run can end.
    target = max(target, rollout_id + 1)
    if target != args.num_rollout:
        print(
            f"[modal-dojo] Step target changed: num_rollout "
            f"{args.num_rollout} -> {target} at rollout {rollout_id}."
        )
        args.num_rollout = target


def rollout_ids(args: Any, start: int) -> Iterator[int]:
    """``range(start, args.num_rollout)`` that rereads the step target each rollout."""
    run_id = os.environ.get("MODAL_DOJO_TRAINING_RUN_ID", "")
    rollout_id = start
    while True:
        # Once the last rollout has run (and saved), the run is over: async
        # loops did not generate another rollout to extend into.
        if rollout_id >= args.num_rollout:
            return
        if run_id:
            _apply_target_steps(args, rollout_id, run_id)
        yield rollout_id
        rollout_id += 1
