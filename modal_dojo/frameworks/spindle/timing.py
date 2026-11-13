"""Translate tinker-cookbook spans into Dojo's timing lanes."""

from typing import Any

from modal_dojo.common.step_timing import PhaseTiming, Role, RoleTimingRecord


_PHASES = {
    "sampling": "generate_rollouts",
    "train_step": "train_models",
    "save_checkpoint_and_get_sampling_client": "checkpoint_save",
    "run_evaluations_parallel": "evaluate_rollouts",
}


def timing_record(
    training_run_id: str,
    step: int,
    spans: list[dict[str, Any]],
    lane_start_unix_s: float,
) -> RoleTimingRecord:
    phases = {}
    for span in spans:
        name = _PHASES.get(span["name"])
        if name is None:
            continue
        start, end = span["wall_start"], span["wall_end"]
        duration = span["duration"]
        if name not in phases:
            phases[name] = PhaseTiming(
                count=0,
                busy_duration_s=0,
                longest_invocation_s=0,
                first_start_s=start,
                last_end_s=end,
            )
        phase = phases[name]
        phase.count += 1
        phase.busy_duration_s += duration
        phase.longest_invocation_s = max(phase.longest_invocation_s, duration)
        phase.first_start_s = min(phase.first_start_s, start)
        phase.last_end_s = max(phase.last_end_s, end)
        phase.invocations.append((start, end))
    return RoleTimingRecord(
        training_run_id=training_run_id,
        rollout_id=step,
        role=Role.DRIVER,
        lane_start_unix_s=lane_start_unix_s,
        final=True,
        phases=phases,
    )
