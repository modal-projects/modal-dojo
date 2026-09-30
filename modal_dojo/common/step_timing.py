from __future__ import annotations

import math
from enum import Enum
from typing import Any, Awaitable

from pydantic import BaseModel, Field, model_validator

from modal_dojo.common.status import SlimeStatus
from modal_dojo.common.timing_recorder import (
    MAX_PHASE_INVOCATIONS,
    MAX_TIMING_PHASES,
    trim_invocation_lists,
)
from modal_dojo.utils.metadata import (
    MetadataStore,
    vol_list,
    vol_put,
)

MAX_ROLLOUT_ID = 1_000_000
# Keep phase names bounded because they become per-record JSON object keys.
MAX_PHASE_NAME_LENGTH = 128


class Role(str, Enum):
    # The orchestrating process that runs the training loop itself; it isn't a
    # GPU worker, it sequences rollout generation, weight sync and checkpointing.
    DRIVER = "driver"
    # The inference engines generating samples from the current policy.
    ROLLOUT = "rollout"
    # The policy model being trained.
    ACTOR = "actor"
    # The value model that scores states, present only in PPO-style runs
    # (use_critic=true); absent for GRPO.
    CRITIC = "critic"


class PhaseTiming(BaseModel):
    count: int
    # Sum of invocation durations; retained as busy time, unlike the wall-clock
    # span between first_start_s and last_end_s.
    busy_duration_s: float
    # Longest single invocation; retained to expose per-invocation outliers.
    longest_invocation_s: float
    # Together these define the wall-clock span
    # (last_end_s - first_start_s), including gaps between invocations.
    first_start_s: float
    last_end_s: float
    invocations: list[tuple[float, float]] = Field(
        default_factory=list, max_length=MAX_PHASE_INVOCATIONS
    )


class RoleTimingRecord(BaseModel):
    training_run_id: str
    rollout_id: int | None = Field(default=None, ge=0, le=MAX_ROLLOUT_ID)
    role: Role
    lane_start_unix_s: float | None = None
    final: bool = False
    phases: dict[str, PhaseTiming] = Field(
        default_factory=dict, max_length=MAX_TIMING_PHASES
    )

    @model_validator(mode="after")
    def _validate_phases(self) -> "RoleTimingRecord":
        def finite_phase(phase: PhaseTiming) -> bool:
            values = (
                phase.busy_duration_s,
                phase.longest_invocation_s,
                phase.first_start_s,
                phase.last_end_s,
                *(value for pair in phase.invocations for value in pair),
            )
            return all(math.isfinite(value) for value in values)

        self.phases = {
            name: phase
            for name, phase in self.phases.items()
            if len(name) <= MAX_PHASE_NAME_LENGTH and finite_phase(phase)
        }
        trimmed = trim_invocation_lists(
            {
                name: [list(pair) for pair in phase.invocations]
                for name, phase in self.phases.items()
            }
        )
        for name, phase in self.phases.items():
            phase.invocations = [(start, end) for start, end in trimmed[name]]
        return self

    @property
    def storage_key(self) -> str:
        rollout = "pre-loop" if self.rollout_id is None else f"{self.rollout_id:08d}"
        return f"{rollout}__{self.role.value}"

    @staticmethod
    def store(training_run_id: str) -> str:
        return f"{MetadataStore.SUBSTEP_TIMING.value}/{training_run_id}"

    def save(self, *, is_async: bool = False) -> None | Awaitable[None]:
        return vol_put(
            self.store(self.training_run_id),
            self.storage_key,
            self.model_dump(mode="json"),
            is_async=is_async,
        )


class Substep(str, Enum):
    INITIAL_WEIGHT_SYNC = "initial_weight_sync"
    EVAL_BEFORE = SlimeStatus.EVAL_ROLLOUT_LOGGING.value
    GENERATE_ROLLOUTS = SlimeStatus.ROLLOUT_LOGGING.value
    GENERATE_SAMPLES = "generate_samples"
    SAMPLE_GENERATION = "sample_generation"
    REWARD = "reward"
    REWARD_BATCH = "reward_batch"
    REWARD_POST_PROCESS = "reward_post_process"
    OFFLOAD_ROLLOUT = SlimeStatus.OFFLOAD_ROLLOUT.value
    TRAIN_MODELS = "train_models"
    COMPUTE_LOG_PROBS = SlimeStatus.COMPUTE_LOG_PROBS.value
    FORWARD_BACKWARD = "forward_backward"
    OPTIMIZER_STEP = SlimeStatus.OPTIMIZER_STEP.value
    TRAIN_STEP_FINALIZE = "train_step_finalize"
    TRAINER_FINALIZE = "trainer_finalize"
    CHECKPOINT_SAVE = SlimeStatus.CHECKPOINT_SAVE.value
    OFFLOAD_TRAIN = SlimeStatus.OFFLOAD_TRAIN.value
    WEIGHT_SYNC = SlimeStatus.WEIGHT_SYNC.value
    EVAL_AFTER = f"{SlimeStatus.EVAL_ROLLOUT_LOGGING.value}_end"
    WAIT_FOR_ROLLOUT = "wait_for_rollout"
    WAIT_FOR_NEXT_ROLLOUT = "wait_for_next_rollout"


def rollout_lanes(records: list[dict[str, Any]]) -> dict[str, Any]:
    lanes = {
        record["role"]: {
            "role": record["role"],
            "lane_start_unix_s": record["lane_start_unix_s"],
            "phases": record["phases"],
        }
        for record in records
    }
    return {"roles": lanes}


def measured_run_times(
    training_run_id: str,
) -> tuple[
    dict[str, dict[str, float | bool | None]],
    dict[str, dict[str, dict[str, float | int | bool | None]]],
]:
    not_in_step = (
        Substep.CHECKPOINT_SAVE.value,
        Substep.EVAL_BEFORE.value,
        Substep.EVAL_AFTER.value,
    )
    step_times: dict[str, dict[str, float | None]] = {}
    substep_times: dict[str, dict[str, dict[str, float | int | bool | None]]] = {}
    for rollout_id, records in sorted(
        load_run(training_run_id).items(),
        key=lambda item: (item[0] is None, item[0] or 0),
    ):
        if rollout_id is None:
            continue
        substeps: dict[str, dict[str, float | None]] = {}
        step_duration = 0.0
        for record in records:
            lane_start = record["lane_start_unix_s"]
            if lane_start is None:
                continue
            role = record["role"]
            for name, phase in record["phases"].items():
                if role == Role.DRIVER.value and name not in not_in_step:
                    step_duration += phase["busy_duration_s"]
                if role == Role.DRIVER.value:
                    key = name
                else:
                    key = f"{name} ({role})"
                wall_duration = phase["last_end_s"] - phase["first_start_s"]
                invocation_count = phase["count"]
                substeps[key] = {
                    "start": lane_start + phase["first_start_s"],
                    "duration_s": phase["busy_duration_s"],
                    "wall_duration_s": wall_duration,
                    "invocation_count": invocation_count,
                    "concurrent": (
                        role != Role.DRIVER.value
                        and invocation_count > 1
                        and phase["busy_duration_s"] > wall_duration * 1.05
                    ),
                }
        if not substeps:
            continue
        step = str(rollout_id + 1)
        driver_records = [
            record for record in records if record["role"] == Role.DRIVER.value
        ]
        partial = not driver_records or not any(
            Substep.TRAIN_MODELS.value in record["phases"] for record in driver_records
        )
        step_times[step] = {
            "duration_s": round(step_duration, 3),
            "partial": partial,
        }
        substep_times[step] = substeps
    return step_times, substep_times


def load_run(training_run_id: str) -> dict[int | None, list[dict[str, Any]]]:
    records = []
    for record in vol_list(RoleTimingRecord.store(training_run_id)):
        validated = validate_timing_record(record)
        if validated is not None:
            records.append(validated)
    grouped: dict[int | None, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["rollout_id"], []).append(record)
    return grouped


def validate_timing_record(record: Any) -> dict[str, Any] | None:
    try:
        return RoleTimingRecord.model_validate(record).model_dump(mode="json")
    except (TypeError, ValueError):
        return None


# Remove once no pre-measured-timing runs remain in the metadata volume.
_LEGACY_RENAMES = {Substep.OPTIMIZER_STEP.value: Substep.TRAIN_MODELS.value}


def legacy_run_to_records(
    substep_times: dict[str, dict[str, dict[str, float | int | bool | None]]] | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for step_key, subs in (substep_times or {}).items():
        if not step_key.isdigit():
            continue
        lane_start = min(
            (
                float(start)
                for sub in subs.values()
                if isinstance(start := sub.get("start"), (int, float))
                and not isinstance(start, bool)
            ),
            default=None,
        )
        if lane_start is None:
            continue
        phases: dict[str, dict[str, float]] = {}
        for name, sub in subs.items():
            start, duration = sub.get("start"), sub.get("duration_s")
            if start is None or duration is None:
                continue
            rel = start - lane_start
            phases[_LEGACY_RENAMES.get(name, name)] = {
                "count": 1,
                "busy_duration_s": round(duration, 6),
                "longest_invocation_s": round(duration, 6),
                "first_start_s": round(rel, 6),
                "last_end_s": round(rel + duration, 6),
            }
        records.append(
            {
                "rollout_id": int(step_key) - 1,
                "role": Role.DRIVER.value,
                "lane_start_unix_s": lane_start,
                "phases": phases,
            }
        )
    return records
