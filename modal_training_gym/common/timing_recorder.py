from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, cast

from modal_training_gym.common import reporting

TIMING_MODE_ENV = "TRAINING_GYM_SUBSTEP_TIMING"
TIMING_DEBUG_ENV = "TRAINING_GYM_TIMING_DEBUG"
LOSS_TYPE_ENV = "TRAINING_GYM_LOSS_TYPE"

MIN_PUBLISH_INTERVAL_S = 3.0
MAX_PHASE_INVOCATIONS = 10_000
MAX_TIMING_PHASES = 64
# Keep posted invocation detail bounded without changing measured aggregates.
MAX_TOTAL_INVOCATIONS = 20_000

PER_SAMPLE_PHASES = frozenset({"reward", "reward_batch", "sample_generation"})
RL_ONLY_PHASES = frozenset(
    {"initial_weight_sync", "weight_sync", "reward_post_process"}
)


def _timing_debug(event: str, **fields: object) -> None:
    if os.environ.get(TIMING_DEBUG_ENV) != "1":
        return
    payload = {"event": event, **fields}
    print(
        f"[timing-debug] {json.dumps(payload, sort_keys=True, default=str)}", flush=True
    )


def trim_invocation_lists(
    invocations_by_phase: dict[str, list[list[float]]],
) -> dict[str, list[list[float]]]:
    remaining = MAX_TOTAL_INVOCATIONS
    trimmed: dict[str, list[list[float]]] = {}
    for name in sorted(invocations_by_phase):
        invocations = invocations_by_phase[name]
        kept = min(len(invocations), remaining)
        trimmed[name] = invocations[:kept]
        remaining -= kept
    return trimmed


_PRELOOP_RECORDERS: dict[str, RoleRecorder] = {}
_PRELOOP_LOCK = threading.Lock()
_TIMING_MODE_CACHE: str | None = None


def timing_mode() -> str:
    """Return the process timing mode, read once before worker hot paths run."""
    global _TIMING_MODE_CACHE
    if _TIMING_MODE_CACHE is None:
        _TIMING_MODE_CACHE = os.environ.get(TIMING_MODE_ENV, "auto")
    return _TIMING_MODE_CACHE


class _NoopRecorder:
    """Stands in for a recorder when timing is off, so `rec.phase(...)` is free."""

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        yield


class RoleRecorder:
    def __init__(
        self,
        role: str,
        rollout_id: int | None,
        publish_gate: Callable[[], bool | None] | None = None,
        persistent: bool = False,
    ) -> None:
        self.role = role
        self.rollout_id = rollout_id

        self._publish_gate = publish_gate
        self._gate_answer: bool | None = None
        self._t0 = time.monotonic()
        self.lane_start_unix_s = time.time()
        self.phases: dict[str, dict[str, float]] = {}
        self.invocations: dict[str, list[list[float]]] = {}
        self._last_publish_t = float("-inf")
        self._lock = threading.Lock()
        self._posted_phases: dict[str, dict[str, object]] | None = None
        self._persistent = persistent
        self._closed = False
        self._dropped_phase_names: set[str] = set()

    def __enter__(self) -> "RoleRecorder":
        return self

    def __exit__(self, *exc: object) -> None:
        if self._persistent:
            try:
                self._publish(force=True)
            except Exception:
                pass
        else:
            self._close()

    def _close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._publish(force=True, final=True)
        except Exception:
            pass

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if timing_mode() == "off" or (
            name in RL_ONLY_PHASES and os.environ.get(LOSS_TYPE_ENV) == "sft_loss"
        ):
            yield
            return
        start = time.monotonic()
        try:
            yield
        finally:
            end = time.monotonic()
            duration = end - start
            dropped = False
            with self._lock:
                timing = self.phases.get(name)
                if timing is None and len(self.phases) >= MAX_TIMING_PHASES:
                    dropped = name not in self._dropped_phase_names
                    self._dropped_phase_names.add(name)
                    timing = {}
                if timing is None:
                    self.phases[name] = {
                        "count": 1,
                        "busy_duration_s": duration,
                        "longest_invocation_s": duration,
                        "first_start_s": start - self._t0,
                        "last_end_s": end - self._t0,
                    }
                    self.invocations[name] = (
                        []
                        if name in PER_SAMPLE_PHASES
                        else [[start - self._t0, end - self._t0]]
                    )
                elif timing:
                    timing["count"] += 1
                    timing["busy_duration_s"] += duration
                    timing["longest_invocation_s"] = max(
                        timing["longest_invocation_s"], duration
                    )
                    timing["first_start_s"] = min(
                        timing["first_start_s"], start - self._t0
                    )
                    timing["last_end_s"] = max(timing["last_end_s"], end - self._t0)
                    if (
                        name not in PER_SAMPLE_PHASES
                        and len(self.invocations[name]) < MAX_PHASE_INVOCATIONS
                    ):
                        self.invocations[name].append(
                            [start - self._t0, end - self._t0]
                        )
            if dropped:
                print(
                    f"[training-gym] dropping timing phase {name!r}: "
                    f"maximum of {MAX_TIMING_PHASES} phases reached",
                    flush=True,
                )
            try:
                self._publish()
            except Exception:
                pass

    def _publish(self, force: bool = False, final: bool = False) -> None:
        with self._lock:
            phase_names = sorted(self.phases)
        debug_fields: dict[str, object] = {
            "lane": self._lane_key,
            "force": force,
            "final": final,
            "phases": phase_names,
            "lane_start_unix_s": self.lane_start_unix_s,
        }
        reason: str | None = None
        with self._lock:
            has_phases = bool(self.phases)
            if not has_phases:
                reason = "no_phases"
        if reason is None and timing_mode() == "off":
            reason = "timing_off"
        training_run_id = ""
        if reason is None:
            training_run_id = os.environ.get("TRAINING_GYM_TRAINING_RUN_ID", "")
            if not training_run_id:
                reason = "no_run_id"
        if reason is None:
            gate_answer: bool | None = self._gate_answer
            if self._publish_gate is not None:
                if gate_answer is None:
                    gate_answer = self._publish_gate()
                    if gate_answer is not None:
                        self._gate_answer = gate_answer
                if gate_answer is False:
                    debug_fields["gate_answer"] = gate_answer
                    reason = "publish_gate"
                elif gate_answer is None and not force:
                    reason = "gate_undecidable"

        snapshot: dict[str, object] | None = None
        if reason is None:
            now = time.monotonic()
            with self._lock:
                elapsed = now - self._last_publish_t
                if not force and elapsed < MIN_PUBLISH_INTERVAL_S:
                    debug_fields["elapsed_s"] = elapsed
                    reason = "throttled"
                else:
                    invocations = trim_invocation_lists(self.invocations)
                    phases = {
                        name: {
                            **{key: round(value, 6) for key, value in timing.items()},
                            "invocations": [
                                [round(start, 6), round(end, 6)]
                                for start, end in invocations[name]
                            ],
                        }
                        for name, timing in self.phases.items()
                    }
                    snapshot = {
                        "training_run_id": training_run_id,
                        "rollout_id": self.rollout_id,
                        "role": self.role,
                        "storage_key": self._lane_key,
                        "lane_start_unix_s": self.lane_start_unix_s,
                        "final": final,
                        "phases": phases,
                    }
                    if (
                        not force
                        and self._posted_phases is not None
                        and self._posted_phases == snapshot["phases"]
                    ):
                        debug_fields["phases"] = sorted(phases)
                        reason = "unchanged"
                    else:
                        self._posted_phases = phases
                        self._last_publish_t = now
                        debug_fields["phases"] = sorted(phases)
                        reason = "queued"

        if reason == "queued":
            reporting._enqueue_timing(cast(dict[str, object], snapshot), final=final)
        _timing_debug(
            "publish",
            outcome="queued" if reason == "queued" else "skipped",
            reason=reason,
            **debug_fields,
        )

    @property
    def _lane_key(self) -> str:
        rollout = "pre-loop" if self.rollout_id is None else str(self.rollout_id)
        return f"{rollout}__{self.role}"


_ACTIVE_LANE: ContextVar[RoleRecorder | None] = ContextVar(
    "training_gym_active_lane", default=None
)


@contextmanager
def time_phase(name: str) -> Iterator[None]:
    if timing_mode() == "off":
        yield
        return
    rec = _ACTIVE_LANE.get()
    if rec is None:
        yield
        return
    with rec.phase(name):
        yield


@contextmanager
def recording_lane(
    role: str,
    rollout_id: int | None,
    publish_gate: Callable[[], bool | None] | None = None,
) -> Iterator[RoleRecorder | _NoopRecorder]:
    if timing_mode() == "off":
        token = _ACTIVE_LANE.set(None)
        try:
            yield _NoopRecorder()
        finally:
            _ACTIVE_LANE.reset(token)
        return
    if rollout_id is None:
        with _PRELOOP_LOCK:
            rec = _PRELOOP_RECORDERS.get(role)
            if rec is None:
                rec = RoleRecorder(role, rollout_id, publish_gate, persistent=True)
                _PRELOOP_RECORDERS[role] = rec
    else:
        rec = RoleRecorder(role, rollout_id, publish_gate)
    token = _ACTIVE_LANE.set(rec)
    try:
        with rec:
            yield rec
    finally:
        _ACTIVE_LANE.reset(token)


def _lowest_rank_publishes() -> bool | None:
    rank: int | None = None
    for env_name in ("RANK", "LOCAL_RANK"):
        raw_rank = os.environ.get(env_name)
        if raw_rank is None:
            continue
        try:
            rank = int(raw_rank)
        except ValueError:
            continue
        break
    try:
        import torch.distributed as dist  # pyright: ignore[reportMissingImports]  # torch is installed only in training images
    except ImportError:
        return rank == 0 if rank is not None else True
    if not dist.is_initialized():
        return rank == 0 if rank is not None else None
    get_group_ranks = getattr(dist, "get_process_group_ranks", None)
    if get_group_ranks is None:
        return dist.get_rank() == 0
    return dist.get_rank() == min(get_group_ranks(dist.group.WORLD))


@contextmanager
def recording_lane_on_reporting_rank(
    rollout_id: int, role: str = "actor"
) -> Iterator[RoleRecorder | _NoopRecorder]:
    with recording_lane(role, rollout_id, _lowest_rank_publishes) as rec:
        yield rec


def _close_preloop_recorders() -> None:
    with _PRELOOP_LOCK:
        recorders = tuple(_PRELOOP_RECORDERS.values())
    for recorder in recorders:
        try:
            recorder._close()
        except Exception:
            pass


reporting.register_pre_drain_hook(_close_preloop_recorders)


__all__ = [
    "RoleRecorder",
    "time_phase",
    "recording_lane",
    "recording_lane_on_reporting_rank",
]
