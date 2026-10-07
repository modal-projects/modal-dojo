"""Composer-style automatic micro-batch inference.

Mirrors Composer's ``device_train_microbatch_size="auto"``: start from the
(deliberately high) per-GPU micro-batch knob the user configured and, every
time training dies with a CUDA OOM, halve it and relaunch. Only the per-GPU
knob moves; ``global_batch_size`` is untouched, so the optimizer sees the same
batches whatever value the search settles on.

The knob is ``max_tokens_per_gpu`` when the recipe uses dynamic batching
(the default) and ``micro_batch_size`` otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

from modal_dojo.common.errors import DojoConfigError

METADATA_KEY = "batch_size_inference"

# Matched case-insensitively against each streamed log line and against Ray's
# recorded driver failure message.
OOM_PATTERNS: tuple[str, ...] = (
    "out of memory",
    "outofmemoryerror",
    "cuda_error_out_of_memory",
    "cudaerrormemoryallocation",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)

_EVIDENCE_LIMIT = 300


def looks_like_oom(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(pattern in lowered for pattern in OOM_PATTERNS)


class OOMDetector:
    """Collects the first OOM-looking line from a streamed Ray job log."""

    def __init__(self) -> None:
        self.evidence: str | None = None

    def observe(self, line: str) -> None:
        if self.evidence is None and looks_like_oom(line):
            self.evidence = line.strip()[:_EVIDENCE_LIMIT]


def _effective(recipe: Any, name: str, default: Any = None) -> Any:
    """A recipe setting as the framework will see it: escape hatch over field."""
    hatch = (
        recipe._escape_hatch_values() if hasattr(recipe, "_escape_hatch_values") else {}
    )
    if name in hatch:
        return hatch[name]
    return getattr(recipe, name, default)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def resolve_batch_size_knob(recipe: Any) -> tuple[str, int, int]:
    """Return ``(knob, initial_value, floor)`` for ``infer_batch_size``.

    Raises ``DojoConfigError`` when the recipe gives the search nothing to
    shrink: the knob is unset, or already at the smallest value that can
    change peak memory.
    """
    dynamic = bool(_effective(recipe, "use_dynamic_batch_size", True))
    knob = "max_tokens_per_gpu" if dynamic else "micro_batch_size"
    value = _as_int(_effective(recipe, knob))
    if value is None or value < 1:
        where = (
            f"set {knob} on the recipe"
            if knob == "max_tokens_per_gpu" or hasattr(recipe, knob)
            else f"set {knob!r} in extra_config"
        )
        raise DojoConfigError(
            f"infer_batch_size=True needs a starting {knob} to shrink from; "
            f"{where} (deliberately high — the launcher halves it on OOM)."
        )

    if dynamic:
        ctx = _as_int(_effective(recipe, "rollout_max_context_len")) or 0
        prompt = _as_int(_effective(recipe, "rollout_max_prompt_len")) or 0
        response = _as_int(_effective(recipe, "rollout_max_response_len")) or 0
        sample = ctx or (prompt + response)
        cp = _as_int(_effective(recipe, "context_parallel_size")) or 1
        # Below the longest single sample, dynamic batching already puts one
        # sample per micro-batch, so shrinking further cannot free memory.
        floor = max(1, math.ceil(sample / cp)) if sample else 1
    else:
        floor = 1

    if value <= floor:
        raise DojoConfigError(
            f"infer_batch_size=True has nothing to search: {knob}={value} is already "
            f"at its floor ({floor}). Start much higher — the launcher halves it "
            "on OOM until training fits."
        )
    return knob, value, floor


def validate_batch_size_inference(recipe: Any) -> None:
    """Fail fast at launch time when ``infer_batch_size`` cannot do anything."""
    if getattr(recipe, "infer_batch_size", False):
        resolve_batch_size_knob(recipe)


def set_recipe_value(recipe: Any, name: str, value: Any) -> None:
    """Set a recipe setting where the framework will read it.

    Keys the user placed in the escape hatch (``extra_config``) win over the
    same-named field, so those are updated in place — before or after the
    launcher has materialized the hatch to YAML. Everything else is a field.
    """
    from modal_dojo.common.launcher_utils import set_escape_hatch_value

    if hasattr(recipe, "_escape_hatch_keys") and name in recipe._escape_hatch_keys():
        set_escape_hatch_value(recipe, name, value)
    elif any(f == name for f in _field_names(recipe)):
        setattr(recipe, name, value)
    else:
        set_escape_hatch_value(recipe, name, value)


def _field_names(recipe: Any) -> tuple[str, ...]:
    import dataclasses

    if dataclasses.is_dataclass(recipe):
        return tuple(f.name for f in dataclasses.fields(recipe))
    return ()


@dataclass
class BatchSizeInference:
    """Search state for one training run; persisted under ``metadata[METADATA_KEY]``."""

    knob: str
    initial: int
    floor: int
    current: int
    attempts: list[dict[str, Any]] = field(default_factory=list)
    settled: bool = False

    @classmethod
    def start(cls, recipe: Any, run_record: Any) -> BatchSizeInference | None:
        """Build the search for ``recipe``, resuming a previous container's state.

        Returns ``None`` when ``infer_batch_size`` is off.
        """
        if not getattr(recipe, "infer_batch_size", False):
            return None
        knob, initial, floor = resolve_batch_size_knob(recipe)
        state = cls(knob=knob, initial=initial, floor=floor, current=initial)
        metadata = getattr(run_record, "metadata", None) or {}
        stored = metadata.get(METADATA_KEY)
        if isinstance(stored, dict) and stored.get("knob") == knob:
            current = _as_int(stored.get("current"))
            if current is not None and floor <= current <= initial:
                state.current = current
            attempts = stored.get("attempts")
            if isinstance(attempts, list):
                state.attempts = [dict(a) for a in attempts if isinstance(a, dict)]
                for attempt in state.attempts:
                    if attempt.get("outcome") == "running":
                        attempt["outcome"] = "interrupted"
        return state

    @property
    def result(self) -> int | None:
        return self.current if self.settled else None

    def apply(self, recipe: Any) -> None:
        set_recipe_value(recipe, self.knob, self.current)

    def begin_attempt(self) -> None:
        self.attempts.append(
            {
                "value": self.current,
                "outcome": "running",
                "started_at": int(time.time()),
            }
        )

    def finish_attempt(self, outcome: str, evidence: str | None = None) -> None:
        if not self.attempts or self.attempts[-1].get("outcome") != "running":
            self.begin_attempt()
        attempt = self.attempts[-1]
        attempt["outcome"] = outcome
        attempt["ended_at"] = int(time.time())
        if evidence:
            attempt["evidence"] = evidence
        if outcome == "succeeded":
            self.settled = True

    def shrink(self) -> int | None:
        """Halve the knob (clamped to the floor); ``None`` once the floor failed."""
        if self.current <= self.floor:
            return None
        self.current = max(self.current // 2, self.floor)
        return self.current

    def to_metadata(self) -> dict[str, Any]:
        return {
            "knob": self.knob,
            "initial": self.initial,
            "floor": self.floor,
            "current": self.current,
            "settled": self.settled,
            "attempts": list(self.attempts),
            "inferred_batch_size_result": self.result,
        }

    def record(self, run_record: Any) -> None:
        if run_record.metadata is None:
            run_record.metadata = {}
        run_record.metadata[METADATA_KEY] = self.to_metadata()


def inferred_batch_size_result(metadata: dict[str, Any] | None) -> int | None:
    """The value ``infer_batch_size`` settled on, or ``None`` until it has."""
    stored = (metadata or {}).get(METADATA_KEY)
    if not isinstance(stored, dict):
        return None
    return _as_int(stored.get("inferred_batch_size_result"))
