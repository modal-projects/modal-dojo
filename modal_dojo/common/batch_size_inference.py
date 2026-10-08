"""Composer-style automatic micro-batch inference.

Mirrors Composer's ``device_train_microbatch_size="auto"``: set the per-GPU
micro-batch knob to ``"auto"`` and the launcher starts from the largest value
that can matter — the whole per-rank share of ``global_batch_size`` in one
micro-batch — then halves it and relaunches every time training dies with a
CUDA OOM. Only the per-GPU knob moves; ``global_batch_size`` is untouched, so
the optimizer sees the same batches whatever value the search settles on.

The knob is ``max_tokens_per_gpu`` when the recipe uses dynamic batching
(the default) and ``micro_batch_size`` otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

from modal_dojo.common.errors import DojoConfigError

AUTO = "auto"
METADATA_KEY = "batch_size_inference"
_KNOBS = ("max_tokens_per_gpu", "micro_batch_size")

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


def is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == AUTO


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _positive(recipe: Any, name: str, default: int = 1) -> int:
    value = _as_int(_effective(recipe, name))
    return value if value is not None and value > 0 else default


def active_knob(recipe: Any) -> str:
    """The per-GPU micro-batch knob this recipe's batching mode reads."""
    dynamic = bool(_effective(recipe, "use_dynamic_batch_size", True))
    return "max_tokens_per_gpu" if dynamic else "micro_batch_size"


def auto_batch_size_enabled(recipe: Any) -> bool:
    """True when the active micro-batch knob is set to ``"auto"``."""
    return is_auto(_effective(recipe, active_knob(recipe)))


def _sample_len(recipe: Any) -> int:
    """Longest sample the recipe declares, in tokens (0 when unknown)."""
    ctx = _as_int(_effective(recipe, "rollout_max_context_len")) or 0
    prompt = _as_int(_effective(recipe, "rollout_max_prompt_len")) or 0
    response = _as_int(_effective(recipe, "rollout_max_response_len")) or 0
    return ctx or (prompt + response)


def _data_parallel_size(recipe: Any) -> int:
    actor_gpus = _positive(recipe, "actor_num_nodes") * _positive(
        recipe, "actor_num_gpus_per_node"
    )
    model_parallel = (
        _positive(recipe, "tensor_model_parallel_size")
        * _positive(recipe, "pipeline_model_parallel_size")
        * _positive(recipe, "context_parallel_size")
    )
    return max(1, actor_gpus // model_parallel)


def resolve_batch_size_knob(recipe: Any) -> tuple[str, int, int]:
    """Return ``(knob, initial_value, floor)`` for a recipe whose knob is ``"auto"``.

    Like Composer, the search starts from the whole per-rank share of
    ``global_batch_size`` packed into a single micro-batch — anything larger
    cannot change what fits — and stops at the smallest value that can still
    free memory: the longest single sample for ``max_tokens_per_gpu`` (one
    sample per micro-batch), ``1`` for ``micro_batch_size``.
    """
    knob = active_knob(recipe)
    if not is_auto(_effective(recipe, knob)):
        raise DojoConfigError(
            f'automatic batch-size inference needs {knob}="auto"; got '
            f"{_effective(recipe, knob)!r}"
        )
    global_batch_size = _as_int(_effective(recipe, "global_batch_size"))
    if global_batch_size is None or global_batch_size < 1:
        raise DojoConfigError(
            f'{knob}="auto" needs global_batch_size to derive its starting point'
        )
    samples_per_rank = math.ceil(global_batch_size / _data_parallel_size(recipe))

    if knob == "max_tokens_per_gpu":
        sample = _sample_len(recipe)
        if sample < 1:
            raise DojoConfigError(
                'max_tokens_per_gpu="auto" needs rollout_max_response_len (or '
                "rollout_max_context_len) to size the search"
            )
        cp = _positive(recipe, "context_parallel_size")
        # Below the longest single sample, dynamic batching already puts one
        # sample per micro-batch, so shrinking further cannot free memory.
        floor = max(1, math.ceil(sample / cp))
        initial = max(floor, math.ceil(samples_per_rank * sample / cp))
    else:
        floor = 1
        initial = max(floor, samples_per_rank)
    return knob, initial, floor


def validate_batch_size_inference(recipe: Any) -> None:
    """Fail fast at launch time when ``"auto"`` is set where it cannot take effect."""
    knob = active_knob(recipe)
    for name in _KNOBS:
        if name != knob and is_auto(_effective(recipe, name)):
            mode = (
                "use_dynamic_batch_size=True reads max_tokens_per_gpu"
                if knob == "max_tokens_per_gpu"
                else "use_dynamic_batch_size=False reads micro_batch_size"
            )
            raise DojoConfigError(
                f'{name}="auto" has no effect here: {mode}, not {name}. '
                f'Set {knob}="auto" instead.'
            )
    if is_auto(_effective(recipe, knob)):
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

        Returns ``None`` unless the active micro-batch knob is ``"auto"``.
        """
        if not auto_batch_size_enabled(recipe):
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
    """The value the ``"auto"`` micro-batch search settled on, or ``None`` until it has."""
    stored = (metadata or {}).get(METADATA_KEY)
    if not isinstance(stored, dict):
        return None
    return _as_int(stored.get("inferred_batch_size_result"))
