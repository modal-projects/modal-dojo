"""Default per-step metrics every run gets: tokens, throughput and cost.

The frameworks log what they measure (response lengths, trainer throughput,
step time); the dashboard turns those into the figures runs are compared
by, prices them with the workspace's GPU rates, and writes them into the
run's metric series under the ``dojo/`` prefix like any other metric. They
show on the Metrics tab and read back through the same chunks, so nothing
downstream needs billing rates or timing records of its own.

A training step's wall time comes from the driver's substep timing when
there is any (it excludes checkpoint saves and evals), else from the
framework's own ``perf/step_time``. Metrics with a recorded ``rollout/step``
axis retain that identity across resumes and missing points. Legacy metrics
without an axis use each source key's arrival order; another step value in
the same merged row is not evidence of that metric's axis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from modal_dojo.common.metric_series import StepTable
from modal_dojo.common.step_timing import Role, Substep

DERIVED_PREFIX = "dojo/"

# What the frameworks log.
RESPONSE_LEN_KEY = "rollout/response_len/mean"
TRAIN_TOK_PER_S_KEY = "perf/actor_train_tok_per_s"
TRAIN_TIME_KEY = "perf/actor_train_time"
STEP_TIME_KEY = "perf/step_time"

# What this module adds. Tokens and seconds are per step; throughput is per
# wall-clock second of the step; cost is USD per million tokens, charging the
# whole step's GPU time to the tokens, so these are end-to-end figures.
STEP_SECONDS_KEY = f"{DERIVED_PREFIX}step_seconds"
GEN_TOKENS_KEY = f"{DERIVED_PREFIX}gen_tokens"
TRAIN_TOKENS_KEY = f"{DERIVED_PREFIX}train_tokens"
TOKENS_KEY = f"{DERIVED_PREFIX}tokens"
GEN_TPS_KEY = f"{DERIVED_PREFIX}gen_tokens_per_s"
TRAIN_TPS_KEY = f"{DERIVED_PREFIX}train_tokens_per_s"
TPS_KEY = f"{DERIVED_PREFIX}tokens_per_s"
STEP_COST_KEY = f"{DERIVED_PREFIX}step_cost_usd"
# The list price the step was costed at, so readers never back it out.
GPU_HOUR_USD_KEY = f"{DERIVED_PREFIX}gpu_hour_usd"
GEN_COST_KEY = f"{DERIVED_PREFIX}gen_usd_per_mtok"
TRAIN_COST_KEY = f"{DERIVED_PREFIX}train_usd_per_mtok"
COST_KEY = f"{DERIVED_PREFIX}usd_per_mtok"

_NOT_IN_STEP = (
    Substep.CHECKPOINT_SAVE.value,
    Substep.EVAL_BEFORE.value,
    Substep.EVAL_AFTER.value,
)


def gpu_hour_cost(rates: Mapping[str, object], gpu_type: str) -> float | None:
    """List price of one GPU-hour, from ``Workspace.billing.rates()``."""
    name = gpu_type.split(":")[0].strip().rstrip("!+").lower().replace("-", "_")
    value = rates.get(f"gpu_hour_cost_{name}")
    return None if value is None else float(value)  # type: ignore[arg-type]


def billing_rates() -> Mapping[str, object]:
    """The workspace's current list rates, keyed like ``gpu_hour_cost_h100``."""
    import modal

    return modal.Workspace.from_context().billing.rates()


def samples_per_step(recipe: Mapping[str, object]) -> int | None:
    """Responses one step generates: prompts times samples per prompt."""
    prompts = recipe.get("rollout_batch_size")
    per_prompt = recipe.get("n_samples_per_prompt") or 1
    return int(prompts) * int(per_prompt) if prompts else None  # type: ignore[call-overload]


@dataclass(frozen=True)
class RunPricing:
    """What a run's steps cost and how many samples each generates."""

    gpus: int
    gpu_hour_usd: float | None
    samples_per_step: int | None
    # Some recipes plot rollout metrics against optimizer steps instead of IDs.
    rollout_step_is_id: bool = True

    @property
    def usd_per_s(self) -> float | None:
        if self.gpu_hour_usd is None or not self.gpus:
            return None
        return self.gpus * self.gpu_hour_usd / 3600

    @classmethod
    def from_recipe(
        cls, recipe: Mapping[str, object], rates: Mapping[str, object]
    ) -> RunPricing:
        """From a run record's ``config["recipe"]`` and the billing rates."""
        from modal_dojo.train_recipes.gpu_allocation import resolve_gpu_allocation

        allocated = recipe.get("allocated_gpus")
        if type(allocated) is int and allocated > 0:
            gpus = allocated
        else:
            try:
                gpus = resolve_gpu_allocation(
                    SimpleNamespace(**recipe), warn=False
                ).total_gpus
            except Exception:  # noqa: BLE001 — an unresolvable allocation prices nothing
                gpus = 0
        gpu_type = str(recipe.get("gpu_type") or "")
        return cls(
            gpus=gpus,
            gpu_hour_usd=gpu_hour_cost(rates, gpu_type) if gpu_type else None,
            samples_per_step=samples_per_step(recipe),
            rollout_step_is_id=not recipe.get("wandb_always_use_train_step", False),
        )


def driver_step_seconds(lanes: Mapping[str, Any]) -> dict[int, float]:
    """Each completed step's duration from the driver's substep timing.

    ``lanes`` is the dashboard's per-rollout view (``rollout_lanes`` per
    rollout id). Checkpoint saves and evals are not part of the step, and a
    rollout without a ``train_models`` phase is still in flight.
    """
    seconds: dict[int, float] = {}
    for rollout_id, lane in lanes.items():
        if not str(rollout_id).isdigit() or not isinstance(lane, Mapping):
            continue
        driver = (lane.get("roles") or {}).get(Role.DRIVER.value)
        if not isinstance(driver, Mapping):
            continue
        phases = driver.get("phases") or {}
        if Substep.TRAIN_MODELS.value not in phases:
            continue
        busy = sum(
            float(phase["busy_duration_s"])
            for name, phase in phases.items()
            if name not in _NOT_IN_STEP
        )
        seconds[int(rollout_id) + 1] = round(busy, 3)
    return seconds


def logged_per_step(
    table: StepTable,
    key: str,
    step_keys: Mapping[int, Mapping[str, str | None]] | None = None,
) -> dict[int, tuple[int, float]]:
    """A metric the framework logs once per step: ``{n: (row, value)}``,
    using its rollout axis when recorded, otherwise arrival order.

    Timing keys are one-based rollout IDs; source rows stay unchanged.
    """
    rows = [
        (row, metrics[key]) for row, metrics in sorted(table.items()) if key in metrics
    ]
    return {
        (row + 1 if (step_keys or {}).get(row, {}).get(key) == "rollout/step" else n): (
            row,
            value,
        )
        for n, (row, value) in enumerate(rows, start=1)
    }


def _finite(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and value == value else None


def derived_points(
    table: StepTable,
    measured_seconds: Mapping[int, float],
    pricing: RunPricing,
    *,
    step_keys: Mapping[int, Mapping[str, str | None]] | None = None,
) -> dict[int, dict[str, float]]:
    """The derived metrics each table row should carry, keyed by row.

    Values land in the row their source metric came from, so they plot on the
    same x-axis. Only what can be computed from the inputs is returned: no
    GPU rate means no cost keys; no samples-per-step means no generated-token
    keys.
    """
    response_len = logged_per_step(table, RESPONSE_LEN_KEY, step_keys)
    train_rate = logged_per_step(table, TRAIN_TOK_PER_S_KEY, step_keys)
    train_time = logged_per_step(table, TRAIN_TIME_KEY, step_keys)
    step_time = logged_per_step(table, STEP_TIME_KEY, step_keys)

    def seconds_for(n: int) -> tuple[int | None, float | None]:
        if pricing.rollout_step_is_id and n in measured_seconds:
            return None, measured_seconds[n]
        if n in step_time:
            row, value = step_time[n]
            return row, _finite(value)
        return None, None

    out: dict[int, dict[str, float]] = {}

    def put(row: int, key: str, value: float | None) -> None:
        if value is not None:
            out.setdefault(row, {})[key] = round(value, 4)

    steps = sorted(set(response_len) | set(train_rate))
    for n in steps:
        gen_row = response_len.get(n, (None, None))[0]
        train_row = train_rate.get(n, (None, None))[0]
        row = gen_row if gen_row is not None else train_row
        assert row is not None
        time_row, seconds = seconds_for(n)
        put(time_row if time_row is not None else row, STEP_SECONDS_KEY, seconds)

        gen = None
        if gen_row is not None and pricing.samples_per_step:
            mean = _finite(response_len[n][1])
            gen = None if mean is None else mean * pricing.samples_per_step
            put(gen_row, GEN_TOKENS_KEY, gen)
        train = None
        if train_row is not None and n in train_time:
            rate, duration = _finite(train_rate[n][1]), _finite(train_time[n][1])
            train = None if rate is None or duration is None else rate * duration
            put(train_row, TRAIN_TOKENS_KEY, train)
        total = None if gen is None and train is None else (gen or 0.0) + (train or 0.0)
        put(row, TOKENS_KEY, total)

        if not seconds or seconds <= 0:
            continue
        for key_tps, key_cost, tokens in (
            (GEN_TPS_KEY, GEN_COST_KEY, gen),
            (TRAIN_TPS_KEY, TRAIN_COST_KEY, train),
            (TPS_KEY, COST_KEY, total),
        ):
            if tokens is None:
                continue
            put(row, key_tps, tokens / seconds)
            if pricing.usd_per_s is not None and tokens > 0:
                put(row, key_cost, pricing.usd_per_s * seconds / tokens * 1e6)
        if pricing.usd_per_s is not None:
            put(row, STEP_COST_KEY, pricing.usd_per_s * seconds)
            put(row, GPU_HOUR_USD_KEY, pricing.gpu_hour_usd)
    return out


def changed_points(
    table: StepTable, points: Mapping[int, Mapping[str, float]]
) -> dict[int, dict[str, float]]:
    """The subset of ``points`` whose values differ from what ``table`` holds."""
    return {
        row: {
            key: value
            for key, value in metrics.items()
            if table.get(row, {}).get(key) != value
        }
        for row, metrics in points.items()
        if any(table.get(row, {}).get(key) != value for key, value in metrics.items())
    }


def derived_keys(keys: Iterable[str]) -> list[str]:
    """The ``dojo/`` keys among ``keys``, for callers that treat them apart."""
    return [key for key in keys if key.startswith(DERIVED_PREFIX)]
