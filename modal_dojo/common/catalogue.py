"""The recipe catalogue behind the dashboard's Autoconfig page.

Each catalogue entry is a public recipe benchmarked for the first ``STEPS``
training steps. The runs come from the dashboard's own metadata volume: a run
belongs to an entry when an Autoconfig sweep launched it (its group tags carry
``AUTOCONFIG_ENTRY_TAG``) or when its id is pinned on the entry. Each step is
priced at Modal's current GPU rates.
"""

from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import modal_dojo
from modal_dojo.common.metric_series import RunMetrics, StepTable, metric_series_store
from modal_dojo.common.models.validation import _ValidationConfig
from modal_dojo.common.run_summary import RunSummary, build_run_summary
from modal_dojo.common.step_timing import measured_run_times
from modal_dojo.train_recipes.gpu_allocation import resolve_gpu_allocation
from modal_dojo.utils.metadata import MetadataStore, vol_get, vol_list

STEPS = 3
REWARD_KEY = "rollout/raw_reward"
# Group ids of sweeps the Autoconfig page launches start with this.
AUTOCONFIG_GROUP_PREFIX = "autoconfig"
# Group-tag override naming the catalogue entry a sweep variant benchmarks.
AUTOCONFIG_ENTRY_TAG = "autoconfig.entry"
HF_URL = "https://huggingface.co"
DOCS_URL = "https://dojo.modal.dev"


@dataclass(frozen=True)
class CatalogueEntry:
    # ``VALIDATION_CONFIGS`` name, which resolves the model and its framework.
    name: str
    # Public recipe class the runs train with; links to its reference page.
    recipe: str
    # Runs pinned to the entry by ``training_run_id``, besides sweep runs.
    run_ids: tuple[str, ...] = ()


CATALOGUE: tuple[CatalogueEntry, ...] = (
    CatalogueEntry("Qwen3.8-27B", "Qwen3_8_27B_Recipe"),
    CatalogueEntry("Qwen3.6-35B-A3B", "Qwen3_6_35B_Recipe"),
    CatalogueEntry("GLM-5.3-Flash-LoRA", "GLM_5_3_Flash_LoRA_Recipe"),
    CatalogueEntry("DeepSeek-V4.1-Flash", "DeepSeek_V4_1_Flash_Recipe"),
    CatalogueEntry("Kimi-K3", "Kimi_K3_LoRA_Recipe"),
)


def find_entry(name: str) -> CatalogueEntry:
    """The catalogue entry named ``name``, case-insensitively."""
    wanted = name.strip().lower()
    for entry in CATALOGUE:
        if entry.name.lower() == wanted:
            return entry
    available = ", ".join(entry.name for entry in CATALOGUE)
    raise ValueError(f"unknown catalogue entry {name!r}; available: {available}")


def logged_per_step(table: StepTable, key: str) -> dict[int, float]:
    """A metric the framework logs once per step, numbered by logging order.

    Mirrored rows merge several logging calls, so the ``rollout/step`` in a
    row does not reliably number the value beside it; arrival order does.
    """
    values = [row[key] for _, row in sorted(table.items()) if key in row]
    return dict(enumerate(values, start=1))


def step_seconds(
    measured: Mapping[str, Mapping[str, Any]], table: StepTable
) -> dict[int, float]:
    """Duration of each completed training step, by step number.

    Prefers driver-measured substep timing, which excludes checkpoint saves
    and evals. Miles images without the timing patch only report the
    framework's own ``perf/step_time``.
    """
    timed = {
        int(step): float(times["duration_s"])
        for step, times in measured.items()
        if not times.get("partial") and times.get("duration_s")
    }
    return timed or logged_per_step(table, "perf/step_time")


def step_rows(
    seconds: Mapping[int, float],
    rewards: Mapping[int, float],
    gpus: int,
    gpu_hour_usd: float | None,
) -> list[dict[str, Any]]:
    """Time, GPU cost, running cost, and raw reward for each step so far."""
    rate = None if gpu_hour_usd is None else gpus * gpu_hour_usd / 3600
    rows: list[dict[str, Any]] = []
    total: float | None = 0.0
    for step in range(1, max([*seconds, *rewards], default=0) + 1):
        duration = seconds.get(step)
        cost = None if duration is None or rate is None else duration * rate
        # A step without a price leaves every later running total unknown.
        total = None if total is None or cost is None else total + cost
        reward = rewards.get(step)
        rows.append(
            {
                "step": step,
                "seconds": None if duration is None else round(duration, 1),
                "cost_usd": None if cost is None else round(cost, 2),
                "total_usd": None if total is None else round(total, 2),
                "raw_reward": None if reward is None else round(reward, 4),
            }
        )
    return rows


def context_length(recipe: Mapping[str, Any]) -> int | None:
    """Most tokens one response can use before it is cut off, prompt excluded.

    The response cap bounds every sample; a context cap, when the recipe sets
    one, bounds prompt plus response and can bind first.
    """
    extra = recipe.get("extra_config") or {}
    caps = [
        int(value)
        for key in ("rollout_max_response_len", "rollout_max_context_len")
        if (value := recipe.get(key) or extra.get(key))
    ]
    return min(caps) if caps else None


def recipe_defaults(recipe: str) -> dict[str, Any]:
    """Field defaults of a public recipe class, as a run config would record them."""
    return {
        field.name: field.default
        for field in dataclasses.fields(getattr(modal_dojo, recipe))
        if field.default is not dataclasses.MISSING
    }


def gpu_hour_cost(rates: Mapping[str, Any], gpu_type: str) -> float | None:
    """List price for one GPU-hour, from ``Workspace.billing.rates()``."""
    name = gpu_type.split(":")[0].strip().rstrip("!+").lower().replace("-", "_")
    value = rates.get(f"gpu_hour_cost_{name}")
    return None if value is None else float(value)


def run_row(
    *,
    run_id: str,
    recipe: Mapping[str, Any],
    gpu_type: str,
    model_name: str,
    created_at: int,
    status: str,
    dataset: str,
    seconds: Mapping[int, float],
    rewards: Mapping[int, float],
    rates: Mapping[str, Any],
    dashboard: str | None = None,
) -> dict[str, Any]:
    """The catalogue row fields one run supplies, from its recorded config and metrics.

    Everything describes the benchmark window, the first ``STEPS`` steps.
    """
    allocation = resolve_gpu_allocation(SimpleNamespace(**recipe), warn=False)
    seconds = {step: value for step, value in seconds.items() if step <= STEPS}
    rewards = {step: value for step, value in rewards.items() if step <= STEPS}
    hourly = gpu_hour_cost(rates, gpu_type)
    steps = step_rows(seconds, rewards, allocation.total_gpus, hourly)
    time_per_step = statistics.median(seconds.values()) if seconds else None
    cost_per_step = (
        allocation.total_gpus * hourly * time_per_step / 3600
        if hourly is not None and time_per_step is not None
        else None
    )
    run = {
        "id": run_id,
        "dashboard_url": (
            f"{(dashboard or '').rstrip('/')}/training/{quote(run_id, safe='')}"
        ),
        "created_at": created_at,
        "status": status,
        "dataset": dataset,
        "gpu_type": gpu_type,
        "gpus": allocation.total_gpus,
        "nodes": allocation.total_nodes,
        "gpu_hour_usd": hourly,
        "steps": steps,
        "steps_completed": len(steps),
        "time_per_step_s": None if time_per_step is None else round(time_per_step, 1),
        "cost_per_step_usd": None if cost_per_step is None else round(cost_per_step, 2),
        "initial_reward": steps[0]["raw_reward"] if steps else None,
    }
    return {"context_length": context_length(recipe), "run": run}


def run_fields(
    entry: CatalogueEntry,
    run_id: str,
    rates: Mapping[str, Any],
    dashboard: str | None = None,
) -> dict[str, Any]:
    """The catalogue row fields one run in the metadata volume supplies."""
    summary = build_run_summary(vol_get(MetadataStore.TRAINING_RUNS, run_id))
    # Run records hold the recipe's effective CLI flags, which omit fields the
    # framework never flags (slime's ``gpu_type``); the class defaults fill those.
    recipe = recipe_defaults(entry.recipe) | (summary.config.get("recipe") or {})
    gpu_type = str(recipe.get("gpu_type") or "")

    metrics = RunMetrics()
    for chunk in vol_list(metric_series_store(run_id)):
        metrics.load_chunk(chunk)
    measured, _ = measured_run_times(run_id)
    model_name = (summary.config.get("model") or {}).get("model_name") or summary.model
    return run_row(
        run_id=run_id,
        recipe=recipe,
        gpu_type=gpu_type,
        model_name=model_name,
        created_at=summary.created_at,
        status=summary.display_status,
        dataset=summary.dataset,
        seconds=step_seconds(measured, metrics.table),
        rewards=logged_per_step(metrics.table, REWARD_KEY),
        rates=rates,
        dashboard=dashboard,
    )


def sweep_entry_name(summary: RunSummary) -> str | None:
    """The catalogue entry an Autoconfig sweep run benchmarks, if it is one."""
    tags = summary.group_tags
    if tags is None or not summary.group_id.startswith(AUTOCONFIG_GROUP_PREFIX):
        return None
    name = tags.overrides.get(AUTOCONFIG_ENTRY_TAG)
    return str(name) if name else None


def entry_run_ids(entry: CatalogueEntry, summaries: Iterable[RunSummary]) -> list[str]:
    """Pinned runs first, then the entry's sweep runs, newest first."""
    sweep_runs = sorted(
        (s for s in summaries if sweep_entry_name(s) == entry.name),
        key=lambda s: s.created_at,
        reverse=True,
    )
    ids = list(entry.run_ids)
    ids.extend(s.training_run_id for s in sweep_runs if s.training_run_id not in ids)
    return ids


def entry_rows(
    entry: CatalogueEntry,
    run_ids: Iterable[str],
    rates: Mapping[str, Any],
    dashboard: str | None = None,
) -> list[dict[str, Any]]:
    """One row per run of the entry, or a pending row when it has none."""
    config = _ValidationConfig.find(entry.name)
    base = {
        "name": entry.name,
        "model": config.model_name,
        "model_href": f"{HF_URL}/{config.model_name}",
        "framework": config.framework.value,
        "recipe": entry.recipe,
        "recipe_href": f"{DOCS_URL}/reference/{entry.recipe.lower()}",
    }
    run_ids = list(run_ids)
    if not run_ids:
        # Until it runs, a recipe's context length follows its class defaults.
        planned = context_length(recipe_defaults(entry.recipe))
        return [{**base, "context_length": planned, "run": None}]
    return [
        {**base, **run_fields(entry, run_id, rates, dashboard)} for run_id in run_ids
    ]


def build_catalogue(
    summaries: Iterable[RunSummary],
    rates: Mapping[str, Any],
    priced_at: str | None,
    dashboard: str | None = None,
) -> dict[str, Any]:
    """The Autoconfig page's payload: every entry, with the runs it has so far."""
    summaries = list(summaries)
    return {
        "steps": STEPS,
        "priced_at": priced_at,
        "entries": [
            {"name": entry.name, "recipe": entry.recipe} for entry in CATALOGUE
        ],
        "rows": [
            row
            for entry in CATALOGUE
            for row in entry_rows(
                entry, entry_run_ids(entry, summaries), rates, dashboard
            )
        ],
    }
