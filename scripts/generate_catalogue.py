"""Generate the data behind the recipe catalogue at dojo.modal.dev/catalogue.

Each catalogue entry is a recipe trained for three steps on SWE-bench. The runs
live in the metadata volume of the Modal environment that launched them; this
script reads them, prices each step at Modal's current GPU rates, and writes
``docs-next/src/generated/catalogue.json`` for the Astro page to render.

    uv run scripts/generate_catalogue.py
    uv run scripts/generate_catalogue.py --modal-env training-gym --output /tmp/catalogue.json

Add a finished run to the catalogue by appending its ``training_run_id`` to
the entry's ``run_ids`` below and rerunning the script.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import functools
import json
import os
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import modal_dojo
from modal_dojo.common.dashboard import (
    LEGACY_DASHBOARD_APP_NAME,
    deployed_dashboard_url,
)
from modal_dojo.common.metric_series import RunMetrics, StepTable, metric_series_store
from modal_dojo.common.models.validation import _ValidationConfig
from modal_dojo.common.run_summary import build_run_summary
from modal_dojo.common.step_timing import measured_run_times
from modal_dojo.train_recipes.gpu_allocation import resolve_gpu_allocation
from modal_dojo.utils.metadata import MetadataStore, vol_get, vol_list

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs-next" / "src" / "generated" / "catalogue.json"
DEFAULT_MODAL_ENV = "autotrain"
BENCHMARK = "SWE-bench"
STEPS = 3
REWARD_KEY = "rollout/raw_reward"
# The length keys sglang reads from a Hugging Face config, in its order.
CONTEXT_LENGTH_KEYS = (
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
)
HF_URL = "https://huggingface.co"


@dataclass(frozen=True)
class CatalogueEntry:
    # ``VALIDATION_CONFIGS`` name, which resolves the model and its framework.
    name: str
    # Public recipe class the runs trained with; links to its reference page.
    recipe: str
    # SWE-bench runs of this recipe, by ``training_run_id``.
    run_ids: tuple[str, ...] = ()


CATALOGUE: tuple[CatalogueEntry, ...] = (
    CatalogueEntry("Qwen3.6-27B", "Qwen3_6_27B_Recipe"),
    CatalogueEntry("Qwen3.8-27B", "Qwen3_8_27B_Recipe"),
    CatalogueEntry("Qwen3.6-35B-A3B", "Qwen3_6_35B_Recipe"),
    CatalogueEntry("GLM-5.3-Flash-LoRA", "GLM_5_3_Flash_LoRA_Recipe"),
    CatalogueEntry("DeepSeek-V4.1-Flash", "DeepSeek_V4_1_Flash_Recipe"),
    CatalogueEntry("Kimi-K3", "Kimi_K3_LoRA_Recipe"),
)


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


def native_context_length(config: Mapping[str, Any]) -> int | None:
    """The context window SGLang derives from a model's Hugging Face config.

    Mirrors sglang's ``get_context_length`` on the language-model sub-config:
    the first length key the config sets, stretched by a plain rope-scaling
    factor. YaRN-style scaling that records its original window, and llama3
    scaling, already state the stretched length.
    """
    text = next(
        (
            config[attr]
            for attr in ("text_config", "llm_config", "language_config")
            if isinstance(config.get(attr), dict)
        ),
        None,
    )
    if text is None:
        thinker = config.get("thinker_config")
        text = (
            thinker.get("text_config", config) if isinstance(thinker, dict) else config
        )
    rope = text.get("rope_scaling") or {}
    stretches = (
        rope
        and "original_max_position_embeddings" not in rope
        and rope.get("rope_type") != "llama3"
    )
    factor = rope.get("factor", 1) if stretches else 1
    for key in CONTEXT_LENGTH_KEYS:
        if text.get(key) is not None:
            return int(factor * text[key])
    return None


@functools.cache
def model_context_length(model_name: str) -> int | None:
    """Default context window of a Hugging Face model, from its config.json."""
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(model_name, "config.json")) as config:
        return native_context_length(json.load(config))


def context_length(
    recipe: Mapping[str, Any], model_name: str
) -> tuple[int | None, str | None]:
    """Longest sequence, prompt plus response, a recipe rolls out, and its source.

    A rollout cap or SGLang context length set by the recipe wins ("recipe");
    otherwise SGLang serves the model's own context window ("model").
    """
    extra = recipe.get("extra_config") or {}
    for key in ("rollout_max_context_len", "sglang_context_length"):
        value = recipe.get(key) or extra.get(key)
        if value:
            return int(value), "recipe"
    native = model_context_length(model_name)
    return native, None if native is None else "model"


def recipe_defaults(recipe: str) -> dict[str, Any]:
    """Field defaults of a public recipe class, as a run config would record them."""
    return {
        field.name: field.default
        for field in dataclasses.fields(getattr(modal_dojo, recipe))
        if field.default is not dataclasses.MISSING
    }


@functools.cache
def dashboard_url(environment: str | None) -> str | None:
    """Web URL of the dashboard serving ``environment``, under either app name.

    The lookup reads the active ``MODAL_ENVIRONMENT``; the argument only keys
    the cache, so each environment is looked up once.
    """
    return deployed_dashboard_url() or deployed_dashboard_url(LEGACY_DASHBOARD_APP_NAME)


def gpu_hour_cost(rates: Mapping[str, Any], gpu_type: str) -> float | None:
    """List price for one GPU-hour, from ``Workspace.billing.rates()``."""
    name = gpu_type.split(":")[0].strip().rstrip("!+").lower().replace("-", "_")
    value = rates.get(f"gpu_hour_cost_{name}")
    return None if value is None else float(value)


def run_fields(run_id: str, rates: Mapping[str, Any]) -> dict[str, Any]:
    """The catalogue row fields one finished run supplies."""
    summary = build_run_summary(vol_get(MetadataStore.TRAINING_RUNS, run_id))
    recipe = summary.config.get("recipe") or {}
    allocation = resolve_gpu_allocation(SimpleNamespace(**recipe), warn=False)
    gpu_type = str(recipe.get("gpu_type") or "")

    metrics = RunMetrics()
    for chunk in vol_list(metric_series_store(run_id)):
        metrics.load_chunk(chunk)
    measured, _ = measured_run_times(run_id)
    # Everything below describes the benchmark window, the first STEPS steps.
    seconds = {
        step: value
        for step, value in step_seconds(measured, metrics.table).items()
        if step <= STEPS
    }
    rewards = {
        step: value
        for step, value in logged_per_step(metrics.table, REWARD_KEY).items()
        if step <= STEPS
    }
    hourly = gpu_hour_cost(rates, gpu_type)
    steps = step_rows(seconds, rewards, allocation.total_gpus, hourly)
    time_per_step = statistics.median(seconds.values()) if seconds else None
    cost_per_step = (
        allocation.total_gpus * hourly * time_per_step / 3600
        if hourly is not None and time_per_step is not None
        else None
    )
    dashboard = dashboard_url(os.environ.get("MODAL_ENVIRONMENT"))
    run = {
        "id": run_id,
        "dashboard_url": (
            f"{dashboard.rstrip('/')}/training/{quote(run_id, safe='')}"
            if dashboard
            else None
        ),
        "created_at": summary.created_at,
        "status": summary.display_status,
        "dataset": summary.dataset,
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
    model_name = (summary.config.get("model") or {}).get("model_name") or summary.model
    context, source = context_length(recipe, model_name)
    return {"context_length": context, "context_source": source, "run": run}


def entry_rows(entry: CatalogueEntry, rates: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = _ValidationConfig.find(entry.name)
    base = {
        "name": entry.name,
        "model": config.model_name,
        "model_href": f"{HF_URL}/{config.model_name}",
        "framework": config.framework.value,
        "recipe": entry.recipe,
        "recipe_href": f"/reference/{entry.recipe.lower()}",
    }
    if not entry.run_ids:
        # Until it runs, a recipe's context length follows its class defaults.
        context, source = context_length(
            recipe_defaults(entry.recipe), config.model_name
        )
        return [
            {**base, "context_length": context, "context_source": source, "run": None}
        ]
    return [{**base, **run_fields(run_id, rates)} for run_id in entry.run_ids]


def build_catalogue(rates: Mapping[str, Any], priced_at: str | None) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK,
        "steps": STEPS,
        "priced_at": priced_at,
        "rows": [row for entry in CATALOGUE for row in entry_rows(entry, rates)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--modal-env",
        default=DEFAULT_MODAL_ENV,
        help="Modal environment whose metadata volume holds the runs.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    # Modal reads MODAL_ENVIRONMENT on each lookup, so this scopes every
    # metadata-volume read below to the environment that ran the catalogue.
    os.environ["MODAL_ENVIRONMENT"] = args.modal_env

    rates: Mapping[str, Any] = {}
    priced_at = None
    if any(entry.run_ids for entry in CATALOGUE):
        import modal

        rates = modal.Workspace.from_context().billing.rates()
        priced_at = dt.date.today().isoformat()

    catalogue = build_catalogue(rates, priced_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalogue, indent=2) + "\n")
    runs = sum(row["run"] is not None for row in catalogue["rows"])
    print(f"Wrote {args.output} ({len(CATALOGUE)} recipes, {runs} runs)")


if __name__ == "__main__":
    main()
