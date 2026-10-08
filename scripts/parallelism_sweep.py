"""Offline EP/PP parallelism sweeps for a model's base recipe.

``launch`` expands a parallelism grid over the recipe that
``scripts/validate_model_configs.py`` trains for a model, drops the grid
points the recipe validators reject, and launches the rest as one detached
``TrainingGroup``. ``collect`` turns the group's timing records into
per-point phase durations and tokens/s/GPU so layouts can be ranked::

    uv run scripts/parallelism_sweep.py launch -m Qwen3.6-35B-A3B --dry-run
    uv run scripts/parallelism_sweep.py launch -m Qwen3.6-35B-A3B --preset ep-pp \\
        --max-gpus 16 --max-points 6
    uv run scripts/parallelism_sweep.py collect --group ep-pp-qwen3-6-35b-a3b \\
        --out sweep.md

Grid presets:

* ``ep-pp`` — ``expert_model_parallel_size`` × ``pipeline_model_parallel_size``
  over powers of two that fit the actor world size and the model's expert
  count. Axes the recipe does not expose (slime recipes without a PP field)
  are dropped and reported.
* ``rollout`` — ``sglang_ep_size`` × ``sglang_dp_size`` over divisors of the
  rollout engine size.

Custom axes stack on either preset with ``--grid recipe.<field>=v1,v2``;
``--set recipe.<field>=v`` changes the base every point shares (for example
``--set recipe.actor_num_gpus_per_node=8`` to size the actor world).
Step 1 is a warm-up and excluded from every per-step metric.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from validation_backends import build_recipe_and_dataset
except ImportError:  # pragma: no cover — imported as ``scripts.parallelism_sweep``
    from scripts.validation_backends import build_recipe_and_dataset

from modal_dojo.common.models.validation import _ValidationConfig
from modal_dojo.common.run import TrainingRun, TrainingRunStatus
from modal_dojo.common.step_timing import measured_run_times
from modal_dojo.common.train import TrainConfig
from modal_dojo.common.training_group import TrainingGroup
from modal_dojo.train_recipes.gpu_allocation import (
    GpuAllocation,
    resolve_gpu_allocation,
)
from modal_dojo.utils.metadata import MetadataStore, vol_get, vol_get_summary_items

DEFAULT_STEPS = 3
WARMUP_STEPS = 1
PRESETS = ("ep-pp", "rollout")

EP_FIELD = "recipe.expert_model_parallel_size"
PP_FIELD = "recipe.pipeline_model_parallel_size"
SGLANG_EP_FIELD = "recipe.sglang_ep_size"
SGLANG_DP_FIELD = "recipe.sglang_dp_size"

# Driver phases reported per point; the rest of the step is summed into
# ``other_s`` so the columns add up to ``step_time_s``.
REPORTED_PHASES = (
    "generate_rollouts",
    "train_models",
    "compute_log_probs",
    "forward_backward",
    "onload_train",
    "offload_rollout",
    "onload_rollout_weights",
    "offload_train_gradients",
)

_OOM_PATTERN = re.compile(r"out of memory|OutOfMemoryError|\bOOM\b", re.IGNORECASE)


# ── Grid construction ───────────────────────────────────────────────────────


def _powers_of_two_up_to(limit: int) -> list[int]:
    values: list[int] = []
    value = 1
    while value <= limit:
        values.append(value)
        value *= 2
    return values


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _recipe_fields(recipe: Any) -> set[str]:
    return {f.name for f in dataclasses.fields(recipe)}


def _actor_world_size(recipe: Any) -> int:
    return int(recipe.actor_num_nodes) * int(recipe.actor_num_gpus_per_node)


def preset_grid(
    preset: str, recipe: Any, model: Any
) -> tuple[dict[str, list[Any]], dict[str, str]]:
    """The grid for ``preset`` on ``recipe``, plus axes dropped with reasons.

    Candidate values are generous on purpose: the recipe validators decide
    which combinations are legal when the group expands with
    ``skip_invalid=True``.
    """
    fields = _recipe_fields(recipe)
    grid: dict[str, list[Any]] = {}
    dropped: dict[str, str] = {}

    def axis(path: str, values: list[Any]) -> None:
        field = path.split(".", 1)[1]
        if field not in fields:
            dropped[path] = f"{type(recipe).__name__} has no field {field!r}"
            return
        current = getattr(recipe, field)
        if current is not None and current not in values:
            values = sorted({*values, current})
        grid[path] = values

    if preset == "ep-pp":
        world = _actor_world_size(recipe)
        architecture = getattr(model, "architecture", None)
        num_experts = int(getattr(architecture, "num_experts", 0) or 0)
        if num_experts:
            ep_values = [
                ep for ep in _powers_of_two_up_to(world) if num_experts % ep == 0
            ]
        else:
            ep_values = [1]
            dropped[EP_FIELD] = f"{type(model).__name__} is dense (num_experts=0)"
        if EP_FIELD not in dropped:
            axis(EP_FIELD, ep_values)
        axis(PP_FIELD, _powers_of_two_up_to(world))
    elif preset == "rollout":
        engine = int(recipe.rollout_num_gpus_per_engine)
        axis(SGLANG_EP_FIELD, _divisors(engine))
        axis(SGLANG_DP_FIELD, _divisors(engine))
    else:
        raise ValueError(f"unknown preset {preset!r}; choose from {PRESETS}")
    return grid, dropped


def parse_set_arg(spec: str) -> tuple[str, Any]:
    """``recipe.field=8`` → ``("recipe.field", 8)``."""
    path, values = parse_grid_arg(spec)
    if len(values) != 1:
        raise argparse.ArgumentTypeError(f"--set takes one value, got {spec!r}")
    return path, values[0]


def parse_grid_arg(spec: str) -> tuple[str, list[Any]]:
    """``recipe.field=1,2,4`` → ``("recipe.field", [1, 2, 4])``."""
    path, sep, raw = spec.partition("=")
    if not sep or not path.strip() or not raw.strip():
        raise argparse.ArgumentTypeError(
            f"expected recipe.<field>=v1,v2 but got {spec!r}"
        )
    values: list[Any] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            try:
                values.append(float(token))
            except ValueError:
                values.append(
                    {"true": True, "false": False, "none": None}.get(
                        token.lower(), token
                    )
                )
    return path.strip(), values


# ── Launch ──────────────────────────────────────────────────────────────────


def build_base_config(model_name: str, *, steps: int) -> tuple[Any, TrainConfig]:
    """The validation-harness base config for ``model_name``."""
    config = _ValidationConfig.find(model_name)
    model = config.model_config()
    recipe, dataset = build_recipe_and_dataset(
        config.framework, model, steps, loss_type=config.loss_type
    )
    recipe.num_rollout = steps
    return config, TrainConfig(model=model, dataset=dataset, recipe=recipe)


def _group_name(preset: str, model_name: str, explicit: str | None) -> str:
    return explicit or f"{preset}-{model_name}"


@dataclasses.dataclass
class SweepPoint:
    overrides: dict[str, Any]
    config: TrainConfig
    allocation: GpuAllocation


def plan_sweep(
    group: TrainingGroup, *, max_gpus: int | None, max_points: int | None
) -> tuple[list[SweepPoint], list[tuple[dict[str, Any], str]]]:
    """Valid points to launch, and every point dropped with its reason."""
    variants = group.iter_variants()
    kept: list[SweepPoint] = []
    dropped: list[tuple[dict[str, Any], str]] = list(group.skipped)
    for overrides, cfg in variants:
        allocation = resolve_gpu_allocation(cfg.recipe, warn=False)
        if max_gpus is not None and allocation.total_gpus > max_gpus:
            dropped.append(
                (
                    overrides,
                    f"needs {allocation.total_gpus} GPUs > --max-gpus {max_gpus}",
                )
            )
            continue
        kept.append(SweepPoint(overrides, cfg, allocation))
    if max_points is not None and len(kept) > max_points:
        for point in kept[max_points:]:
            dropped.append((point.overrides, f"beyond --max-points {max_points}"))
        kept = kept[:max_points]
    return kept, dropped


def apply_fixed_overrides(base: TrainConfig, fixed: dict[str, Any]) -> TrainConfig:
    """``base`` with ``--set`` values applied and the recipe revalidated."""
    if not fixed:
        return base
    cfg = TrainingGroup(
        base, grid={k: [v] for k, v in fixed.items()}
    ).get_train_configs()[0]
    cfg.group_id = None
    cfg.group_overrides = None
    cfg.group_axes = None
    return cfg


def _fmt_overrides(overrides: dict[str, Any]) -> str:
    return ", ".join(
        f"{path.split('.')[-1]}={value}" for path, value in overrides.items()
    )


def cmd_launch(args: argparse.Namespace) -> int:
    config, base = build_base_config(args.model, steps=args.steps)
    base = apply_fixed_overrides(base, dict(args.set or []))
    grid, dropped_axes = preset_grid(args.preset, base.recipe, base.model)
    for path, values in args.grid or []:
        grid[path] = values
    for path, reason in dropped_axes.items():
        print(f"axis {path} dropped: {reason}")
    if not grid:
        print("nothing to sweep: every preset axis was dropped", file=sys.stderr)
        return 2

    group = TrainingGroup(
        base,
        grid=grid,
        name=_group_name(args.preset, config.name, args.name),
        skip_invalid=True,
    )
    points, dropped = plan_sweep(
        group, max_gpus=args.max_gpus, max_points=args.max_points
    )

    print(f"group {group.group_id}: {len(points)} point(s) to launch")
    for point in points:
        print(
            f"  launch  {_fmt_overrides(point.overrides)}  [{point.allocation.summary()}]"
        )
    for overrides, reason in dropped:
        print(f"  drop    {_fmt_overrides(overrides)}  — {reason}")
    if not points:
        return 2
    if args.dry_run:
        return 0

    launched = _launch_points(group, points)
    print(f"launched {len(launched)} run(s) in group {group.group_id}:")
    for run in launched:
        print(f"  {run.training_run_id}  {run.modal_app_url}")
    print(
        f"collect with: uv run scripts/parallelism_sweep.py collect "
        f"--group {group.group_id}"
    )
    return 0 if len(launched) == len(points) else 1


def _launch_points(group: TrainingGroup, points: list[SweepPoint]) -> list[TrainingRun]:
    """Launch only the planned points through the group's bookkeeping."""
    group._variants = [(p.overrides, p.config) for p in points]
    return group.launch(continue_on_error=True)


# ── Collect ─────────────────────────────────────────────────────────────────


def classify_run(run: TrainingRun) -> dict[str, Any]:
    """Status plus a coarse failure class: ``oom``, ``preempted`` or ``error``."""
    metadata = run.metadata or {}
    attempts = int(metadata.get("attempt_count") or 1)
    failure_class: str | None = None
    if run.status is TrainingRunStatus.FAILED:
        if run.error_message and _OOM_PATTERN.search(run.error_message):
            failure_class = "oom"
        elif attempts > 1:
            failure_class = "preempted"
        else:
            failure_class = "error"
    return {
        "status": run.status.value,
        "failure_class": failure_class,
        "attempts": attempts,
        "error": (run.error_message or "").strip().splitlines()[:1],
    }


def _measured_steps(
    step_times: dict[str, dict[str, Any]], *, warmup_steps: int = WARMUP_STEPS
) -> list[str]:
    return sorted(
        (
            key
            for key, step in step_times.items()
            if key.isdigit() and int(key) > warmup_steps and not step.get("partial")
        ),
        key=int,
    )


def _mean(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values]
    return round(statistics.fmean(values), 3) if values else None


def derive_timing(
    step_times: dict[str, dict[str, Any]],
    substep_times: dict[str, dict[str, dict[str, Any]]],
    *,
    warmup_steps: int = WARMUP_STEPS,
) -> dict[str, Any]:
    """Mean step and driver-phase durations over complete post-warm-up steps."""
    steps = _measured_steps(step_times, warmup_steps=warmup_steps)
    phases: dict[str, float | None] = {}
    for phase in REPORTED_PHASES:
        values = [
            substep_times.get(step, {}).get(phase, {}).get("duration_s")
            for step in steps
        ]
        values = [v for v in values if v is not None]
        phases[phase] = _mean(values) if len(values) == len(steps) and steps else None
    step_time = _mean(step_times[step]["duration_s"] for step in steps)
    reported = sum(v for v in phases.values() if v is not None)
    return {
        "measured_steps": [int(s) for s in steps],
        "step_time_s": step_time,
        "phase_times_s": phases,
        "other_s": round(step_time - reported, 3) if step_time is not None else None,
    }


def rollout_tokens(payload: dict[str, Any]) -> tuple[int, int] | None:
    """``(tokens_in, tokens_out)`` summed over a rollout's samples, if recorded."""
    tokens_in = tokens_out = 0
    seen = False
    for sample in payload.get("samples") or []:
        inference = (sample.get("metadata") or {}).get("inference")
        if not isinstance(inference, dict):
            continue
        seen = True
        tokens_in += int(inference.get("tokens_in") or 0)
        tokens_out += int(inference.get("tokens_out") or 0)
    return (tokens_in, tokens_out) if seen else None


def derive_throughput(
    tokens_per_step: dict[int, tuple[int, int]],
    timing: dict[str, Any],
    allocation: GpuAllocation | None,
) -> dict[str, Any]:
    """tokens/s/GPU over the measured steps; ``None`` where inputs are missing."""
    steps = [s for s in timing["measured_steps"] if s in tokens_per_step]
    if not steps or timing["step_time_s"] is None or allocation is None:
        return {
            "tokens_in_per_step": None,
            "tokens_out_per_step": None,
            "tokens_per_s_per_gpu": None,
            "gen_tokens_per_s_per_rollout_gpu": None,
        }
    tokens_in = statistics.fmean(tokens_per_step[s][0] for s in steps)
    tokens_out = statistics.fmean(tokens_per_step[s][1] for s in steps)
    total = tokens_in + tokens_out
    generate_s = timing["phase_times_s"].get("generate_rollouts")
    rollout_gpus = allocation.rollout_gpus or allocation.total_gpus
    return {
        "tokens_in_per_step": round(tokens_in),
        "tokens_out_per_step": round(tokens_out),
        "tokens_per_s_per_gpu": round(
            total / timing["step_time_s"] / allocation.total_gpus, 2
        ),
        "gen_tokens_per_s_per_rollout_gpu": (
            round(tokens_out / generate_s / rollout_gpus, 2) if generate_s else None
        ),
    }


def group_run_ids(group_id: str) -> list[str]:
    items = vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY) or []
    run_ids: list[str] = []
    for item in items:
        try:
            run = TrainingRun.model_validate(item)
        except Exception:  # noqa: BLE001 — summary rows are best-effort
            continue
        if run.group_id == group_id:
            run_ids.append(run.training_run_id)
    return sorted(run_ids)


def _run_overrides(run: TrainingRun) -> dict[str, Any]:
    tags = (run.metadata or {}).get("group_tags") or {}
    overrides = tags.get("overrides") if isinstance(tags, dict) else None
    return dict(overrides) if isinstance(overrides, dict) else {}


def _run_allocation(
    run: TrainingRun, overrides: dict[str, Any], *, model_name: str | None
) -> GpuAllocation | None:
    """Re-derive the launched recipe from the run's model and overrides."""
    name = model_name or (run.config or {}).get("model", {}).get("model_name")
    if not name:
        return None
    try:
        _, base = build_base_config(name, steps=DEFAULT_STEPS)
        group = TrainingGroup(base, grid={k: [v] for k, v in overrides.items()})
        ((_, cfg),) = group.iter_variants()
        return resolve_gpu_allocation(cfg.recipe, warn=False)
    except Exception as exc:  # noqa: BLE001 — report the point without GPU math
        print(f"  {run.training_run_id}: cannot rebuild recipe: {exc}", file=sys.stderr)
        return None


def collect_run(
    run: TrainingRun, *, warmup_steps: int = WARMUP_STEPS, model_name: str | None = None
) -> dict[str, Any]:
    overrides = _run_overrides(run)
    step_times, substep_times = measured_run_times(run.training_run_id)
    timing = derive_timing(step_times, substep_times, warmup_steps=warmup_steps)
    tokens: dict[int, tuple[int, int]] = {}
    for step in timing["measured_steps"]:
        key = f"{run.training_run_id}__{step - 1:08d}"
        try:
            payload = vol_get(MetadataStore.TRAINING_ROLLOUTS, key)
        except KeyError:
            continue
        if counted := rollout_tokens(payload):
            tokens[step] = counted
    allocation = _run_allocation(run, overrides, model_name=model_name)
    return {
        "training_run_id": run.training_run_id,
        "group_id": run.group_id,
        "modal_app_url": run.modal_app_url,
        "overrides": overrides,
        "total_gpus": allocation.total_gpus if allocation else None,
        **classify_run(run),
        **timing,
        **derive_throughput(tokens, timing, allocation),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_markdown(group_id: str, points: list[dict[str, Any]]) -> str:
    axes: list[str] = []
    for point in points:
        for path in point["overrides"]:
            if path not in axes:
                axes.append(path)
    header = (
        [path.split(".")[-1] for path in axes]
        + ["status", "gpus", "steps", "step s"]
        + [f"{phase} s" for phase in REPORTED_PHASES]
        + ["other s", "tok/s/gpu", "gen tok/s/gpu", "run"]
    )
    ranked = sorted(
        points,
        key=lambda p: (p["step_time_s"] is None, p["step_time_s"] or 0.0),
    )
    rows = []
    for point in ranked:
        status = point["status"]
        if point["failure_class"]:
            status += f" ({point['failure_class']})"
        rows.append(
            [_fmt(point["overrides"].get(path)) for path in axes]
            + [
                status,
                _fmt(point["total_gpus"]),
                ",".join(map(str, point["measured_steps"])) or "—",
                _fmt(point["step_time_s"]),
            ]
            + [_fmt(point["phase_times_s"].get(phase)) for phase in REPORTED_PHASES]
            + [
                _fmt(point["other_s"]),
                _fmt(point["tokens_per_s_per_gpu"]),
                _fmt(point["gen_tokens_per_s_per_rollout_gpu"]),
                point["training_run_id"],
            ]
        )
    lines = [
        f"## Parallelism sweep `{group_id}`",
        "",
        f"{len(points)} point(s); steps after the first {WARMUP_STEPS} warm-up "
        "step(s) are averaged, partial steps excluded.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def cmd_collect(args: argparse.Namespace) -> int:
    run_ids = list(args.run or [])
    if args.group:
        run_ids += [r for r in group_run_ids(args.group) if r not in run_ids]
    if not run_ids:
        print("no runs found; pass --group or --run", file=sys.stderr)
        return 2
    points = []
    for run_id in run_ids:
        run = TrainingRun.from_id(run_id)
        points.append(
            collect_run(run, warmup_steps=args.warmup_steps, model_name=args.model)
        )
    group_id = args.group or (points[0].get("group_id") or "runs")
    markdown = render_markdown(group_id, points)
    if args.out:
        out = Path(args.out)
        if out.suffix == ".json":
            out.write_text(
                json.dumps({"group_id": group_id, "points": points}, indent=2) + "\n"
            )
        else:
            out.write_text(markdown)
        print(f"wrote {out}")
    print(markdown)
    return 0


# ── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="expand a parallelism grid and launch it")
    launch.add_argument("-m", "--model", required=True, help="validation registry name")
    launch.add_argument("--preset", choices=PRESETS, default="ep-pp")
    launch.add_argument(
        "--grid",
        action="append",
        type=parse_grid_arg,
        metavar="recipe.FIELD=V1,V2",
        help="extra or replacement axis; repeatable",
    )
    launch.add_argument(
        "--set",
        action="append",
        type=parse_set_arg,
        metavar="recipe.FIELD=V",
        help="fixed override applied to every point, e.g. actor GPU count",
    )
    launch.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    launch.add_argument("--max-gpus", type=int, help="drop points needing more GPUs")
    launch.add_argument("--max-points", type=int, help="launch at most this many")
    launch.add_argument("--name", help="group id; default <preset>-<model>")
    launch.add_argument("--dry-run", action="store_true")
    launch.set_defaults(func=cmd_launch)

    collect = sub.add_parser("collect", help="summarize a launched sweep")
    collect.add_argument("--group", help="TrainingGroup id")
    collect.add_argument("--run", action="append", help="run id; repeatable")
    collect.add_argument("-m", "--model", help="registry name when not on the run")
    collect.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    collect.add_argument("--out", help="write .json or markdown here")
    collect.set_defaults(func=cmd_collect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
