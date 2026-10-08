"""``modal-dojo autoconfig``: shape and launch recipe sweeps from an intent.

The commands an agent needs to turn "a 27B Qwen with 32k context on SWE-bench"
into runs: list the models and their recipe knobs, dry-run a plan with the GPU
memory estimate per variant, submit it as a sweep, and poll the launch.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

import click
from rich.text import Text

from modal_dojo._autoconfig import (
    DATASET_PRESETS,
    HIGHLIGHTED_KNOBS,
    SweepRequest,
    describe_plan,
    list_models,
    recipe_knobs,
    resolve_dataset,
)

from .client import DashboardClient
from .commands import _DojoGroup
from .errors import CLIError, ExitCode
from .options import json_option
from .output import print_json, print_table

POLL_INTERVAL_SECONDS = 5.0
_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*=\s*(.*)$", re.DOTALL)
_TOKENS = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*$")


def parse_value(text: str) -> Any:
    """A flag value: JSON when it parses (``1e-6``, ``true``, ``[1,2]``), else text."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse_assignment(text: str) -> tuple[str, str]:
    match = _ASSIGNMENT.match(text)
    if not match:
        raise click.BadParameter(f"expected PATH=VALUE, got {text!r}")
    return match.group(1), match.group(2)


def parse_sets(values: tuple[str, ...]) -> dict[str, Any]:
    """``--set recipe.gpu_type=B300`` flags to base overrides."""
    return {path: parse_value(raw) for path, raw in map(parse_assignment, values)}


def parse_grid(values: tuple[str, ...]) -> dict[str, list[Any]]:
    """``--grid recipe.lr=1e-6,5e-6`` flags to a ``TrainingGroup`` grid."""
    grid: dict[str, list[Any]] = {}
    for path, raw in map(parse_assignment, values):
        choices = [parse_value(part) for part in raw.split(",") if part.strip()]
        if not choices:
            raise click.BadParameter(f"--grid {path} needs at least one value")
        grid[path] = choices
    return grid


def parse_tokens(text: str | None) -> int | None:
    """A context length such as ``32k``, ``32768``, or ``1m`` in tokens."""
    if text is None:
        return None
    match = _TOKENS.match(text)
    if not match:
        raise click.BadParameter(f"expected a token count such as 32k, got {text!r}")
    scale = {"": 1, "k": 1024, "m": 1024 * 1024}[match.group(2).lower()]
    return int(float(match.group(1)) * scale)


def build_request(
    *,
    models: tuple[str, ...],
    steps: int,
    context: str | None,
    dataset: str,
    split: str | None,
    input_column: str | None,
    output_column: str | None,
    sets: tuple[str, ...],
    grid: tuple[str, ...],
    name: str | None,
    allow_oom: bool,
) -> SweepRequest:
    try:
        spec = resolve_dataset(
            dataset, split=split, input_column=input_column, output_column=output_column
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--dataset") from exc
    return SweepRequest(
        entries=list(models),
        steps=steps,
        context_length=parse_tokens(context),
        dataset=spec,
        overrides=parse_sets(sets),
        grid=parse_grid(grid),
        name=name,
        allow_oom=allow_oom,
    )


def plan_or_fail(request: SweepRequest) -> dict[str, Any]:
    try:
        return describe_plan(request)
    except ValueError as exc:
        raise CLIError(
            str(exc),
            error="invalid_sweep",
            exit_code=ExitCode.USAGE,
            hint="modal-dojo autoconfig knobs <model>",
        ) from exc


def _fmt(value: Any) -> str:
    return "—" if value is None else str(value)


def _memory_cell(memory: dict[str, Any] | None) -> Text:
    if memory is None:
        return Text("—")
    label = f"~{memory['peak_gib']:g} / {memory['gpu_gib']:g} GiB"
    return Text(label, style="green" if memory["fits"] else "bold red")


def print_plan(plan: dict[str, Any]) -> None:
    dataset = plan["dataset"]
    print_table(
        ["Model", "Recipe", "Overrides", "GPUs", "Context", "Est. memory"],
        [
            (
                v["entry"],
                v["recipe"],
                ", ".join(f"{k}={json.dumps(val)}" for k, val in v["overrides"].items())
                or "(defaults)",
                f"{v['gpus']}x {v['gpu_type']}"
                + (f" ({v['nodes']} nodes)" if v["nodes"] > 1 else ""),
                _fmt(v["context_length"]),
                _memory_cell(v["memory"]),
            )
            for v in plan["variants"]
        ],
        title=(
            f"{len(plan['variants'])} run(s) x {plan['steps']} steps on "
            f"{dataset['hf_repo']} [{dataset['hf_split']}]"
        ),
    )
    if not plan["fits"]:
        click.echo(
            f"{plan['oom_variants']} variant(s) exceed GPU memory by estimate; "
            "change the shape or pass --allow-oom.",
            err=True,
        )


# ── Commands ─────────────────────────────────────────────────────────────────


@click.group("autoconfig", cls=_DojoGroup)
def autoconfig_group() -> None:
    """Shape and launch recipe sweeps."""


@autoconfig_group.command(
    "models",
    help="List the models a sweep can benchmark, with their recipe defaults.",
    epilog=(
        "Examples:\n"
        "  modal-dojo autoconfig models\n"
        "  modal-dojo autoconfig models --json | jq '.[] | select(.name | test(\"27B\"))'"
    ),
)
@click.option(
    "--history/--no-history",
    default=True,
    help="Join catalogue runs from the dashboard ($/step, s/step, reward).",
)
@json_option
def models_command(*, history: bool, json_output: bool) -> None:
    models = list_models()
    if history:
        for model in models:
            model["history"] = None
        try:
            with DashboardClient() as client:
                payload = client.get_json("/api/autoconfig/catalogue")
        except CLIError:
            payload = None
        rows = payload.get("rows") if isinstance(payload, dict) else None
        for model in models:
            runs = [
                row["run"]
                for row in rows or []
                if row.get("name") == model["name"] and row.get("run")
            ]
            if runs:
                latest = runs[0]
                model["history"] = {
                    "runs": len(runs),
                    "latest_run_id": latest["id"],
                    "status": latest["status"],
                    "cost_per_step_usd": latest["cost_per_step_usd"],
                    "time_per_step_s": latest["time_per_step_s"],
                    "initial_reward": latest["initial_reward"],
                }
    if json_output:
        print_json(models)
        return
    print_table(
        ["Model", "Recipe", "Framework", "GPUs", "Context", "$/step", "s/step", "Runs"],
        [
            (
                m["name"],
                m["recipe"],
                m["framework"],
                f"{m['gpus']}x {m['gpu_type']}",
                _fmt(m["context_length"]),
                _fmt((m.get("history") or {}).get("cost_per_step_usd")),
                _fmt((m.get("history") or {}).get("time_per_step_s")),
                _fmt((m.get("history") or {}).get("runs")),
            )
            for m in models
        ],
    )


@autoconfig_group.command(
    "knobs",
    help="Show the recipe fields a sweep of MODEL can set, with defaults and docs.",
    epilog=(
        "Examples:\n"
        "  modal-dojo autoconfig knobs Qwen3.8-27B\n"
        "  modal-dojo autoconfig knobs Qwen3.8-27B --all --json"
    ),
)
@click.argument("model")
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    default=False,
    help="Every field, not just the common ones.",
)
@json_option
def knobs_command(*, model: str, show_all: bool, json_output: bool) -> None:
    try:
        payload = recipe_knobs(model)
    except ValueError as exc:
        raise CLIError(
            str(exc),
            error="unknown_model",
            exit_code=ExitCode.NOT_FOUND,
            hint="modal-dojo autoconfig models",
        ) from exc
    knobs = [k for k in payload["knobs"] if show_all or k["highlight"]]
    if json_output:
        print_json({**payload, "knobs": knobs})
        return
    print_table(
        ["Path", "Type", "Default", "Doc"],
        [(k["path"], k["type"], json.dumps(k["default"]), k["doc"]) for k in knobs],
        title=f"{payload['name']} — {payload['recipe']} ({payload['framework']})",
    )
    if not show_all:
        click.echo(
            f"{len(HIGHLIGHTED_KNOBS)} common fields shown; --all lists "
            f"{len(payload['knobs'])}.",
            err=True,
        )


def _sweep_options(function: Callable[..., Any]) -> Callable[..., Any]:
    presets = ", ".join(sorted(DATASET_PRESETS))
    options = [
        click.option(
            "-m",
            "--model",
            "models",
            multiple=True,
            required=True,
            help="Registry model to benchmark; repeat for several.",
        ),
        click.option(
            "--steps",
            type=click.IntRange(1, 100),
            default=3,
            show_default=True,
            help="Training steps per run.",
        ),
        click.option(
            "--context",
            default=None,
            help="Longest response in tokens, e.g. 32k; sets the recipe's response and packing caps.",
        ),
        click.option(
            "--dataset",
            default="swe-bench",
            show_default=True,
            help=f"A preset ({presets}) or a Hugging Face dataset repo.",
        ),
        click.option("--split", default=None, help="Dataset split (default: train)."),
        click.option(
            "--input-column",
            default=None,
            help="Prompt column of a Hugging Face dataset.",
        ),
        click.option(
            "--output-column",
            default=None,
            help="Answer column of a Hugging Face dataset.",
        ),
        click.option(
            "--set",
            "sets",
            multiple=True,
            metavar="PATH=VALUE",
            help="Set a TrainConfig field on every run, e.g. recipe.gpu_type=B300.",
        ),
        click.option(
            "--grid",
            multiple=True,
            metavar="PATH=V1,V2",
            help="Sweep a field over comma-separated values, e.g. recipe.lr=1e-6,5e-6.",
        ),
        click.option(
            "--name", default=None, help="Sweep name; becomes part of the group id."
        ),
        click.option(
            "--allow-oom",
            is_flag=True,
            default=False,
            help="Launch variants the memory estimate says will not fit.",
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function


@autoconfig_group.command(
    "plan",
    help="Dry-run a sweep: expand the grid and estimate GPU memory per run. Spends nothing.",
    epilog=(
        "Examples:\n"
        "  modal-dojo autoconfig plan -m Qwen3.8-27B --context 32k --dataset swe-bench\n"
        "  modal-dojo autoconfig plan -m Qwen3.8-27B --grid recipe.lr=1e-6,5e-6 --json"
    ),
)
@_sweep_options
@json_option
def plan_command(*, json_output: bool, **options: Any) -> None:
    request = build_request(**options)
    plan = plan_or_fail(request)
    if json_output:
        print_json({"request": request.model_dump(mode="json"), **plan})
        return
    print_plan(plan)


@autoconfig_group.command(
    "sweep",
    help="Plan a sweep, then launch it through the dashboard.",
    epilog=(
        "Examples:\n"
        "  modal-dojo autoconfig sweep -m Qwen3.8-27B --context 32k --grid recipe.lr=1e-6,5e-6\n"
        "  modal-dojo autoconfig sweep -m Qwen3.8-27B -m Qwen3.6-27B --name 27b-swe --wait --json"
    ),
)
@_sweep_options
@click.option(
    "--wait", is_flag=True, default=False, help="Poll until every run has launched."
)
@json_option
def sweep_command(*, wait: bool, json_output: bool, **options: Any) -> None:
    request = build_request(**options)
    plan = plan_or_fail(request)
    if not plan["fits"] and not request.allow_oom:
        if not json_output:
            print_plan(plan)
        raise CLIError(
            f"{plan['oom_variants']} variant(s) exceed GPU memory by estimate.",
            error="sweep_exceeds_gpu_memory",
            exit_code=ExitCode.USAGE,
            hint="Change gpu_type, parallelism, or context, or pass --allow-oom.",
        )
    with DashboardClient() as client:
        operation = client.post_json(
            "/api/autoconfig/sweeps", request.model_dump(mode="json"), timeout=60.0
        )
        if wait:
            operation = _wait(client, operation)
    payload = {"plan": plan, "operation": operation}
    if json_output:
        print_json(payload)
        return
    print_plan(plan)
    print_operation(operation)


def _wait(client: DashboardClient, operation: dict[str, Any]) -> dict[str, Any]:
    while operation.get("status") == "pending":
        time.sleep(POLL_INTERVAL_SECONDS)
        operation = client.get_json(
            f"/api/autoconfig/sweeps/{operation['operation_id']}"
        )
    return operation


def print_operation(operation: dict[str, Any]) -> None:
    status = operation.get("status")
    click.echo(f"operation {operation.get('operation_id')}: {status}")
    if status == "pending":
        click.echo(
            f"  group {operation.get('group_id')}; poll with "
            f"`modal-dojo autoconfig status {operation.get('operation_id')}`"
        )
    elif status == "failed":
        error = operation.get("error") or {}
        click.echo(f"  {error.get('type')}: {error.get('message')}", err=True)
    else:
        result = operation.get("result") or {}
        for run in result.get("launched", []):
            click.echo(
                f"  launched {run['training_run_id']} — {run['entry']} {run['overrides']}"
            )
        for failure in result.get("failures", []):
            click.echo(
                f"  failed   {failure['entry']} {failure['overrides']}: {failure['error']}",
                err=True,
            )
        if result.get("group_id"):
            click.echo(
                f"  runs are in group {result['group_id']}: "
                f"`modal-dojo run list --group-id {result['group_id']}`"
            )


@autoconfig_group.command(
    "status",
    help="Show whether a sweep's runs have launched, and their run ids.",
    epilog=(
        "Examples:\n"
        "  modal-dojo autoconfig status fc-01ABC\n"
        "  modal-dojo autoconfig status fc-01ABC --wait --json"
    ),
)
@click.argument("operation_id")
@click.option(
    "--wait", is_flag=True, default=False, help="Poll until the launch finishes."
)
@json_option
def status_command(*, operation_id: str, wait: bool, json_output: bool) -> None:
    not_found = CLIError(
        f"Sweep operation {operation_id!r} was not found.",
        error="sweep_not_found",
        exit_code=ExitCode.NOT_FOUND,
        operation_id=operation_id,
    )
    with DashboardClient() as client:
        operation = client.get_json(
            f"/api/autoconfig/sweeps/{operation_id}", not_found_error=not_found
        )
        if wait:
            operation = _wait(client, operation)
    if json_output:
        print_json(operation)
        return
    print_operation(operation)
