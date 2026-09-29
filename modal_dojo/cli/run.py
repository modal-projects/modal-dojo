"""Commands for inspecting training runs."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import click
from pydantic import ValidationError
from rich.columns import Columns
from rich.console import Group
from rich.filesize import decimal as format_filesize
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from modal_dojo.common.run_list import run_list_field_metadata
from modal_dojo.common.run_summary import RunSummary
from modal_dojo.common.time import parse_time
from modal_dojo.common.training_rollout import (
    TrainingRolloutResult,
    TrainingRolloutSummary,
)

from .client import DashboardClient
from .commands import _TrainingGymGroup
from .errors import CLIError, ExitCode
from .options import confirm_or_require_yes, json_option, yes_option
from .output import print_json, print_renderable, print_table


DEFAULT_RUN_LIMIT = 50
DEFAULT_LOG_TAIL = 100
MAX_LOG_TAIL = 20_000
TRACE_DOWNLOAD_TIMEOUT_SECONDS = 300.0
CLI_FIELD_NAMES = {
    "display_status": "status",
    "display_stage": "stage",
    "group_id": "group",
    "updated_at": "last_updated_at",
}


def _run_filter_options(function: Callable[..., Any]) -> Callable[..., Any]:
    """Generate Click filters from list fields marked filterable on RunSummary."""
    for name, metadata in reversed(run_list_field_metadata().items()):
        if not metadata.get("filterable"):
            continue
        option_name = CLI_FIELD_NAMES.get(name, name)
        function = click.option(
            f"--{option_name.replace('_', '-')}",
            name,
            default=None,
            metavar=option_name.upper(),
            help=f"Filter by {str(metadata['label']).lower()}.",
        )(function)
    return function


def _format_timestamp(value: object) -> str | None:
    if not isinstance(value, (int, float)) or not value:
        return None
    return (
        datetime.fromtimestamp(value, tz=UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _format_table_timestamp(value: object, *, now: float | None = None) -> str:
    if not isinstance(value, (int, float)) or not value:
        return "—"
    age = max(0, (time.time() if now is None else now) - value)
    if age < 60:
        return "now"
    if age < 3_600:
        return f"{int(age // 60)}m ago"
    if age < 86_400:
        return f"{int(age // 3_600)}h ago"
    if age < 2_592_000:
        return f"{int(age // 86_400)}d ago"
    return datetime.fromtimestamp(value, tz=UTC).strftime("%Y-%m-%d")


def _table_rows(
    summaries: list[RunSummary],
    fields: dict[str, dict[str, object]],
) -> list[list[object]]:
    rows: list[list[object]] = []
    for summary in summaries:
        rows.append(
            [
                _format_table_timestamp(getattr(summary, name))
                if metadata.get("timestamp")
                else getattr(summary, name) or "—"
                for name, metadata in fields.items()
            ]
        )
    return rows


def _format_reward(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _current_step(summary: RunSummary) -> tuple[int | None, int | None, str]:
    progress = summary.framework_progress
    if progress is None:
        return None, None, "step"
    return progress.current, progress.total, progress.unit


def _run_payload(summary: RunSummary) -> dict[str, object]:
    current_step, total_steps, step_unit = _current_step(summary)
    return {
        "run_id": summary.run_id,
        "modal_app_id": summary.modal_app_id or None,
        "status": summary.display_status,
        "stage": summary.display_stage or None,
        "current_step": current_step,
        "total_steps": total_steps,
        "step_unit": step_unit,
        "current_reward": (
            summary.latest_rollout.mean if summary.latest_rollout is not None else None
        ),
        "resume_state": (
            summary.resume_state.model_dump(mode="json")
            if summary.resume_state is not None
            else None
        ),
        "model": summary.model or None,
        "dataset": summary.dataset or None,
        "recipe": summary.recipe or None,
        "group": summary.group_id or None,
        "created_at": _format_timestamp(summary.created_at),
        "last_updated_at": _format_timestamp(summary.updated_at),
    }


def _validate_run_summary(payload: object) -> RunSummary:
    try:
        return RunSummary.model_validate(payload)
    except ValidationError as exc:
        raise CLIError(
            "Dashboard returned an invalid run summary.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        ) from exc


def _validate_rollouts(payload: object) -> list[TrainingRolloutSummary]:
    if not isinstance(payload, list):
        raise CLIError(
            "Dashboard returned invalid rollout data.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        )
    try:
        return [TrainingRolloutSummary.model_validate(item) for item in payload]
    except ValidationError as exc:
        raise CLIError(
            "Dashboard returned invalid rollout data.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        ) from exc


def _parse_steps(value: str | None) -> set[int] | None:
    """Parse steps and Python-style ``start-end[:stride]`` ranges."""
    if value is None:
        return None

    steps: set[int] = set()
    try:
        for raw_part in value.split(","):
            part = raw_part.strip()
            if not part:
                raise ValueError
            if "-" not in part:
                step = int(part)
                if step < 0:
                    raise ValueError
                steps.add(step)
                continue

            bounds, separator, stride_text = part.partition(":")
            start_text, dash, end_text = bounds.partition("-")
            if not dash or not start_text or not end_text:
                raise ValueError
            start, end = int(start_text), int(end_text)
            stride = int(stride_text) if separator else 1
            if start < 0 or end <= start or stride < 1:
                raise ValueError
            steps.update(range(start, end, stride))
    except ValueError as exc:
        raise click.BadParameter(
            "Must be a comma-separated list of non-negative steps or "
            "start-inclusive, end-exclusive ranges such as 4-100:2.",
            param_hint="--step",
        ) from exc
    return steps


def _trace_output_path(out: str, run_id: str) -> Path:
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise click.BadParameter(
            "Must be a single safe path component.",
            param_hint="RUN_ID",
        )
    output_root = Path(out).expanduser().resolve()
    target = output_root / run_id
    if target.is_symlink():
        raise CLIError(
            f"Trace output path {str(target)!r} is a symbolic link.",
            error="unsafe_output_path",
        )
    return target


def _download_trace_step(
    *,
    client: DashboardClient,
    encoded_run_id: str,
    run_id: str,
    summary: TrainingRolloutSummary,
    staging_path: Path,
    file_name: str,
) -> tuple[dict[str, object], int]:
    rollout_id = summary.rollout_id
    payload = client.get_json(
        f"/api/runs/{encoded_run_id}/rollouts/{rollout_id}",
        params=None,
        not_found_error=CLIError(
            f"Step {rollout_id} for run {run_id!r} was not found.",
            error="rollout_not_found",
            exit_code=ExitCode.NOT_FOUND,
            run_id=run_id,
            step=rollout_id,
        ),
        timeout=TRACE_DOWNLOAD_TIMEOUT_SECONDS,
    )
    try:
        rollout = TrainingRolloutResult.model_validate(payload)
    except ValidationError as exc:
        raise CLIError(
            "Dashboard returned invalid rollout data.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        ) from exc

    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode()
    (staging_path / file_name).write_bytes(data)
    return (
        {
            "step": rollout_id,
            "file_name": file_name,
            "samples": rollout.total,
            "mean_reward": rollout.mean,
            "export_size_bytes": len(data),
        },
        len(data),
    )


def download_run_traces(
    *,
    run_id: str,
    out: str,
    step: str | None,
    dry_run: bool,
    skip_confirmation: bool,
    json_output: bool,
) -> None:
    """Download selected rollout payloads and write a local trace manifest."""
    selected_steps = _parse_steps(step)
    output_path = _trace_output_path(out, run_id)
    encoded_run_id = quote(run_id, safe="")
    not_found_error = CLIError(
        f"Training run {run_id!r} was not found.",
        error="run_not_found",
        exit_code=ExitCode.NOT_FOUND,
        run_id=run_id,
        hint="training-gym run list",
    )

    with DashboardClient() as client:
        _validate_run_summary(
            client.get_json(
                f"/api/runs/{encoded_run_id}",
                params=None,
                not_found_error=not_found_error,
            )
        )
        rollout_summaries = _validate_rollouts(
            client.get_json(
                f"/api/runs/{encoded_run_id}/rollouts",
                params=None,
            )
        )
        summaries_by_step = {
            rollout.rollout_id: rollout for rollout in rollout_summaries
        }
        if selected_steps is None:
            selected_summaries = sorted(
                rollout_summaries,
                key=lambda rollout: rollout.rollout_id,
            )
        else:
            missing = sorted(selected_steps - summaries_by_step.keys())
            if missing:
                missing_text = ", ".join(str(value) for value in missing)
                raise click.BadParameter(
                    f"Step(s) not found for run {run_id!r}: {missing_text}.",
                    param_hint="--step",
                )
            selected_summaries = [
                summaries_by_step[value] for value in sorted(selected_steps)
            ]

        step_width = max(4, len(str(max(summaries_by_step, default=0))))
        known_sizes = [summary.export_size_bytes for summary in selected_summaries]
        total_size = (
            sum(size for size in known_sizes if size is not None)
            if all(size is not None for size in known_sizes)
            else None
        )
        report: dict[str, object] = {
            "run_id": run_id,
            "output_path": str(output_path),
            "dry_run": dry_run,
            "step_count": len(selected_summaries),
            "sample_count": sum(summary.total for summary in selected_summaries),
            "export_size_bytes": total_size,
            "steps": [
                {
                    "step": summary.rollout_id,
                    "file_name": f"step_{summary.rollout_id:0{step_width}d}.json",
                    "samples": summary.total,
                    "mean_reward": summary.mean,
                    "export_size_bytes": summary.export_size_bytes,
                }
                for summary in selected_summaries
            ],
        }
        estimated_size = report["export_size_bytes"]
        if not isinstance(estimated_size, int):
            estimated_size_text = "unknown size"
        else:
            estimated_size_text = format_filesize(estimated_size)

        if dry_run:
            if json_output:
                print_json(report)
            else:
                click.echo(
                    f"{report['step_count']} steps, {report['sample_count']} samples, "
                    f"approximately {estimated_size_text}"
                )
            return

        if not skip_confirmation:
            confirm_or_require_yes(
                f"Download {report['step_count']} steps "
                f"({report['sample_count']} samples, "
                f"approximately {estimated_size_text}) "
                f"to {output_path}?"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path = Path(
            tempfile.mkdtemp(
                prefix=f".{run_id}-",
                dir=output_path.parent,
            )
        )
        downloaded_steps: list[dict[str, object]] = []
        downloaded_size = 0
        try:
            for rollout_summary in selected_summaries:
                entry, size_bytes = _download_trace_step(
                    client=client,
                    encoded_run_id=encoded_run_id,
                    run_id=run_id,
                    summary=rollout_summary,
                    staging_path=staging_path,
                    file_name=(
                        f"step_{rollout_summary.rollout_id:0{step_width}d}.json"
                    ),
                )
                downloaded_steps.append(entry)
                downloaded_size += size_bytes

            manifest = {"run_id": run_id, "steps": downloaded_steps}
            (staging_path / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            output_path.mkdir(parents=True, exist_ok=True)
            for old_path in output_path.glob("step_*.json"):
                old_path.unlink()
            manifest_path = output_path / "manifest.json"
            if manifest_path.exists():
                manifest_path.unlink()
            for staged_file in list(staging_path.iterdir()):
                staged_file.replace(output_path / staged_file.name)
        except OSError as exc:
            raise CLIError(
                f"Could not write trace files to {str(output_path)!r}.",
                error="trace_write_failed",
            ) from exc
        finally:
            shutil.rmtree(staging_path, ignore_errors=True)

    report["export_size_bytes"] = downloaded_size
    report["steps"] = downloaded_steps
    if json_output:
        print_json(report)
    else:
        click.echo(
            f"Downloaded {report['step_count']} steps, "
            f"{report['sample_count']} samples ({format_filesize(downloaded_size)})"
        )
        click.echo(str(output_path))


def _format_step(summary: RunSummary) -> str:
    current, total, unit = _current_step(summary)
    if current is None:
        return "—"
    value = f"{current} / {total}" if total is not None else str(current)
    return f"{value} {unit}".strip()


def _chip(value: str, *, style: str) -> Text:
    return Text(f" {value} ", style=style)


def _run_summary_panel(summary: RunSummary) -> Panel:
    status = summary.display_status or "pending"
    status_style = {
        "completed": "bold bright_green",
        "failed": "bold red",
        "cancelled": "bold yellow",
        "stopped": "bold yellow",
        "pending": "bold cyan",
    }.get(status, "bold")
    heading = Text()
    heading.append("● ", style=status_style)
    heading.append(status.upper(), style=status_style)
    if summary.display_stage:
        heading.append("  ")
        heading.append(summary.display_stage, style="bold")

    reward = summary.latest_rollout.mean if summary.latest_rollout is not None else None
    metrics = Table.grid(padding=(0, 4))
    metrics.add_row(
        Text.assemble(("Step  ", "dim"), (_format_step(summary), "bold")),
        Text.assemble(("Reward  ", "dim"), (_format_reward(reward), "bold")),
    )

    chips = [
        _chip(summary.model, style="black on bright_green") if summary.model else None,
        _chip(summary.dataset, style="black on cyan") if summary.dataset else None,
        _chip(summary.recipe, style="black on white") if summary.recipe else None,
        _chip(summary.group_id, style="white on grey23") if summary.group_id else None,
    ]
    footer = Text.assemble(
        ("Updated ", "dim"),
        (_format_table_timestamp(summary.updated_at), "dim bold"),
        ("  ·  Created ", "dim"),
        (_format_table_timestamp(summary.created_at), "dim bold"),
    )
    modal_app = Text.assemble(
        ("Modal app  ", "dim"),
        (summary.modal_app_id or "—", "bold"),
    )
    body = Group(
        heading,
        Text(""),
        metrics,
        Text(""),
        Columns([chip for chip in chips if chip is not None], padding=(0, 1)),
        Text(""),
        modal_app,
        Text(""),
        footer,
    )
    return Panel(
        body,
        title=summary.run_id,
        title_align="left",
        border_style="bright_green",
        padding=(1, 2),
    )


def _reward_sparkline(rollouts: list[TrainingRolloutSummary]) -> str:
    if not rollouts:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    rewards = [rollout.mean for rollout in rollouts]
    low, high = min(rewards), max(rewards)
    if low == high:
        return blocks[len(blocks) // 2] * len(rewards)
    return "".join(
        blocks[round((reward - low) / (high - low) * (len(blocks) - 1))]
        for reward in rewards
    )


def _reward_panel(rollouts: list[TrainingRolloutSummary]) -> Panel:
    if not rollouts:
        content: Text | Group = Text("No rollout rewards recorded.", style="dim")
    else:
        first, latest = rollouts[0].mean, rollouts[-1].mean
        table = Table(
            "Rollout",
            "Reward",
            "Samples",
            "Duration",
            "Errors",
            box=None,
            header_style="bold bright_green",
            pad_edge=False,
            expand=True,
        )
        for rollout in rollouts:
            table.add_row(
                str(rollout.rollout_id),
                _format_reward(rollout.mean),
                str(rollout.total),
                (
                    f"{rollout.rollout_time:.2f}s"
                    if rollout.rollout_time is not None
                    else "—"
                ),
                (
                    str(rollout.error_summary.get("verdict", "—"))
                    if rollout.error_summary is not None
                    else "—"
                ),
            )
        content = Group(
            Text(_reward_sparkline(rollouts), style="bold bright_green"),
            Text.assemble(
                (_format_reward(first), "dim"),
                ("  →  ", "dim"),
                (_format_reward(latest), "bold"),
                (f"   {len(rollouts)} rollouts", "dim"),
            ),
            Text(""),
            table,
        )
    return Panel(
        content,
        title="Reward over time",
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
    )


def get_run(*, run_id: str, verbose: bool, json_output: bool) -> None:
    """Fetch and render one run, optionally including rollout history."""
    encoded_run_id = quote(run_id, safe="")
    not_found_error = CLIError(
        f"Training run {run_id!r} was not found.",
        error="run_not_found",
        exit_code=ExitCode.NOT_FOUND,
        run_id=run_id,
        hint="training-gym run list",
    )
    with DashboardClient() as client:
        summary = _validate_run_summary(
            client.get_json(
                f"/api/runs/{encoded_run_id}",
                params=None,
                not_found_error=not_found_error,
            )
        )
        rollouts = (
            _validate_rollouts(
                client.get_json(
                    f"/api/runs/{encoded_run_id}/rollouts",
                    params=None,
                )
            )
            if verbose
            else []
        )

    if json_output:
        payload = _run_payload(summary)
        if verbose:
            payload["reward_over_time"] = [
                {
                    "rollout_id": rollout.rollout_id,
                    "reward": rollout.mean,
                    "created_at": _format_timestamp(rollout.created_at),
                }
                for rollout in rollouts
            ]
            payload["rollouts"] = [
                rollout.model_dump(mode="json", exclude_none=True)
                for rollout in rollouts
            ]
        print_json(payload)
        return

    print_renderable(_run_summary_panel(summary))
    if not verbose:
        return

    print_renderable(_reward_panel(rollouts))


def show_run_params(*, run_id: str, json_output: bool) -> None:
    """Fetch and render the framework recipe parameters for one run."""
    encoded_run_id = quote(run_id, safe="")
    not_found_error = CLIError(
        f"Training run {run_id!r} was not found.",
        error="run_not_found",
        exit_code=ExitCode.NOT_FOUND,
        run_id=run_id,
        hint="training-gym run list",
    )
    with DashboardClient() as client:
        summary = _validate_run_summary(
            client.get_json(
                f"/api/runs/{encoded_run_id}",
                params=None,
                not_found_error=not_found_error,
            )
        )

    config = summary.config
    params = config.get("recipe") or config.get("preset") or {}
    if not isinstance(params, dict):
        raise CLIError(
            "Dashboard returned invalid framework parameters.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
            run_id=run_id,
        )

    if json_output:
        print_json(params)
        return

    configured_params = {
        name: value
        for name, value in params.items()
        if value is not None and value != ""
    }
    print_table(
        ["Parameter", "Value"],
        [
            [
                name,
                (
                    value
                    if isinstance(value, str)
                    else json.dumps(value, ensure_ascii=False)
                ),
            ]
            for name, value in configured_params.items()
        ],
        title=f"Training recipe for {run_id}",
        show_header=False,
    )


def _validate_log_payload(
    payload: object,
) -> tuple[list[dict[str, object]], bool, int | float | None]:
    if not isinstance(payload, dict) or not isinstance(payload.get("logs"), list):
        raise CLIError(
            "Dashboard returned invalid log data.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        )

    logs: list[dict[str, object]] = []
    for entry in payload["logs"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("line"), str):
            raise CLIError(
                "Dashboard returned invalid log data.",
                error="invalid_dashboard_response",
                exit_code=ExitCode.BACKEND,
            )
        logs.append(entry)

    has_more = payload.get("has_more", False)
    next_until = payload.get("next_until")
    if not isinstance(has_more, bool) or (
        next_until is not None
        and (not isinstance(next_until, (int, float)) or isinstance(next_until, bool))
    ):
        raise CLIError(
            "Dashboard returned invalid log pagination data.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        )
    return logs, has_more, next_until


def _print_log_line(line: str) -> None:
    click.echo(line, nl=not line.endswith("\n"))


def _decode_stream_event(event: str, data: str) -> dict[str, object]:
    try:
        payload = json.loads(data)
    except (TypeError, ValueError) as exc:
        raise CLIError(
            "Dashboard returned malformed log stream data.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        ) from exc
    if not isinstance(payload, dict):
        raise CLIError(
            "Dashboard returned invalid log stream data.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        )
    if event != "message":
        payload = {"event": event, **payload}
    return payload


def show_run_logs(
    *,
    run_id: str,
    follow: bool,
    since: str | None,
    until: str | None,
    tail: int | None,
    search: str | None,
    json_output: bool,
) -> None:
    """Show historical logs or follow the live dashboard log stream."""
    if follow and (since or until or tail is not None):
        raise click.UsageError(
            "--since, --until, and --tail apply only when fetching logs "
            "without --follow."
        )

    encoded_run_id = quote(run_id, safe="")
    not_found_error = CLIError(
        f"Training run {run_id!r} was not found.",
        error="run_not_found",
        exit_code=ExitCode.NOT_FOUND,
        run_id=run_id,
        hint="training-gym run list",
    )

    with DashboardClient() as client:
        if not follow:
            payload = client.get_json(
                f"/api/runs/{encoded_run_id}/logs",
                params={
                    "since": since,
                    "until": until,
                    "max_lines": tail or DEFAULT_LOG_TAIL,
                    "search": search,
                },
                not_found_error=not_found_error,
            )
            logs, has_more, next_until = _validate_log_payload(payload)
            if has_more:
                message = (
                    f"[showing the newest {len(logs)} logs; older logs are available"
                )
                if next_until is not None:
                    message += f" with --until {next_until}"
                click.echo(f"{message}]", err=True)
            if json_output:
                print_json(
                    {
                        "logs": logs,
                        "has_more": has_more,
                        "next_until": next_until,
                    }
                )
                return
            for entry in logs:
                _print_log_line(str(entry["line"]))
            return

        for event, data in client.iter_event_stream(
            f"/api/runs/{encoded_run_id}/logs/stream",
            params={"search": search},
            not_found_error=not_found_error,
        ):
            decoded = _decode_stream_event(event, data)
            if event == "done":
                return
            if event == "error":
                raise CLIError(
                    str(decoded.get("error") or "Dashboard log stream failed."),
                    error="log_stream_failed",
                    exit_code=ExitCode.BACKEND,
                    run_id=run_id,
                )
            if json_output:
                click.echo(json.dumps(decoded, ensure_ascii=False))
            elif event == "message":
                line = decoded.get("line")
                if not isinstance(line, str):
                    raise CLIError(
                        "Dashboard returned invalid log stream data.",
                        error="invalid_dashboard_response",
                        exit_code=ExitCode.BACKEND,
                    )
                _print_log_line(line)
            elif event == "dropped":
                click.echo(
                    f"[dropped {decoded.get('dropped', 0)} log lines]",
                    err=True,
                )
            elif event == "reconnect":
                click.echo("[reconnecting log stream]", err=True)
        raise CLIError(
            "Dashboard log stream ended unexpectedly.",
            error="log_stream_disconnected",
            exit_code=ExitCode.BACKEND,
            run_id=run_id,
            hint="Re-run the command to reconnect.",
        )


def list_runs(
    *,
    since: str | None,
    limit: int,
    json_output: bool,
    filters: dict[str, str | None],
) -> None:
    """Fetch, validate, and render the run list."""
    parsed_since = parse_time(since, time.time()) if since else None
    if since and parsed_since is None:
        raise click.BadParameter(
            "Must be epoch seconds, ISO 8601, or a relative time such as 24h"
        )
    params: dict[str, str | int | None] = {
        **filters,
        "since": int(parsed_since) if parsed_since is not None else None,
        "limit": limit,
    }
    with DashboardClient() as client:
        payload = client.get_json("/api/runs", params=params)

    if not isinstance(payload, list):
        raise CLIError(
            "Dashboard returned an invalid run list.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        )
    summaries = [_validate_run_summary(item) for item in payload]
    fields = run_list_field_metadata()
    if json_output:
        print_json(
            [
                {
                    CLI_FIELD_NAMES.get(name, name): (
                        _format_timestamp(getattr(summary, name))
                        if metadata.get("timestamp")
                        else getattr(summary, name)
                    )
                    for name, metadata in fields.items()
                }
                for summary in summaries
            ]
        )
    else:
        print_table(
            [str(metadata["label"]) for metadata in fields.values()],
            _table_rows(summaries, fields),
        )


@click.group("run", cls=_TrainingGymGroup)
def run_group() -> None:
    """Inspect and manage runs."""


@run_group.command(
    "get",
    help="Show a run's status and top-level metadata.",
    epilog=(
        "Examples:\n"
        "  training-gym run get brave-falcon-3fa8\n"
        "  training-gym run get brave-falcon-3fa8 --verbose"
    ),
)
@click.argument("run_id")
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Include reward-over-time and rollout data.",
)
@json_option
def get_command(*, run_id: str, verbose: bool, json_output: bool) -> None:
    """Show a run's status and top-level metadata."""
    get_run(run_id=run_id, verbose=verbose, json_output=json_output)


@run_group.command(
    "params",
    help="Show a run's framework training recipe.",
    epilog=(
        "Examples:\n"
        "  training-gym run params brave-falcon-3fa8\n"
        "  training-gym run params brave-falcon-3fa8 --json"
    ),
)
@click.argument("run_id")
@json_option
def params_command(*, run_id: str, json_output: bool) -> None:
    """Show a run's framework training recipe."""
    show_run_params(run_id=run_id, json_output=json_output)


@run_group.command(
    "logs",
    help="Show logs for a training run.",
    epilog=(
        "Examples:\n"
        "  training-gym run logs brave-falcon-3fa8 --follow\n"
        "  training-gym run logs brave-falcon-3fa8 --since 30m -j"
    ),
)
@click.argument("run_id")
@click.option(
    "-f",
    "--follow",
    is_flag=True,
    default=False,
    help="Keep streaming new logs until interrupted or the run stops.",
)
@click.option(
    "--since",
    default=None,
    metavar="START",
    help="Show logs since a timestamp or relative time, such as 30m or 2h.",
)
@click.option(
    "--until",
    default=None,
    metavar="END",
    help="Show logs up to a timestamp or relative time, such as 30m or 2h.",
)
@click.option(
    "-n",
    "--tail",
    type=click.IntRange(min=1, max=MAX_LOG_TAIL),
    default=None,
    show_default=str(DEFAULT_LOG_TAIL),
    metavar="N",
    help="Show at most N recent logs in the requested window.",
)
@click.option(
    "--search",
    default=None,
    metavar="TEXT",
    help="Filter by search text.",
)
@json_option
def logs_command(
    *,
    run_id: str,
    follow: bool,
    since: str | None,
    until: str | None,
    tail: int | None,
    search: str | None,
    json_output: bool,
) -> None:
    """Show logs for a training run."""
    show_run_logs(
        run_id=run_id,
        follow=follow,
        since=since,
        until=until,
        tail=tail,
        search=search,
        json_output=json_output,
    )


@run_group.command(
    "trace",
    help="Download agent traces for a run.",
    epilog=(
        "Examples:\n"
        "  training-gym run trace brave-falcon-3fa8 --out ./traces --step 4-100:2\n"
        "  training-gym run trace brave-falcon-3fa8 --out ./traces "
        "--step 1,4,9 --dry-run\n"
        "  training-gym run trace brave-falcon-3fa8 --out ./traces --yes"
    ),
)
@click.argument("run_id")
@click.option(
    "--out",
    required=True,
    metavar="DIR",
    help="Output directory.",
)
@click.option(
    "--step",
    default=None,
    metavar="STEP",
    help="Select steps by list or end-exclusive range, or omit to select all steps.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Report step/sample count and approximate size without downloading.",
)
@yes_option
@json_option
def trace_command(
    *,
    run_id: str,
    out: str,
    step: str | None,
    dry_run: bool,
    yes: bool,
    json_output: bool,
) -> None:
    """Download agent traces for a run."""
    download_run_traces(
        run_id=run_id,
        out=out,
        step=step,
        dry_run=dry_run,
        skip_confirmation=yes,
        json_output=json_output,
    )


@run_group.command(
    "list",
    help=(
        "List training runs and top-level metadata.\n\n"
        "Results are sorted by most recently updated."
    ),
    epilog=(
        "Examples:\n"
        "  training-gym run list --status failed --since 24h\n"
        "  training-gym run list --status completed "
        "--group nightly-tau-bench -j"
    ),
)
@_run_filter_options
@click.option(
    "--since",
    default=None,
    metavar="TIME",
    help="Include runs created or updated since this timestamp or relative time.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=DEFAULT_RUN_LIMIT,
    show_default=True,
    metavar="N",
    help="Maximum number of runs to return.",
)
@json_option
def list_command(
    *,
    since: str | None,
    limit: int,
    json_output: bool,
    **filters: str | None,
) -> None:
    """List training runs and top-level metadata."""
    list_runs(
        since=since,
        limit=limit,
        json_output=json_output,
        filters=filters,
    )
