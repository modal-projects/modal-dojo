"""Weekly synthetic monitoring for modal-dojo model validation.

Test it out with:

```shell
    uv run modal run -e training-gym -m synthetic_monitoring.model_validation --model qwen3-4b
```
"""

from __future__ import annotations

import os
import runpy
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import modal

from modal_dojo.common.models.validation import _ValidationConfig
from modal_dojo.common.run import TrainingRun, TrainingRunStatus
from scripts.validate_model_configs import ValidationResult, run_base_training
from synthetic_monitoring.chart import RunPoint, render_timing_history_chart

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_TIMEOUT_S = 2 * 60 * 60
CLEANUP_GRACE_S = 5 * 60
LAUNCH_TIMEOUT_S = PROBE_TIMEOUT_S + CLEANUP_GRACE_S + 30 * 60
MODAL_ENV = "training-gym"
HISTORY_DICT_NAME = "gym-synmon-timing-baselines"
QUICKSTART_NAME = "quickstart"

probe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(REPO_ROOT), extra_options="--no-dev")
    .uv_pip_install("slack-sdk==3.27.1", "matplotlib==3.10.1")
    .env({"MODAL_ENVIRONMENT": MODAL_ENV})
    .add_local_python_source("modal_dojo", "synthetic_monitoring", "scripts")
    .add_local_dir(
        str(REPO_ROOT / "dashboards" / "frontend"),
        remote_path="/root/dashboards/frontend",
        ignore=["node_modules", "dist"],
    )
)

slack_secret = modal.Secret.from_name("gym-bot-slack", environment_name=MODAL_ENV)
hf_secret = modal.Secret.from_name("huggingface-secret", environment_name="main")

app = modal.App("gym-synmon-launcher")


def _history_dict():
    return modal.Dict.from_name(
        HISTORY_DICT_NAME, create_if_missing=True, environment_name=MODAL_ENV
    )


def _raw_cell(text: str) -> dict:
    return {"type": "raw_text", "text": text}


def slack_report_blocks(
    rows: list[dict], *, when: datetime | None = None
) -> list[dict]:
    day = (when or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    table_rows = [
        [
            _raw_cell("Model"),
            _raw_cell("Total runtime"),
            _raw_cell("Delta vs previous successful run"),
        ]
    ]
    for row in rows:
        if row["status"] == "success":
            total = int(round(row["total_duration_s"]))
            hours, rem = divmod(total, 3600)
            minutes, seconds = divmod(rem, 60)
            if hours:
                runtime_text = f"{hours}h {minutes:02d}m {seconds:02d}s"
            elif minutes:
                runtime_text = f"{minutes}m {seconds:02d}s"
            else:
                runtime_text = f"{seconds}s"
            runtime, delta = _raw_cell(runtime_text), _raw_cell(row["delta"])
        elif row.get("modal_app_url"):
            runtime, delta = (
                _raw_cell("FAILED"),
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "link",
                                    "text": "logs",
                                    "url": row["modal_app_url"],
                                }
                            ],
                        }
                    ],
                },
            )
        else:
            runtime, delta = (
                _raw_cell("FAILED"),
                _raw_cell((row.get("error") or "failed")[:200]),
            )
        table_rows.append([_raw_cell(row["model"]), runtime, delta])
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"Weekly model validation - {day}"},
        },
        {
            "type": "table",
            "column_settings": [
                {"is_wrapped": True},
                {"align": "left"},
                {"is_wrapped": True},
            ],
            "rows": table_rows,
        },
    ]


def _lookup_app_url(training_run_id: str) -> str | None:
    if not training_run_id:
        return None
    try:
        url = TrainingRun.from_id(training_run_id).modal_app_url
    except Exception:
        return None
    return url or None


def _record(model: str, point: RunPoint) -> list[RunPoint]:
    try:
        history = _history_dict()
        points = [RunPoint(**item) for item in history.get(model, [])]
        if point.training_run_id and any(
            p.training_run_id == point.training_run_id for p in points
        ):
            return points
        points.append(point)
        points.sort(key=lambda p: (p.ts, p.training_run_id))
        history[model] = [asdict(p) for p in points]
        return points
    except Exception as exc:
        print(f"error: append_history failed for {model}: {exc}")
        return [point]


def _format_error(exc: BaseException, subject: str) -> str:
    message = str(exc)
    if not message and isinstance(exc, TimeoutError):
        message = f"{subject} did not finish within {PROBE_TIMEOUT_S}s"
    return f"{type(exc).__name__}: {message}"


def _row(
    model: str, point: RunPoint, history: list[RunPoint], error: str | None = None
) -> dict:
    prior = next(
        (
            p
            for p in reversed(history)
            if p.status == "success"
            and (p.training_run_id, p.ts) != (point.training_run_id, point.ts)
        ),
        None,
    )
    ok = point.status == "success"
    if ok:
        prior_s = prior.total_duration_s if prior else None
        if prior_s is None:
            delta = "n/a"
        else:
            delta_s = point.total_duration_s - prior_s
            delta = (
                f"{delta_s:+.0f}s"
                if prior_s <= 0
                else f"{delta_s:+.0f}s ({delta_s / prior_s * 100:+.1f}%)"
            )
    else:
        delta = "n/a"
    return {
        "model": model,
        "status": point.status,
        "total_duration_s": point.total_duration_s if ok else None,
        "delta": delta,
        "modal_app_url": point.modal_app_url,
        "error": error,
    }


def _post_report(rows: list[dict]) -> None:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    channel = os.environ["SLACK_CHANNEL_ID"]
    try:
        client.conversations_join(channel=channel)
    except SlackApiError as exc:
        if exc.response["error"] != "already_in_channel":
            raise
    header = f"Weekly model validation - {datetime.now(timezone.utc):%Y-%m-%d}"
    thread_ts = client.chat_postMessage(
        channel=channel,
        text=header,
        blocks=slack_report_blocks(rows),
        mrkdwn=True,
        unfurl_links=False,
        unfurl_media=False,
    )["ts"]
    for row in rows:
        try:
            if row["status"] == "success":
                png = render_timing_history_chart(
                    [
                        RunPoint(**item)
                        for item in _history_dict().get(row["model"], [])
                    ],
                    model_name=row["model"],
                )
                if png:
                    client.files_upload_v2(
                        channel=channel,
                        thread_ts=thread_ts,
                        content=png,
                        filename=f"timing_history_{row['model'].replace('/', '_')}.png",
                        title=row["model"],
                    )
                continue
            text = f"`{row['model']}` failed"
            if row.get("error"):
                text += f"\n```{row['error'][:1500]}```"
            if row.get("modal_app_url"):
                text += f"\n<{row['modal_app_url']}|Modal logs>"
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=text,
                mrkdwn=True,
                unfurl_links=False,
                unfurl_media=False,
            )
        except Exception as exc:
            print(f"error: reply failed for {row['model']}: {exc}")


@app.function(
    image=probe_image,
    timeout=PROBE_TIMEOUT_S + CLEANUP_GRACE_S,
    secrets=[hf_secret],
)
def monitor(model: str = "", num_steps: int = 1) -> dict:
    if model == QUICKSTART_NAME:
        print(f"synmon: {QUICKSTART_NAME!r}")
        run: TrainingRun | None = None
        try:
            run = runpy.run_module("scripts.quickstart", run_name="__main__")["run"]
            completed = run.result(timeout=PROBE_TIMEOUT_S)
            training_run = TrainingRun.from_id(completed.training_run_id)
            url = training_run.modal_app_url or _lookup_app_url(
                training_run.training_run_id
            )
            ok = training_run.status == TrainingRunStatus.COMPLETED
            point = RunPoint(
                ts=time.time(),
                timings={},
                training_run_id=training_run.training_run_id,
                total_duration_s=float(training_run.duration_seconds or 0),
                status="success" if ok else "failed",
                modal_app_url=url,
            )
            error = None if ok else f"quickstart failed: {training_run.status.value}"
            return _row(QUICKSTART_NAME, point, _record(QUICKSTART_NAME, point), error)
        except Exception as exc:
            error = _format_error(exc, QUICKSTART_NAME)
            print(f"error: probe failed for {QUICKSTART_NAME}: {error}")
            if run is not None:
                run.close()
            url = _lookup_app_url(run.training_run_id) if run else None
            point = RunPoint(
                ts=time.time(),
                timings={},
                training_run_id=run.training_run_id,
                total_duration_s=run.duration_seconds,
                status="failed",
                modal_app_url=url,
            )
            return _row(QUICKSTART_NAME, point, _record(QUICKSTART_NAME, point), error)

    selected = _ValidationConfig.find(model).name
    print(f"synmon: model={selected!r} num_steps={num_steps}")
    result: ValidationResult | None = None
    try:
        result = run_base_training(
            selected, step_count=num_steps, timeout=PROBE_TIMEOUT_S
        )
        url = _lookup_app_url(result.training_run_id)
        point = RunPoint.from_validation_result(result, modal_app_url=url)
        error = (
            None
            if result.succeeded
            else (
                f"model validation failed for {selected}: "
                f"{result.training_run_status.value}"
            )
        )
        return _row(selected, point, _record(selected, point), error)
    except Exception as exc:
        error = _format_error(exc, selected)
        print(f"error: probe failed for {selected}: {error}")
        training_run_id = (
            result.training_run_id
            if result
            else str(getattr(exc, "training_run_id", "") or "")
        )
        url = _lookup_app_url(training_run_id)
        point = RunPoint(
            ts=time.time(),
            timings={},
            training_run_id=training_run_id,
            total_duration_s=float(getattr(result, "total_duration_s", 0) or 0),
            status="failed",
            modal_app_url=url,
        )
        return _row(selected, point, _record(selected, point), error)


@app.function(
    image=probe_image,
    timeout=LAUNCH_TIMEOUT_S,
    schedule=modal.Cron("17 0 * * 0"),
    secrets=[slack_secret],
)
def launch_weekly(model: str = "", num_steps: int = 1) -> list[dict]:
    if not model:
        names = [c.name for c in _ValidationConfig.select()] + [QUICKSTART_NAME]
    elif model == QUICKSTART_NAME:
        names = [QUICKSTART_NAME]
    else:
        names = [_ValidationConfig.find(model).name]
    if not names:
        raise RuntimeError("no validatable models registered")

    rows: list[dict] = []
    payloads = list(
        monitor.map(names, kwargs={"num_steps": num_steps}, return_exceptions=True)
    )
    for name, payload in zip(names, payloads):
        if isinstance(payload, Exception):
            tb = "".join(
                traceback.format_exception(
                    type(payload), payload, payload.__traceback__
                )
            )
            print(f"error: probe failed for {name}:\n{tb}")
            rows.append(
                {
                    "model": name,
                    "status": "failed",
                    "total_duration_s": None,
                    "delta": "n/a",
                    "modal_app_url": None,
                    "error": f"{type(payload).__name__}: {payload}",
                }
            )
            continue
        rows.append(payload)

    _post_report(rows)
    if any(row["status"] != "success" for row in rows):
        raise RuntimeError("synmon fan-out finished with failures")
    return rows


@app.local_entrypoint()
def main(model: str = "", num_steps: int = 1) -> None:
    print(launch_weekly.remote(model=model, num_steps=num_steps))
