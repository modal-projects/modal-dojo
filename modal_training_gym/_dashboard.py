"""Self-contained training-gym dashboard app.

The image builds the frontend from ``dashboards/frontend`` in a repo
checkout, or the copy the wheel ships at ``modal_training_gym/_frontend``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets as _secrets
import subprocess
import tempfile
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Awaitable,
    Callable,
    Iterable,
    TypedDict,
    cast,
)

import modal
from modal.exception import Error

if TYPE_CHECKING:
    from google.protobuf.timestamp_pb2 import Timestamp
    from modal.client import _Client
    from modal_proto import api_pb2

# Imported at module scope so FastAPI can resolve endpoint annotations such as
# ``request: Request`` in stream_run_logs(). Under ``from __future__ import
# annotations`` all type hints are strings, and FastAPI evaluates them
# against the *defining function's* ``__globals__`` (i.e. this module).
# Importing ``Request`` only inside ``fastapi_app()`` makes the name
# invisible to FastAPI's introspection, which then mistakes the parameter
# for a query string and 422s with ``{"loc": ["query", "request"]}``.
from fastapi import Query
from starlette.requests import Request

# Used as endpoint parameter annotations, so — like ``Request`` above — these
# must resolve from this module's globals.
from modal_training_gym.common.advantage_distribution import AdvantageDistribution
from modal_training_gym.common.config import (
    DASHBOARD_PASSWORD_SECRET_NAME,
    DASHBOARD_PROXY_AUTH_PATH,
    DASHBOARD_VERSION_PATH,
    dashboard_requires_proxy_auth,
    get_dashboard_trajectory_viewer,
)
from modal_training_gym.common.dashboard import (
    DASHBOARD_APP_NAME,
    DASHBOARD_PREVIEW_ENV_KEY,
    DASHBOARD_VERSION_ENV_KEY,
    current_dashboard_version,
)
from modal_training_gym.common.dashboard_components import (
    DASHBOARD_OVERLAY_VOLUME_NAME,
    MAX_COMPONENT_BYTES,
    DashboardComponent,
)
from modal_training_gym.common.run import (
    FrameworkStatusUpdate,
    TrainingRun,
    TrainingRunStatus,
)
from modal_training_gym.common.run_list import (
    FACET_NAMES,
    count_run_facets,
    filter_run_summaries,
    run_list_field_metadata,
)
from modal_training_gym.common.run_summary import (
    JsonDict,
    RunSummary,
    build_run_summary,
    build_run_summaries,
)
from modal_training_gym.common.metric_series import (
    MAX_POINTS_PER_KEY,
    MetricPointsBatch,
    RunMetrics,
    metric_series,
    metric_series_store,
)
from modal_training_gym.common.step_timing import (
    RoleTimingRecord,
    legacy_run_to_records,
    rollout_lanes,
    validate_timing_record,
)
from modal_training_gym.common.time import parse_time as _parse_log_time
from modal_training_gym.common.training_rollout import (
    TrainingRolloutResult,
    TrainingRolloutSummary,
    _apply_parsed,
)
from modal_training_gym.utils.metadata import (
    bounded_gather_with_retries,
    vol_get as _metadata_vol_get,
    vol_list as _metadata_vol_list,
    vol_list_metadata_with_failures,
    vol_put_many as _metadata_vol_put_many,
)

SummaryLoader = Callable[[], Awaitable[list[JsonDict]]]

# Repeated params (``?status=failed&status=stopped``) mirror the run list's
# multi-select chips; an absent facet means "every bucket".
FacetParam = Annotated[list[str] | None, Query()]


# A single historical log line from ``AppFetchLogs``
class LogEntry(TypedDict):
    task_id: str
    line: str
    fd: int
    ts: float | None
    ts_ns: int | None


class TimingFileCache(TypedDict):
    mtime: int
    size: int
    checked_at: float
    record: JsonDict | None


DASHBOARD_REQUIRES_PROXY_AUTH_ENV_KEY = "DASHBOARD_REQUIRES_PROXY_AUTH"
TIMING_DEBUG_ENV = "TRAINING_GYM_TIMING_DEBUG"
DASHBOARD_COMPONENT_VOLUME_MOUNT = "/mnt/training-gym-dashboard-overlay"
dashboard_component_volume = modal.Volume.from_name(
    DASHBOARD_OVERLAY_VOLUME_NAME, create_if_missing=True
)


def _is_preview() -> bool:
    return os.environ.get(DASHBOARD_PREVIEW_ENV_KEY, "").lower() in {
        "1",
        "true",
        "yes",
    }


# A preview deploy is one dashboard per open PR (see scripts/previews) reading
# the same metadata volume as the real one, so it serves the API and nothing
# else: the scheduled jobs rewrite that shared metadata, and a warm container
# per open PR is a standing bill for a review aid.
IS_PREVIEW = _is_preview()


def _build_image() -> modal.Image:
    _pkg = Path(__file__).resolve().parent
    _checkout = _pkg.parent / "dashboards" / "frontend"
    _frontend = _checkout if _checkout.is_dir() else _pkg / "_frontend"
    base = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("curl")
        .run_commands(
            "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
            "apt-get install -y nodejs",
        )
        .pip_install("fastapi[standard]==0.118.0", "modal")
        .add_local_dir(
            str(_frontend),
            remote_path="/app/frontend",
            copy=True,
            ignore=["node_modules", "dist"],
        )
    )

    trajectory_viewer = get_dashboard_trajectory_viewer()
    if trajectory_viewer:
        viewer_path = Path(trajectory_viewer).expanduser().resolve()
        if not viewer_path.is_file():
            raise FileNotFoundError(
                "Configured trajectory viewer does not exist: "
                f"{viewer_path}. Run `training-gym setup --trajectory-viewer PATH` "
                "with a Svelte component file."
            )
        base = base.add_local_file(
            str(viewer_path),
            remote_path="/app/frontend/src/components/TrajectoryViewer.svelte",
            copy=True,
        )

    return (
        base.run_commands("cd /app/frontend && npm install && npm run build")
        .add_local_python_source("modal_training_gym", copy=True)
        .env(
            {
                DASHBOARD_REQUIRES_PROXY_AUTH_ENV_KEY: "true"
                if dashboard_requires_proxy_auth()
                else "false",
                DASHBOARD_VERSION_ENV_KEY: current_dashboard_version(),
                TIMING_DEBUG_ENV: os.environ.get(TIMING_DEBUG_ENV, ""),
                DASHBOARD_PREVIEW_ENV_KEY: "true" if IS_PREVIEW else "",
            }
        )
    )


image = _build_image()

app = modal.App(DASHBOARD_APP_NAME, image=image)

STATIC_DIR = "/app/frontend/dist"


# Underscore-prefixed so it shows up as an auto-managed secret in the Modal
# Secrets UI and is auto-created on first deploy from ~/.modal.toml.
MODAL_CREDS_SECRET_NAME = "_training-gym-modal-creds"

# Routes that must bypass Basic Auth. These endpoints authenticate with their
# own per-run bearer token; the proxy-auth status route must report only the
# Modal-layer setting, independent of dashboard password protection.
PASSWORD_EXEMPT_PATHS = frozenset(
    {
        DASHBOARD_PROXY_AUTH_PATH,
        DASHBOARD_VERSION_PATH,
        "/api/framework-status",
        "/api/training-rollouts",
        "/api/advantage-distributions",
        "/api/timing-events",
        "/api/metric-points",
    }
)

# Only ever the *expected* side of a comparison, so publishing it is safe.
_MISSING_TOKEN_DUMMY = "training-gym-missing-token-dummy-never-issued"

# Suffix on the metric chunk files this container writes (one per process).
METRIC_WRITER_ID = _secrets.token_hex(4)


def _is_local() -> bool:
    """True when we're not running inside a Modal container."""
    return not os.environ.get("MODAL_IS_REMOTE")


def _timing_debug(event: str, **fields: object) -> None:
    if os.environ.get(TIMING_DEBUG_ENV) != "1":
        return
    payload = {"event": event, **fields}
    print(
        f"[timing-debug] {json.dumps(payload, sort_keys=True, default=str)}", flush=True
    )


def _resolve_log_window(
    since_ts: float | None,
    until_ts: float | None,
    *,
    default_since: float,
    default_until: float,
    now: float,
) -> tuple[float, float]:
    """Resolve the ``[since, until]`` log window to concrete epoch seconds.

    ``since_ts`` / ``until_ts`` are the already-parsed request bounds (``None``
    when the caller didn't supply one). Unset bounds fall back to
    ``default_since`` / ``default_until`` (typically the run's lifetime), and
    ``until`` defaults to ``now`` whenever it resolves to a non-positive value
    (e.g. the run hasn't ended). ``since`` is floored at 0 so a negative
    relative age can't reach before the epoch.
    """
    if since_ts is None:
        since_ts = float(default_since)
    if until_ts is None:
        until_ts = float(default_until)
    if until_ts <= 0:
        until_ts = now
    return max(0.0, since_ts), until_ts


def _to_timestamp(secs: float) -> Timestamp:
    """Convert epoch seconds (with fractional part) to a protobuf ``Timestamp``."""
    from google.protobuf.timestamp_pb2 import Timestamp

    ts = Timestamp()
    ts.seconds = int(secs)
    ts.nanos = int((secs - int(secs)) * 1e9)
    return ts


def _parse_log_batches(batches: Iterable[api_pb2.TaskLogsBatch]) -> list[LogEntry]:
    """Flatten Modal ``AppFetchLogs`` batches into ``LogEntry`` dicts."""
    logs: list[LogEntry] = []
    for batch in batches:
        for item in batch.items:
            if not item.data:
                continue
            ts = float(getattr(item, "timestamp", 0) or 0)
            ts_ns = int(getattr(item, "timestamp_ns", 0) or 0)
            entry: LogEntry = {
                "task_id": batch.task_id,
                "line": item.data,
                "fd": int(getattr(item, "file_descriptor", 0) or 0),
                "ts": ts or None,
                "ts_ns": ts_ns or None,
            }
            logs.append(entry)
    return logs


def _compute_next_page(logs: list[LogEntry], limit: int) -> tuple[bool, float | None]:
    """Derive ``(has_more, next_until)`` for keyset pagination."""
    has_more = len(logs) >= limit
    next_until: float | None = None
    if has_more and logs:
        oldest = logs[0]
        oldest_ns = oldest.get("ts_ns") or int(
            (oldest.get("ts") or 0.0) * 1_000_000_000
        )
        next_until = (oldest_ns - 1) / 1_000_000_000
    return has_more, next_until


def ensure_creds_secret(interactive: bool = False) -> bool:
    """Make sure the ``_training-gym-modal-creds`` Modal Secret exists.

    Idempotent: returns True if the secret was already present or if we
    successfully created it from ``~/.modal.toml``. Returns False if we
    can't find creds and ``interactive`` is False (or the user skipped).

    Called both from ``training-gym setup`` and at module-load of this file
    so that ``modal deploy dashboards/app.py`` works without any prior
    setup step — as long as the user has a valid ``~/.modal.toml``.
    """
    if not _is_local():
        # Inside a Modal container we have no credentials and no need to
        # create anything; the secret was already provisioned at deploy.
        return True

    from modal_training_gym.common.config import resolve_modal_creds

    token_id, token_secret, source = resolve_modal_creds()

    if not token_id or not token_secret:
        if not interactive:
            return False
        from getpass import getpass

        print(
            "\nThe dashboard needs Modal workspace credentials to stream "
            "training-run logs into the UI.\n"
            "Couldn't find creds in MODAL_TOKEN_* env vars or "
            "~/.modal.toml — provide them now (or Ctrl-C to skip).\n"
            "Find your tokens at https://modal.com/settings/tokens.\n"
        )
        try:
            token_id = input("MODAL_TOKEN_ID: ").strip()
            token_secret = getpass("MODAL_TOKEN_SECRET (hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping Modal Secret setup.")
            return False
        source = "user input"

    if not token_id or not token_secret:
        return False

    try:
        env_values = {"MODAL_TOKEN_ID": token_id, "MODAL_TOKEN_SECRET": token_secret}
        modal.Secret.objects.create(
            MODAL_CREDS_SECRET_NAME, env_values, allow_existing=True
        )
        modal.Secret.from_name(MODAL_CREDS_SECRET_NAME).update(env_values)
        print(f"Provisioned Modal Secret {MODAL_CREDS_SECRET_NAME!r} (from {source}).")
        return True
    except Exception as exc:
        print(
            f"WARNING: failed to create Modal Secret {MODAL_CREDS_SECRET_NAME!r}: {exc}"
        )
        return False


def _function_secrets() -> list[modal.Secret]:
    """Secrets mounted on the ASGI function.

    The password Secret is optional and mounted only when it exists; absent
    it, ``DASHBOARD_PASSWORD`` is never injected and the dashboard is open.
    """
    secrets = [modal.Secret.from_name(MODAL_CREDS_SECRET_NAME)]
    password = modal.Secret.from_name(DASHBOARD_PASSWORD_SECRET_NAME)
    try:
        password.hydrate()
    except Exception:
        return secrets
    return [*secrets, password]


def set_dashboard_password(password: str) -> None:
    """Set (or clear) the dashboard password.

    Creates the ``_training-gym-dashboard-password`` Secret on demand. An empty
    string disables auth. The new value only takes effect once the dashboard app
    is redeployed, since containers read it from the environment at startup.
    """
    env_values = {"DASHBOARD_PASSWORD": password}
    modal.Secret.objects.create(
        DASHBOARD_PASSWORD_SECRET_NAME, env_values, allow_existing=True
    )
    modal.Secret.from_name(DASHBOARD_PASSWORD_SECRET_NAME).update(env_values)


# Auto-create the creds secret at module-load so `modal deploy
# dashboards/app.py` works out of the box. Gated to local context so it never
# fires inside the deployed container. The password secret is intentionally
# NOT auto-created — no secret means no auth.
if _is_local():
    ensure_creds_secret(interactive=False)


def _run_compact_sync() -> None:
    """Rebuild all summary stores from canonical per-item metadata."""
    from modal_training_gym.utils.metadata import (
        MetadataStore,
        compact_summary_store,
    )

    for summary_store in (
        MetadataStore.TRAINING_RUNS_SUMMARY,
        MetadataStore.TRAIN_RESULTS_SUMMARY,
    ):
        compact_summary_store(summary_store)


# Host document for run-scoped components. It is loaded into a sandboxed
# <iframe> (opaque origin, no network) and only talks to the dashboard through
# postMessage: the parent sends ``props``; the frame reports ``ready``,
# ``rendered``, ``resize`` and ``error``.
_COMPONENT_FRAME_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="dark light">
<style>
  html, body { margin: 0; padding: 0; background: transparent; }
  body { color: #d1d1d1; font: 12px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  /* ``flow-root`` keeps child margins inside #viewer, so its box height is
     the component's real content height. */
  #viewer { min-height: 1px; display: flow-root; }
</style>
</head>
<body>
<div id="viewer"></div>
<script type="module" nonce="__NONCE__">
__COMPONENT__

const __target = document.getElementById("viewer");
let __mounted = null;
function __post(message) {
  window.parent.postMessage({ trainingGymDashboardComponent: true, ...message }, "*");
}
function __render(props) {
  if (__mounted && typeof __mounted.unmount === "function") __mounted.unmount();
  __mounted = null;
  __target.replaceChildren();
  __mounted = mountViewer(__target, props);
}
window.addEventListener("message", (event) => {
  if (event.source !== window.parent) return;
  const data = event.data;
  if (!data || data.trainingGymDashboardComponent !== true || data.type !== "props") return;
  try {
    __render(data.props);
    __post({ type: "rendered" });
  } catch (error) {
    __post({ type: "error", message: String((error && error.message) || error) });
  }
});
window.addEventListener("error", (event) => {
  __post({ type: "error", message: String(event.message || "component error") });
});
// Measure the mount target, not the document: the parent sizes this frame to
// whatever height we report, so `documentElement.scrollHeight` (never smaller
// than the viewport) would ratchet the frame up and never let it shrink back
// when a shorter rollout renders.
function __reportHeight() {
  const box = __target.getBoundingClientRect().height;
  __post({ type: "resize", height: Math.max(box, __target.scrollHeight) });
}
new ResizeObserver(__reportHeight).observe(__target);
__post({ type: "ready" });
</script>
</body>
</html>
"""


class DashboardComponentCompileError(Exception):
    """A user-supplied dashboard component could not be compiled."""


def _component_compiler_script() -> Path:
    installed = Path("/app/frontend/component_bundle.mjs")
    if installed.is_file():
        return installed
    checkout = (
        Path(__file__).resolve().parent.parent
        / "dashboards"
        / "frontend"
        / "component_bundle.mjs"
    )
    if checkout.is_file():
        return checkout
    raise DashboardComponentCompileError("Dashboard component compiler unavailable")


def compile_dashboard_component_source(source: bytes) -> bytes:
    """Compile Svelte component source into a self-contained ES module.

    The source is written to a scratch directory that is discarded afterwards,
    so the compiler never touches the overlay Volume or the caller's paths.
    """
    script = _component_compiler_script()
    with tempfile.TemporaryDirectory(prefix="training-gym-component-") as scratch:
        source_path = Path(scratch) / "Component.svelte"
        output_path = Path(scratch) / "bundle.js"
        source_path.write_bytes(source)
        try:
            completed = subprocess.run(
                ["node", str(script), str(source_path), str(output_path)],
                capture_output=True,
                text=True,
                timeout=90,
            )
        except subprocess.TimeoutExpired as exc:
            raise DashboardComponentCompileError("timed out") from exc
        if completed.returncode != 0 or not output_path.is_file():
            detail = (completed.stderr or completed.stdout or "compile failed").strip()
            raise DashboardComponentCompileError(detail[-1000:])
        return output_path.read_bytes()


# Deliberately no secrets and no Volumes: the compiler only ever sees the
# bytes it is handed, and a runaway or hostile build is bounded by this
# container's limits instead of the dashboard's.
@app.function(timeout=120, cpu=1.0, memory=1024)
def compile_dashboard_component(source: bytes) -> bytes:
    return compile_dashboard_component_source(source)


@app.function(
    schedule=None if IS_PREVIEW else modal.Cron("*/30 * * * *"),
    retries=3,
    timeout=1800,
)
def compact_summaries() -> None:
    """Scheduled compaction of summary stores (every 30 min)."""
    _run_compact_sync()
    print("Compaction complete.")


@app.function(
    schedule=None if IS_PREVIEW else modal.Cron("*/30 * * * *"),
    secrets=_function_secrets(),
    retries=3,
    timeout=1800,
)
def reconcile() -> None:
    """Reconcile orphaned training runs every 30 minutes."""
    from modal_training_gym.common.reconcile import reconcile as _reconcile

    outcome = _reconcile()
    if outcome.runs:
        print(f"Reconciled {len(outcome.runs)} orphaned run(s):")
        for result in outcome.runs:
            print(f"  {result.training_run_id}: {result.reason}")
    else:
        print("No orphaned runs to reconcile.")


@app.function(
    min_containers=0 if IS_PREVIEW else 1,
    secrets=_function_secrets(),
    volumes={DASHBOARD_COMPONENT_VOLUME_MOUNT: dashboard_component_volume},
)
@modal.concurrent(max_inputs=50, target_inputs=20)
@modal.asgi_app(requires_proxy_auth=dashboard_requires_proxy_auth())
def fastapi_app():
    import base64
    import binascii

    from fastapi import (
        FastAPI,
        Header,
        HTTPException,
        Path as FastAPIPath,
    )  # Request and Query imported at module scope
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import (
        FileResponse,
        JSONResponse,
        Response,
        StreamingResponse,
    )
    from fastapi.staticfiles import StaticFiles

    from modal_training_gym.common.modal_urls import modal_app_dashboard_url
    from modal_training_gym.utils.metadata import (
        MetadataStore,
        summary_items_from_payload,
        vol_get,
        vol_get_summary_items_healed,
        vol_put_summary_items,
    )

    web = FastAPI()

    # ── Optional password protection ──────────────────────────────────────
    # When DASHBOARD_PASSWORD is set we gate the whole app behind HTTP Basic
    # Auth (the username is ignored). An empty value means open access.
    dashboard_password = os.environ.get("DASHBOARD_PASSWORD", "")

    def _password_ok(authorization: str | None) -> bool:
        scheme, _, encoded = (authorization or "").partition(" ")
        if scheme.lower() != "basic" or not encoded:
            return False
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            return False
        _user, _, supplied = decoded.partition(":")
        return _secrets.compare_digest(supplied, dashboard_password)

    @web.middleware("http")
    async def require_password(request: Request, call_next):
        if dashboard_password and request.url.path not in PASSWORD_EXEMPT_PATHS:
            if not _password_ok(request.headers.get("Authorization")):
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="training-gym"'},
                )
        return await call_next(request)

    @web.get(DASHBOARD_PROXY_AUTH_PATH)
    async def proxy_auth_status() -> bool:
        return os.environ.get(DASHBOARD_REQUIRES_PROXY_AUTH_ENV_KEY, "false") == "true"

    @web.get(DASHBOARD_VERSION_PATH)
    async def version() -> str:
        return os.environ.get(DASHBOARD_VERSION_ENV_KEY, "")

    cache_ttl_seconds = 30.0
    cache_keys = ("runs", "train_results", "evals")
    # Each entry holds (expires_at, values, loaded_at). ``loaded_at == 0.0``
    # means "never successfully loaded", which lets the very first request block
    # for real data instead of flashing an empty list.
    cache_entries: dict[str, tuple[float, list[JsonDict], float]] = {
        key: (0.0, [], 0.0) for key in cache_keys
    }

    TIMING_CACHE_MAX_RUNS = 64
    TIMING_CACHE_TTL_S = 15.0
    TIMING_FAILURE_RETRY_WINDOW_S = 1.0

    class TimingEntry:
        def __init__(self) -> None:
            self.lanes: JsonDict = {}
            self.file_records: dict[str, TimingFileCache] = {}
            self.persisted_records: dict[str, JsonDict] = {}
            self.pending_records: dict[str, JsonDict] = {}
            self.legacy_records: list[JsonDict] = []
            self.legacy_derived = False
            self.touched_at = time.monotonic()
            self.read_at: float | None = None
            self.stale = False
            self.lock = asyncio.Lock()

        @property
        def fresh(self) -> bool:
            if self.read_at is None:
                return False
            ttl = TIMING_FAILURE_RETRY_WINDOW_S if self.stale else TIMING_CACHE_TTL_S
            return time.monotonic() - self.read_at < ttl

    timing_cache: dict[str, TimingEntry] = {}
    timing_cache_lock = asyncio.Lock()

    # ── Mirrored scalar metrics ──────────────────────────────────────────
    # One in-memory table per run; a background loop writes dirty chunks to
    # the volume every few seconds (sooner on the final batch). Each
    # container writes its own copy of a chunk (``chunk-000001-<writer>``) and
    # readers merge every copy (newest ingest per step wins), so autoscaled
    # replicas can't clobber each other.
    METRIC_FLUSH_INTERVAL_S = 5.0
    METRIC_READ_TTL_S = 10.0
    METRIC_CACHE_MAX_RUNS = 64

    class MetricEntry:
        def __init__(self) -> None:
            self.metrics = RunMetrics()
            self.dirty: set[str] = set()
            self.loaded_at: float | None = None
            self.lock = asyncio.Lock()

    metric_cache: dict[str, MetricEntry] = {}

    async def _metric_entry_for(training_run_id: str) -> MetricEntry:
        entry = metric_cache.pop(training_run_id, None) or MetricEntry()
        metric_cache[training_run_id] = entry  # re-insert: dict order is LRU
        while len(metric_cache) > METRIC_CACHE_MAX_RUNS:
            old_id, old = next(iter(metric_cache.items()))
            del metric_cache[old_id]
            async with old.lock:
                await _flush_metrics(old_id, old)
        return entry

    async def _load_metrics(training_run_id: str, entry: MetricEntry) -> None:
        """Merge the volume's chunks under in-memory points. Caller holds the lock."""
        for chunk in await _metadata_vol_list(
            metric_series_store(training_run_id), is_async=True
        ):
            entry.metrics.load_chunk(chunk)
        entry.loaded_at = time.monotonic()

    async def _flush_metrics(training_run_id: str, entry: MetricEntry) -> None:
        """Write every dirty chunk in one commit. Caller holds the lock."""
        if entry.dirty:
            await _metadata_vol_put_many(
                metric_series_store(training_run_id),
                {
                    f"{chunk}-{METRIC_WRITER_ID}": entry.metrics.chunk_payload(chunk)
                    for chunk in entry.dirty
                },
                is_async=True,
            )
            entry.dirty.clear()

    async def _flush_dirty_metrics() -> None:
        for training_run_id, entry in list(metric_cache.items()):
            if entry.dirty:
                try:
                    async with entry.lock:
                        await _flush_metrics(training_run_id, entry)
                except Exception as exc:
                    print(f"[dashboard] metric flush failed: {exc}", flush=True)

    async def _metric_flush_loop(stop: asyncio.Event) -> None:
        while not stop.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), METRIC_FLUSH_INTERVAL_S)
            await _flush_dirty_metrics()

    metric_flush_stop: asyncio.Event | None = None

    @web.on_event("startup")
    async def _start_metric_flush_loop() -> None:
        nonlocal metric_flush_stop
        metric_flush_stop = asyncio.Event()
        asyncio.create_task(_metric_flush_loop(metric_flush_stop))

    @web.on_event("shutdown")
    async def _stop_metric_flush_loop() -> None:
        if metric_flush_stop is not None:
            metric_flush_stop.set()
        await _flush_dirty_metrics()

    def _rebuild_timing_lanes(entry: TimingEntry) -> None:
        records = dict(entry.persisted_records)
        for storage_key, record in entry.pending_records.items():
            stored = records.get(storage_key)
            if (
                stored is None
                or record.get("final", False)
                or not stored.get("final", False)
            ):
                records[storage_key] = record
        grouped: dict[int | None, list[JsonDict]] = {}
        for record in [*entry.legacy_records, *records.values()]:
            rollout_id = cast(int | None, record["rollout_id"])
            grouped.setdefault(rollout_id, []).append(record)
        entry.lanes = {
            ("pre-loop" if rollout_id is None else str(rollout_id)): rollout_lanes(
                rollout_records
            )
            for rollout_id, rollout_records in sorted(
                grouped.items(), key=lambda item: (item[0] is None, item[0] or 0)
            )
        }
        if entry.legacy_derived:
            entry.lanes["metadata"] = {"legacy_derived": True}

    cache_locks = {key: asyncio.Lock() for key in cache_keys}
    # Hold strong refs to background refresh tasks so they aren't GC'd mid-flight.
    refresh_tasks: set[asyncio.Task[list[JsonDict]]] = set()

    class _HashedAssets(StaticFiles):
        """Serve Vite's content-hashed bundles as immutable.

        The filename changes whenever the contents do, so a cached copy can
        never be stale; pairing this with a revalidated ``index.html`` (see the
        SPA fallback) is what keeps a deploy from ever serving old HTML that
        points at a bundle this image no longer has.
        """

        def file_response(self, *args, **kwargs):
            response = super().file_response(*args, **kwargs)
            if response.status_code == 200:
                response.headers["Cache-Control"] = (
                    "public, max-age=31536000, immutable"
                )
            return response

    web.mount("/assets", _HashedAssets(directory=f"{STATIC_DIR}/assets"), name="assets")

    # ── Shared Modal client ───────────────────────────────────────────────
    # Opens a client at startup and reuses it across all requests.
    modal_client: _Client | None = None
    modal_client_lock = asyncio.Lock()

    async def get_modal_client() -> _Client | None:
        """Return the shared, connection-open Modal client.

        Creates it once (guarded by a lock) and reuses it thereafter. Returns
        ``None`` when no credentials are configured.
        """
        nonlocal modal_client
        if modal_client is not None:
            return modal_client

        token_id = os.environ.get("MODAL_TOKEN_ID", "")
        token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
        if not token_id or not token_secret:
            return None

        from modal.client import _Client

        async with modal_client_lock:
            if modal_client is None:
                modal_client = await _Client.from_credentials(token_id, token_secret)
            return modal_client

    @web.on_event("startup")
    async def _open_modal_client() -> None:
        # Warm the shared client at startup. Failures here are non-fatal:
        # endpoints fall back to lazy init and surface error if creds are missing.
        try:
            await get_modal_client()
        except Exception:
            pass

    async def _flush_unpersisted_timing(
        training_run_id: str, entry: TimingEntry
    ) -> bool:
        async with entry.lock:
            records = {
                storage_key: record
                for storage_key, record in entry.pending_records.items()
                if record.get("final", False)
                or not entry.persisted_records.get(storage_key, {}).get("final", False)
            }
            if not records:
                return False
            await _metadata_vol_put_many(
                RoleTimingRecord.store(training_run_id),
                records,
                is_async=True,
            )
            entry.persisted_records.update(records)
            for storage_key in records:
                entry.pending_records.pop(storage_key, None)
                entry.file_records.pop(
                    f"{RoleTimingRecord.store(training_run_id)}/{storage_key}.json",
                    None,
                )
            _rebuild_timing_lanes(entry)
            _timing_debug(
                "timing_flush",
                training_run_id=training_run_id,
                written=sorted(records),
            )
            return True

    async def _timing_entry_for(training_run_id: str) -> TimingEntry:
        evicted: tuple[str, TimingEntry] | None = None
        async with timing_cache_lock:
            entry = timing_cache.get(training_run_id)
            if entry is not None:
                entry.touched_at = time.monotonic()
                return entry
            if len(timing_cache) >= TIMING_CACHE_MAX_RUNS:
                evictable = [
                    (other.touched_at, run_id)
                    for run_id, other in timing_cache.items()
                    if not other.lock.locked()
                ]
                if evictable:
                    evicted_run_id = min(evictable)[1]
                    evicted = (evicted_run_id, timing_cache.pop(evicted_run_id))
            entry = timing_cache.setdefault(training_run_id, TimingEntry())
        if evicted is not None:
            evicted_run_id, evicted_entry = evicted
            try:
                await _flush_unpersisted_timing(evicted_run_id, evicted_entry)
            except Exception as exc:
                _timing_debug(
                    "timing_flush_failed",
                    training_run_id=evicted_run_id,
                    detail=str(exc),
                )
        return entry

    async def refresh_cache(key: str, loader: SummaryLoader) -> list[JsonDict]:
        async with cache_locks[key]:
            now = time.monotonic()
            expires_at, values, loaded_at = cache_entries[key]
            if now < expires_at:
                return values
            try:
                values = await loader()
                cache_entries[key] = (now + cache_ttl_seconds, values, now)
            except Exception:
                # Keep serving the last known data and back off so a slow/failing
                # loader (e.g. a heavy summary rebuild) can't be retried on every
                # request — it must never block or break the endpoint.
                cache_entries[key] = (now + cache_ttl_seconds, values, loaded_at)
            return values

    def invalidate_cache(key: str) -> None:
        # Force the next read to revalidate, but keep the last values and the
        # "loaded once" marker so it refreshes in the background instead of
        # blocking on a cold rebuild.
        _expires_at, values, loaded_at = cache_entries[key]
        cache_entries[key] = (0.0, values, loaded_at)

    async def get_cached_list(key: str, loader: SummaryLoader) -> list[JsonDict]:
        now = time.monotonic()
        expires_at, values, loaded_at = cache_entries[key]
        if now < expires_at:
            return values

        # Nothing cached yet: block once on the loader so the first paint has
        # real data rather than an empty flash.
        if loaded_at == 0.0:
            return await refresh_cache(key, loader)

        # Stale-while-revalidate: return the last good data immediately and
        # refresh in the background. The expensive runs rebuild then happens off
        # the request path, so the UI is never left waiting on it.
        if not cache_locks[key].locked():
            task = asyncio.create_task(refresh_cache(key, loader))
            refresh_tasks.add(task)
            task.add_done_callback(refresh_tasks.discard)
        return values

    def add_modal_app_urls(
        items: list[JsonDict],
    ) -> tuple[list[JsonDict], bool]:
        updated: list[JsonDict] = []
        changed = False
        for item in items:
            new_item = dict(item)
            if not new_item.get("modal_app_url"):
                app_id = str(new_item.get("modal_app_id", "") or "")
                if app_id:
                    new_item["modal_app_url"] = modal_app_dashboard_url(app_id)
                    changed = True
            updated.append(new_item)
        return updated, changed

    def merge_missing_fields(
        merged: JsonDict, source: JsonDict, fields: tuple[str, ...]
    ) -> None:
        for field in fields:
            value = source.get(field)
            if field not in merged and value is not None:
                merged[field] = value

    async def fetch_by_id(
        store: MetadataStore, key: str
    ) -> tuple[str, JsonDict | None]:
        try:
            return key, await run_in_threadpool(vol_get, store, key)
        except KeyError:
            return key, None

    async def fetch_all_by_id(
        store: MetadataStore, keys: list[str]
    ) -> dict[str, JsonDict]:
        fetched = await asyncio.gather(*(fetch_by_id(store, key) for key in keys))
        return {key: value for key, value in fetched if value is not None}

    async def load_eval_summaries() -> list[JsonDict]:
        try:
            payload = await run_in_threadpool(vol_get, MetadataStore.EVALS, "summary")
        except KeyError:
            return []

        summaries = summary_items_from_payload(payload, payload_key="summaries")
        if not summaries:
            return []

        results_by_id = await fetch_all_by_id(
            MetadataStore.EVAL_RESULTS,
            [str(s.get("eval_id") or "") for s in summaries if s.get("eval_id")],
        )
        configs_by_id = await fetch_all_by_id(
            MetadataStore.EVAL_CONFIGS,
            [
                str(s.get("eval_config_id") or "")
                for s in summaries
                if s.get("eval_config_id")
            ],
        )

        enriched: list[JsonDict] = []
        for summary in summaries:
            merged = dict(summary)
            result = results_by_id.get(str(summary.get("eval_id") or ""))
            if result:
                merge_missing_fields(merged, result, ("config", "status", "model_name"))
            eval_config = configs_by_id.get(str(summary.get("eval_config_id") or ""))
            if eval_config:
                merged["eval_config"] = eval_config
                merge_missing_fields(
                    merged,
                    eval_config,
                    (
                        "dataset_name",
                        "eval_fn_name",
                        "prompt_column",
                        "generate_kwargs",
                    ),
                )
            enriched.append(merged)
        return enriched

    async def load_list_summary(
        summary_store: MetadataStore,
    ) -> list[JsonDict]:
        items = await run_in_threadpool(vol_get_summary_items_healed, summary_store)
        if not items:
            return []
        items, changed = add_modal_app_urls(items)
        if changed:
            await run_in_threadpool(vol_put_summary_items, summary_store, items)
        return items

    async def load_runs() -> list[JsonDict]:
        run_records = await load_list_summary(MetadataStore.TRAINING_RUNS_SUMMARY)
        try:
            result_records = await load_list_summary(
                MetadataStore.TRAIN_RESULTS_SUMMARY
            )
        except Exception:
            # A result-store outage must not hide otherwise healthy run records.
            result_records = []
        return [
            summary.model_dump(mode="json")
            for summary in build_run_summaries(run_records, result_records)
        ]

    async def load_train_results() -> list[JsonDict]:
        return await load_list_summary(MetadataStore.TRAIN_RESULTS_SUMMARY)

    def _bearer_token(authorization: str | None) -> str:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer":
            return ""
        return token.strip()

    async def _require_framework_status_token(
        training_run_id: str,
        authorization: str | None,
    ) -> None:
        """403 unless ``authorization`` carries the run's status token.

        Handlers must call this *before* any lookup that can 404, or the
        status code tells an anonymous caller which run ids exist. For the
        same reason an unknown run is indistinguishable from a wrong token.
        """
        try:
            expected_token = str(
                (
                    await run_in_threadpool(
                        vol_get,
                        MetadataStore.FRAMEWORK_STATUS_TOKENS,
                        training_run_id,
                    )
                ).get("token", "")
            )
        except KeyError:
            expected_token = ""
        supplied = _bearer_token(authorization)
        if not expected_token:
            _secrets.compare_digest(supplied, _MISSING_TOKEN_DUMMY)
            raise HTTPException(status_code=403, detail="Invalid status token")
        if not _secrets.compare_digest(supplied, expected_token):
            raise HTTPException(status_code=403, detail="Invalid status token")

    async def _get_run_or_404(training_run_id: str) -> TrainingRun:
        try:
            return await TrainingRun.from_id(training_run_id, is_async=True)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"TrainingRun {training_run_id!r} not found",
            )

    # ── Training runs ────────────────────────────────────────────────────

    # The run list renders none of the full config or the per-step timing maps —
    # the per-run detail endpoint serves those — yet they are most of the list
    # payload (``config`` alone is ~60% of it), so the list drops them rather
    # than shipping them on every poll. ``metadata`` stays: the list's group and
    # tag columns fall back to it for runs that predate ``group_tags``.
    run_list_excluded_fields = {"config", "step_times", "substep_times"}

    def _requested_facets(
        status: list[str] | None,
        recipe: list[str] | None,
        group: list[str] | None,
        training_type: list[str] | None,
    ) -> dict[str, set[str]]:
        selected = {
            "status": status,
            "recipe": recipe,
            "group": group,
            "training_type": training_type,
        }
        return {name: set(values) for name, values in selected.items() if values}

    async def load_run_summaries() -> list[RunSummary]:
        try:
            data = await get_cached_list("runs", load_runs)
        except Exception:
            data = []
        return [
            RunSummary.model_validate(item) for item in data if isinstance(item, dict)
        ]

    # ``response_model`` is left off: FastAPI ignores ``response_model_exclude``
    # for sequence response models, so the exclusion is applied here instead.
    @web.get("/api/runs")
    async def runs(
        request: Request,
        since: int | None = None,
        limit: int | None = None,
        offset: int = 0,
        q: str = "",
        status: FacetParam = None,
        recipe: FacetParam = None,
        group: FacetParam = None,
        training_type: FacetParam = None,
    ):
        if limit is not None and limit < 1:
            raise HTTPException(status_code=400, detail="Limit must be positive")
        if offset < 0:
            raise HTTPException(status_code=400, detail="Offset must not be negative")
        summaries = await load_run_summaries()
        # Facet params are multi-select unions, declared above: taking them here
        # too would intersect the union with whichever repeated value ``get``
        # happens to return.
        filters = {
            name: request.query_params.get(name, "")
            for name, metadata in run_list_field_metadata().items()
            if metadata.get("filterable") and name not in FACET_NAMES
        }
        facets = _requested_facets(status, recipe, group, training_type)
        filtered = filter_run_summaries(
            summaries,
            filters=filters,
            facets=facets,
            query=q,
            since=since,
            limit=limit,
            offset=offset,
            # The list shows runs newest-first, and paging is only stable if the
            # server orders by the same key: sorting by update time reshuffles
            # the pages under the client whenever a run reports progress.
            sort_by="created",
        )
        return JSONResponse(
            [
                summary.model_dump(mode="json", exclude=run_list_excluded_fields)
                for summary in filtered
            ]
        )

    # Declared before ``/api/runs/{training_run_id}`` so "counts" isn't read as a
    # run id. The page's totals and filter-chip counts come from here, since a
    # paged list can't count runs the client hasn't loaded. Chip counts cover
    # every run (they're what the chips would select); ``matching`` counts the
    # current query, which is how many rows paging can still reach.
    @web.get("/api/runs/counts")
    async def run_counts(
        q: str = "",
        status: FacetParam = None,
        recipe: FacetParam = None,
        group: FacetParam = None,
        training_type: FacetParam = None,
    ):
        summaries = await load_run_summaries()
        counts = count_run_facets(summaries)
        counts["matching"] = len(
            filter_run_summaries(
                summaries,
                facets=_requested_facets(status, recipe, group, training_type),
                query=q,
            )
        )
        return counts

    @web.get("/api/runs/{training_run_id}", response_model=RunSummary)
    async def get_run(training_run_id: str):
        try:
            run = await TrainingRun.from_id(training_run_id, is_async=True)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Training run {training_run_id!r} not found",
            )
        try:
            result = await vol_get(
                MetadataStore.TRAIN_RESULTS,
                training_run_id,
                is_async=True,
            )
        except KeyError:
            result = None
        return build_run_summary(run.model_dump(mode="json"), result)

    def _component_manifest(
        run: TrainingRun, component_type: str, digest: str | None = None
    ) -> JsonDict:
        """Resolve a run's component manifest from run metadata.

        The source and an association copy are both mounted from the overlay
        Volume. Metadata remains the authoritative list of components because
        it is already part of the run record returned by the dashboard API.
        When several names register the same component type, the most
        recently attached one wins (``add_dashboard_component`` re-inserts an
        entry on replace so insertion order is attachment order). With
        ``digest`` the entry carrying exactly that ``sha256`` is returned, so
        a URL that names an artifact can never be answered with another one.
        """
        try:
            kind = DashboardComponent(component_type).value
        except ValueError as exc:
            raise HTTPException(
                status_code=404, detail="Unknown dashboard component"
            ) from exc
        components = (run.metadata or {}).get("dashboard_components")
        if not isinstance(components, dict):
            raise HTTPException(
                status_code=404, detail="No dashboard component attached"
            )
        for value in reversed(list(components.values())):
            if not isinstance(value, dict) or value.get("type") != kind:
                continue
            manifest = dict(value)
            if not manifest.get("path") or not manifest.get("sha256"):
                continue
            if digest is not None and manifest["sha256"] != digest:
                continue
            return manifest
        raise HTTPException(status_code=404, detail="No dashboard component attached")

    def _component_source_path(manifest: JsonDict) -> Path:
        relative = str(manifest.get("path", ""))
        root = Path(DASHBOARD_COMPONENT_VOLUME_MOUNT).resolve()
        source = (root / relative).resolve()
        if root != source and root not in source.parents:
            raise HTTPException(
                status_code=500, detail="Invalid dashboard component path"
            )
        if not source.is_file():
            raise HTTPException(
                status_code=404, detail="Dashboard component source unavailable"
            )
        return source

    def _read_component_source(source: Path, digest: str) -> bytes:
        """Read source whose content matches the manifest's sha256.

        Run metadata selects the component, but the overlay Volume is the
        only thing that should decide what code ships to the browser.
        """
        data = source.read_bytes()
        if len(data) > MAX_COMPONENT_BYTES:
            raise HTTPException(
                status_code=413, detail="Dashboard component source is too large"
            )
        actual = hashlib.sha256(data).hexdigest()
        if not hmac.compare_digest(actual, digest):
            raise HTTPException(
                status_code=409,
                detail="Dashboard component source does not match its manifest",
            )
        return data

    component_compile_locks: dict[str, asyncio.Lock] = {}

    async def _compile_component_bundle(
        source: Path, digest: str, bundle: Path
    ) -> None:
        """Compile ``source`` into ``bundle`` exactly once per digest.

        Concurrent first requests for the same digest wait on a shared lock;
        the finished bundle is installed atomically so readers never observe a
        partial file. The lock is dropped once the compile settles so the map
        does not grow with every component revision.
        """
        lock = component_compile_locks.setdefault(digest, asyncio.Lock())
        try:
            async with lock:
                if bundle.is_file():
                    return
                data = await run_in_threadpool(_read_component_source, source, digest)
                try:
                    if _is_local():
                        compiled = await run_in_threadpool(
                            compile_dashboard_component_source, data
                        )
                    else:
                        compiled = await compile_dashboard_component.remote.aio(data)
                except DashboardComponentCompileError as exc:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Dashboard component failed to compile: {exc}",
                    ) from exc
                staging = bundle.with_name(f"{bundle.stem}.{_secrets.token_hex(8)}.js")
                try:
                    await run_in_threadpool(staging.write_bytes, compiled)
                    os.replace(staging, bundle)
                finally:
                    staging.unlink(missing_ok=True)
        finally:
            if not lock.locked() and component_compile_locks.get(digest) is lock:
                del component_compile_locks[digest]

    _CLOSE_SCRIPT_RE = re.compile(r"</(script)", re.IGNORECASE)
    _COMPONENT_DIGEST_RE = re.compile(r"[0-9a-f]{64}")

    def _component_frame_html(bundle_js: str, nonce: str) -> str:
        """Wrap a compiled component in a self-contained host document.

        The document is served to a sandboxed ``<iframe>`` so the component
        runs in an opaque origin: it receives props over ``postMessage`` and
        cannot read dashboard storage or issue authenticated requests.
        """
        script = _CLOSE_SCRIPT_RE.sub(r"<\\/\1", bundle_js)
        return _COMPONENT_FRAME_TEMPLATE.replace("__NONCE__", nonce).replace(
            "__COMPONENT__", script
        )

    async def _refresh_dashboard_component_volume() -> None:
        """Refresh the mounted overlay before reading a newly registered component.

        Modal volumes are mounted from a snapshot.  A training client can attach a
        component after the dashboard container has started, so the container must
        reload the volume to make that version visible at the mount point.  Keep
        this best-effort for local/test contexts where ``reload`` is unavailable.
        """
        if _is_local():
            # ``reload`` is only meaningful inside a Modal container; the source
            # path check below still works for local test mounts.
            return
        try:
            await run_in_threadpool(dashboard_component_volume.reload)
        except (AttributeError, RuntimeError, Error):
            return

    @web.get("/api/runs/{training_run_id}/dashboard-components/{component_type}")
    async def get_dashboard_component_manifest(
        training_run_id: str, component_type: str
    ):
        run = await _get_run_or_404(training_run_id)
        manifest = _component_manifest(run, component_type)
        return JSONResponse(manifest)

    @web.get(
        "/api/runs/{training_run_id}/dashboard-components/{component_type}"
        "/{digest}/frame.html"
    )
    async def get_dashboard_component_frame(
        training_run_id: str, component_type: str, digest: str
    ):
        if not _COMPONENT_DIGEST_RE.fullmatch(digest):
            raise HTTPException(status_code=404, detail="Unknown dashboard component")
        run = await _get_run_or_404(training_run_id)
        manifest = _component_manifest(run, component_type, digest)
        await _refresh_dashboard_component_volume()
        source = _component_source_path(manifest)
        cache_root = Path("/tmp/training-gym-dashboard-components")
        cache_root.mkdir(parents=True, exist_ok=True)
        bundle = cache_root / f"{digest}.js"

        if not bundle.is_file():
            await _compile_component_bundle(source, digest, bundle)

        nonce = _secrets.token_urlsafe(16)
        bundle_js = await run_in_threadpool(bundle.read_text, "utf-8")
        csp = "; ".join(
            [
                "default-src 'none'",
                f"script-src 'nonce-{nonce}'",
                "style-src 'unsafe-inline'",
                "img-src data: blob:",
                "font-src data:",
                "connect-src 'none'",
                "frame-ancestors 'self'",
                "base-uri 'none'",
                "form-action 'none'",
                "sandbox allow-scripts",
            ]
        )
        return Response(
            _component_frame_html(bundle_js, nonce),
            media_type="text/html",
            headers={
                "Cache-Control": "private, max-age=31536000, immutable",
                "Content-Security-Policy": csp,
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @web.post("/api/framework-status")
    async def framework_status(
        update: FrameworkStatusUpdate,
        authorization: str | None = Header(default=None),
    ):
        await _require_framework_status_token(update.training_run_id, authorization)
        run = await _get_run_or_404(update.training_run_id)

        status = await run_in_threadpool(run.apply_framework_status, update)
        if status is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported framework status {update.phase!r} "
                    f"for {run.framework.value}"
                ),
            )
        await run.save(is_async=True)
        invalidate_cache("runs")
        if run.status in {
            TrainingRunStatus.STOPPED,
            TrainingRunStatus.CANCELLED,
            TrainingRunStatus.COMPLETED,
            TrainingRunStatus.FAILED,
        }:
            try:
                await _flush_unpersisted_timing(
                    update.training_run_id,
                    await _timing_entry_for(update.training_run_id),
                )
            except Exception as exc:
                _timing_debug(
                    "timing_flush_failed",
                    training_run_id=update.training_run_id,
                    detail=str(exc),
                )
            if metric_entry := metric_cache.get(update.training_run_id):
                try:
                    async with metric_entry.lock:
                        await _flush_metrics(update.training_run_id, metric_entry)
                except Exception as exc:
                    print(f"[dashboard] metric flush failed: {exc}", flush=True)
        return JSONResponse({"status": "ok", "framework_status": status.value})

    @web.post("/api/metric-points")
    async def metric_points(
        batch: MetricPointsBatch,
        authorization: str | None = Header(default=None),
    ):
        await _require_framework_status_token(batch.training_run_id, authorization)
        entry = await _metric_entry_for(batch.training_run_id)
        async with entry.lock:
            if entry.loaded_at is None:
                try:
                    await _load_metrics(batch.training_run_id, entry)
                except Exception:
                    raise HTTPException(
                        status_code=503, detail="Metric store unavailable"
                    )
            entry.dirty |= entry.metrics.merge_points(batch.points)
            if batch.final:
                await _flush_metrics(batch.training_run_id, entry)
        return JSONResponse({"status": "ok", "accepted": len(batch.points)})

    @web.get("/api/runs/{training_run_id}/metrics")
    async def get_run_metrics(training_run_id: str = FastAPIPath()):
        run = await _get_run_or_404(training_run_id)
        entry = await _metric_entry_for(training_run_id)
        async with entry.lock:
            stale = False
            if (
                entry.loaded_at is None
                or time.monotonic() - entry.loaded_at >= METRIC_READ_TTL_S
            ):
                try:
                    await _load_metrics(training_run_id, entry)
                except Exception:
                    stale = True
            if run.status in {
                TrainingRunStatus.STOPPED,
                TrainingRunStatus.CANCELLED,
                TrainingRunStatus.COMPLETED,
                TrainingRunStatus.FAILED,
            }:
                try:
                    await _flush_metrics(training_run_id, entry)
                except Exception:
                    stale = True
            series = metric_series(entry.metrics.table, MAX_POINTS_PER_KEY)
        return JSONResponse({"series": series, "stale": stale})

    # ── Training rollouts ────────────────────────────────────────────────

    @web.post("/api/training-rollouts")
    async def training_rollout(
        result: TrainingRolloutResult,
        authorization: str | None = Header(default=None),
    ):
        await _require_framework_status_token(result.training_run_id, authorization)
        run = await _get_run_or_404(result.training_run_id)

        await run_in_threadpool(result.save)
        run.record_latest_rollout(result)
        await run.save(is_async=True)
        invalidate_cache("runs")

        return JSONResponse(
            {"status": "ok", "rollout_id": result.rollout_id, "mean": result.mean}
        )

    @web.get(
        "/api/runs/{training_run_id}/rollouts",
        response_model=list[TrainingRolloutSummary],
    )
    async def list_run_rollouts(training_run_id: str):
        return await run_in_threadpool(
            TrainingRolloutResult.list_summaries_for_run, training_run_id
        )

    @web.get("/api/runs/{training_run_id}/rollouts/{rollout_id}")
    async def get_run_rollout(training_run_id: str, rollout_id: int):
        key = f"{training_run_id}__{int(rollout_id):08d}"
        try:
            data = await run_in_threadpool(
                vol_get, MetadataStore.TRAINING_ROLLOUTS, key
            )
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Rollout {rollout_id} for run {training_run_id!r} not found",
            )

        if isinstance(data, dict):
            _apply_parsed(data.get("samples"))
        return JSONResponse(data)

    # ── Per-group advantage distributions ────────────────────────────────

    @web.post("/api/advantage-distributions")
    async def advantage_distribution(
        shard: AdvantageDistribution,
        authorization: str | None = Header(default=None),
    ):
        await _require_framework_status_token(shard.training_run_id, authorization)
        await _get_run_or_404(shard.training_run_id)

        await shard.save(is_async=True)
        return JSONResponse(
            {
                "status": "ok",
                "rollout_id": shard.rollout_id,
                "dp_rank": shard.dp_rank,
                "samples": len(shard.samples),
            }
        )

    @web.get("/api/runs/{training_run_id}/advantages")
    async def list_run_advantages(training_run_id: str):
        steps = await run_in_threadpool(
            AdvantageDistribution.list_steps_for_run, training_run_id
        )
        return JSONResponse(steps)

    @web.get("/api/runs/{training_run_id}/advantages/{rollout_id}")
    async def get_run_advantages(training_run_id: str, rollout_id: int):
        merged = await run_in_threadpool(
            AdvantageDistribution.merged_for_step, training_run_id, rollout_id
        )
        if merged is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No advantage distribution for rollout {rollout_id} "
                    f"of run {training_run_id!r}"
                ),
            )
        return JSONResponse(merged)

    @web.post("/api/timing-events")
    async def timing_event(
        record: RoleTimingRecord,
        authorization: str | None = Header(default=None),
    ):
        await _require_framework_status_token(record.training_run_id, authorization)

        entry = await _timing_entry_for(record.training_run_id)
        async with entry.lock:
            record_payload = record.model_dump(mode="json")
            _timing_debug(
                "timing_arrival",
                training_run_id=record.training_run_id,
                storage_key=record.storage_key,
                lane_start_unix_s=record.lane_start_unix_s,
                final=record.final,
                phases=sorted(record.phases),
                total_count=sum(phase.count for phase in record.phases.values()),
            )
            if record.final:
                entry.pending_records[record.storage_key] = record_payload
                _rebuild_timing_lanes(entry)
                try:
                    await _metadata_vol_put_many(
                        RoleTimingRecord.store(record.training_run_id),
                        {record.storage_key: record_payload},
                        is_async=True,
                    )
                except Exception:
                    _timing_debug(
                        "timing_flush_failed",
                        training_run_id=record.training_run_id,
                        written=[],
                        detail="final timing POST write failed",
                    )
                    raise
                else:
                    entry.persisted_records[record.storage_key] = record_payload
                    entry.pending_records.pop(record.storage_key, None)
            else:
                stored = entry.persisted_records.get(record.storage_key)
                if stored is None or not stored.get("final", False):
                    entry.pending_records[record.storage_key] = record_payload
            _rebuild_timing_lanes(entry)
            entry.file_records.pop(
                f"{RoleTimingRecord.store(record.training_run_id)}/"
                f"{record.storage_key}.json",
                None,
            )
        return JSONResponse({"status": "ok"})

    async def _read_run_timings(
        training_run_id: str,
        entry: TimingEntry,
    ) -> tuple[JsonDict, bool]:
        listed, had_read_failures = await vol_list_metadata_with_failures(
            RoleTimingRecord.store(training_run_id),
            is_async=True,
        )
        if had_read_failures:
            return {}, True
        current = {item["path"]: (item["mtime"], item["size"]) for item in listed}
        unchanged = {
            path: cached
            for path, cached in entry.file_records.items()
            if current.get(path) == (cached["mtime"], cached["size"])
        }
        changed = [item for item in listed if item["path"] not in unchanged]
        read_results = await bounded_gather_with_retries(
            [
                lambda item=item: _metadata_vol_get(
                    RoleTimingRecord.store(training_run_id),
                    item["path"].rsplit("/", 1)[-1][:-5],
                    is_async=True,
                )
                for item in changed
            ]
        )
        updated: dict[str, TimingFileCache] = {}
        for item, result in zip(changed, read_results, strict=True):
            if isinstance(result, (KeyError, FileNotFoundError)):
                cached = entry.file_records.get(item["path"])
                if cached is not None:
                    updated[item["path"]] = cached
                continue
            if isinstance(result, (json.JSONDecodeError, UnicodeDecodeError)):
                updated[item["path"]] = {
                    "mtime": current[item["path"]][0],
                    "size": current[item["path"]][1],
                    "checked_at": time.time(),
                    "record": None,
                }
                continue
            if isinstance(result, BaseException):
                return {}, True
            validated = validate_timing_record(result)
            updated[item["path"]] = {
                "mtime": current[item["path"]][0],
                "size": current[item["path"]][1],
                "checked_at": time.time(),
                "record": validated,
            }
        entry.file_records = {**unchanged, **updated}
        stored_records = {
            path.rsplit("/", 1)[-1][:-5]: cached["record"]
            for path, cached in entry.file_records.items()
            if cached["record"] is not None
        }
        for storage_key, record in entry.persisted_records.items():
            current_record = stored_records.get(storage_key)
            if current_record is None or (
                record.get("final", False) and not current_record.get("final", False)
            ):
                stored_records[storage_key] = record
        entry.persisted_records = stored_records
        for storage_key in list(entry.pending_records):
            pending = entry.pending_records[storage_key]
            if entry.persisted_records.get(storage_key, {}).get(
                "final", False
            ) and not pending.get("final", False):
                entry.pending_records.pop(storage_key, None)
        entry.legacy_derived = False
        entry.legacy_records = []
        if (
            not listed
            and not had_read_failures
            and not entry.persisted_records
            and not entry.pending_records
        ):
            try:
                run = await _get_run_or_404(training_run_id)
            except Error:
                return {}, True
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
            else:
                legacy_records = legacy_run_to_records(run.substep_times)
                entry.legacy_derived = bool(legacy_records)
                entry.legacy_records = legacy_records
        _rebuild_timing_lanes(entry)
        return entry.lanes, had_read_failures

    @web.get("/api/runs/{training_run_id}/timings")
    async def get_run_timings(
        training_run_id: str = FastAPIPath(),
    ):
        run = await _get_run_or_404(training_run_id)
        entry = await _timing_entry_for(training_run_id)

        def response() -> JsonDict:
            if not entry.stale:
                return entry.lanes
            metadata_value = entry.lanes.get("metadata")
            metadata = metadata_value if isinstance(metadata_value, dict) else {}
            return {
                **entry.lanes,
                "metadata": {**metadata, "timing_stale": True},
            }

        if not entry.fresh:
            async with entry.lock:
                if not entry.fresh:
                    _, had_read_failures = await _read_run_timings(
                        training_run_id, entry
                    )
                    if had_read_failures:
                        entry.read_at = time.monotonic()
                        entry.stale = True
                    else:
                        entry.read_at = time.monotonic()
                        entry.stale = False
        if (
            run.status
            in {
                TrainingRunStatus.STOPPED,
                TrainingRunStatus.CANCELLED,
                TrainingRunStatus.COMPLETED,
                TrainingRunStatus.FAILED,
            }
            and entry.pending_records
        ):
            try:
                await _flush_unpersisted_timing(training_run_id, entry)
            except Exception as exc:
                _timing_debug(
                    "timing_flush_failed",
                    training_run_id=training_run_id,
                    detail=str(exc),
                )
        return JSONResponse(response())

    # ── Live Modal log stream (SSE, pure pass-through) ───────────────────

    @web.get("/api/runs/{training_run_id}/logs/stream")
    async def stream_run_logs(
        training_run_id: str,
        request: Request,
        search: str = "",
        max_lines_per_sec: int = 0,
    ):
        """Server-Sent Events stream of the underlying Modal app's logs.

        Pure pass-through: we open a long-poll ``AppGetLogs`` stream against
        the run's ``modal_app_id`` and forward each batch as an SSE ``data``
        event. Nothing is persisted on the dashboard side.

        Query params:
          - ``search``: case-insensitive substring filter; lines that don't
            match are silently dropped.
          - ``max_lines_per_sec``: integer rate cap. Lines exceeding the cap
            in any 1-second window are dropped; a single ``dropped`` event
            is emitted per second summarizing the count.
        """
        import json

        from modal_proto import api_pb2

        run = await _get_run_or_404(training_run_id)

        app_id = (run.modal_app_id or "").strip()
        if not app_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"TrainingRun {training_run_id!r} has no modal_app_id "
                    "yet — logs not available."
                ),
            )

        search_lower = search.strip().lower() if search else ""
        rate_cap = max(0, int(max_lines_per_sec or 0))

        async def event_stream():
            try:
                client = await get_modal_client()
            except Exception as exc:
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'error': f'auth failed: {exc!s}'})}\n\n"
                )
                return
            if client is None:
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'error': 'No Modal credentials configured. Run training-gym setup.'})}\n\n"
                )
                return

            last_entry_id = ""
            window_start = time.monotonic()
            window_emitted = 0
            window_dropped = 0
            consecutive_errors = 0
            max_consecutive_errors = 10

            def _drain_drop_event() -> str | None:
                nonlocal window_dropped
                if not window_dropped:
                    return None
                payload = {"dropped": window_dropped}
                window_dropped = 0
                return f"event: dropped\ndata: {json.dumps(payload)}\n\n"

            while True:
                if await request.is_disconnected():
                    return
                req = api_pb2.AppGetLogsRequest(
                    app_id=app_id,
                    timeout=55,
                    last_entry_id=last_entry_id,
                )
                try:
                    async for log_batch in client.stub.AppGetLogs.unary_stream(req):
                        if await request.is_disconnected():
                            return
                        consecutive_errors = 0
                        if log_batch.entry_id:
                            last_entry_id = log_batch.entry_id
                        for log in log_batch.items:
                            if not log.data:
                                continue
                            if search_lower and search_lower not in log.data.lower():
                                continue

                            now = time.monotonic()
                            if now - window_start >= 1.0:
                                drop_event = _drain_drop_event()
                                if drop_event:
                                    yield drop_event
                                window_start = now
                                window_emitted = 0

                            if rate_cap and window_emitted >= rate_cap:
                                window_dropped += 1
                                continue

                            window_emitted += 1
                            payload: dict[str, str | float] = {
                                "task_id": log_batch.task_id,
                                "line": log.data,
                            }
                            ts = getattr(log, "timestamp", 0) or 0
                            if ts:
                                payload["ts"] = ts
                            yield f"data: {json.dumps(payload)}\n\n"
                        if log_batch.app_done:
                            drop_event = _drain_drop_event()
                            if drop_event:
                                yield drop_event
                            yield "event: done\ndata: {}\n\n"
                            return
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        yield (
                            "event: error\n"
                            f"data: {json.dumps({'error': f'log stream failed after {consecutive_errors} retries: {exc!s}'})}\n\n"
                        )
                        return
                    backoff = min(2 ** (consecutive_errors - 1), 10)
                    yield (
                        "event: reconnect\n"
                        f"data: {json.dumps({'reason': str(exc)})}\n\n"
                    )
                    try:
                        await asyncio.sleep(backoff)
                    except asyncio.CancelledError:
                        return

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Historical Modal logs ──────────────

    @web.get("/api/runs/{training_run_id}/logs")
    async def get_run_logs(
        training_run_id: str,
        since: str = "",
        until: str = "",
        max_lines: int = 100,
        search: str = "",
    ):
        """Historical log fetch for a run, backed by Modal's ``AppFetchLogs``.

        Returns the most recent ``max_lines`` entries within the ``[since, until]``
        window, oldest-first.

        Query params:
          - ``since`` / ``until``: window bounds as epoch seconds, ISO 8601, or a
            relative age (``30m`` / ``2h`` / ``1d`` / ``45s`` = "N ago").
            ``since`` is exclusive and ``until`` is inclusive. Defaults to the run's
            lifetime.
          - ``max_lines``: max entries to return (default 100). Throws if negative or too large.
            These are the newest entries in the window (ClickHouse caps a single
            fetch at 20000).
          - ``search``: case-insensitive substring filter.

        Returns a JSON object with the following fields:
          - ``logs``: a list of log entries
          - ``has_more``: whether there are more log entries to fetch
          - ``next_until``: the timestamp of the next log entry to fetch
        """
        from modal_proto import api_pb2

        if max_lines < 0:
            raise HTTPException(status_code=400, detail="max_lines must be positive")

        run = await _get_run_or_404(training_run_id)

        app_id = (run.modal_app_id or "").strip()
        if not app_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"TrainingRun {training_run_id!r} has no modal_app_id "
                    "yet — logs not available."
                ),
            )

        now = time.time()
        since_ts = _parse_log_time(since, now)
        until_ts = _parse_log_time(until, now)
        for value, parsed, name in (
            (since, since_ts, "since"),
            (until, until_ts, "until"),
        ):
            if value and parsed is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{name} must be epoch seconds, ISO 8601, "
                        "or a relative time such as 24h"
                    ),
                )
        since_ts, until_ts = _resolve_log_window(
            since_ts,
            until_ts,
            default_since=run.started_at or run.created_at or 0,
            default_until=run.ended_at or run.completed_at or 0,
            now=now,
        )

        try:
            client = await get_modal_client()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Modal auth: {exc!s}")
        if client is None:
            raise HTTPException(
                status_code=503,
                detail="No Modal credentials configured. Run training-gym setup.",
            )

        req = api_pb2.AppFetchLogsRequest(
            app_id=app_id,
            since=_to_timestamp(since_ts),
            until=_to_timestamp(until_ts),
            limit=max_lines,
            search_text=search.strip(),
        )
        try:
            resp = await client.stub.AppFetchLogs(req)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"AppFetchLogs: {exc!s}")

        logs = _parse_log_batches(resp.batches)
        has_more, next_until = _compute_next_page(logs, max_lines)

        return JSONResponse(
            {
                "logs": logs,
                "has_more": has_more,
                "next_until": next_until,
            }
        )

    # ── Train results ────────────────────────────────────────────────────

    @web.get("/api/train-results")
    async def train_results():
        try:
            data = await get_cached_list("train_results", load_train_results)
        except Exception:
            data = []
        return JSONResponse(data)

    @web.get("/api/train-results/{training_run_id}")
    async def train_result(training_run_id: str):
        try:
            data = await run_in_threadpool(
                vol_get, MetadataStore.TRAIN_RESULTS, training_run_id
            )
            return JSONResponse(data)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"No training result for {training_run_id!r}",
            )

    # ── Eval results ─────────────────────────────────────────────────────

    @web.get("/api/evals")
    async def evals():
        try:
            data = await get_cached_list("evals", load_eval_summaries)
        except Exception:
            data = []
        return JSONResponse(data)

    @web.get("/api/evals/{eval_id}")
    async def eval_detail(eval_id: str):
        try:
            data = await run_in_threadpool(vol_get, MetadataStore.EVAL_RESULTS, eval_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"EvalResult {eval_id!r} not found",
            )
        if isinstance(data, dict):
            _apply_parsed(data.get("rows"))
        return JSONResponse(data)

    @web.get("/favicon.svg", include_in_schema=False)
    async def favicon():
        return FileResponse(f"{STATIC_DIR}/favicon.svg", media_type="image/svg+xml")

    @web.get("/apple-touch-icon.png", include_in_schema=False)
    async def apple_touch_icon():
        return FileResponse(
            f"{STATIC_DIR}/apple-touch-icon.png", media_type="image/png"
        )

    # ── SPA fallback ─────────────────────────────────────────────────────

    # Declared before the fallback so an unknown API path is a JSON 404 rather
    # than the SPA's HTML served with a 200: a frontend newer than the deployed
    # backend would otherwise parse index.html as JSON and report the parser's
    # error ("The string did not match the expected pattern", on WebKit).
    @web.get("/api/{full_path:path}", include_in_schema=False)
    async def api_not_found(full_path: str):
        raise HTTPException(status_code=404, detail=f"No such API path: {full_path}")

    @web.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # index.html names content-hashed bundles that the next deploy deletes,
        # so it must never be served from cache without revalidating: a browser
        # holding yesterday's HTML would request a bundle this image no longer
        # has and get a 404 with a blank page. The document is a fraction of a
        # kilobyte, so refetching it on every load costs nothing next to the
        # immutable bundles it points at.
        return FileResponse(
            f"{STATIC_DIR}/index.html",
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    return web
