"""HTTP queue / URL / token plumbing + run-context helpers for in-container
dashboard reporting.

Shared by the slime and miles frameworks (both expose it through their
``phase_reporting`` modules). Everything here is duck-typed and
dependency-light so it stays importable inside the training container without
slime/miles/torch present.
"""

from __future__ import annotations

import json
import os
import threading
import time
from queue import Full, Queue
import atexit
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from modal_dojo.common.sample_extraction import _coerce_float

# Legacy env names kept for backwards compatibility with older images.
PHASE_REPORT_URL_ENV = "SLIME_PHASE_REPORT_URL"
PHASE_REPORT_TOKEN_ENV = "SLIME_PHASE_REPORT_TOKEN"

# Internal queue entry: each item is {"_url": str, "_timeout": float, **payload}.
# Each dashboard endpoint has its own queue and worker (a lane), so posts stay
# ordered per endpoint and a slow endpoint can't hold up the others.
_REPORTER_DRAINING = False
_REPORTER_LOCK = threading.Lock()
_UNACKNOWLEDGED_TIMING_LOCK = threading.Lock()
_PRE_DRAIN_HOOKS: list[Callable[[], None]] = []
_UNACKNOWLEDGED_TIMING_FINALS: dict[tuple[str, str], dict[str, Any]] = {}
_MAX_UNACKNOWLEDGED_TIMING_FINALS = 128
_PHASE_PATH = "/api/framework-status"
_ROLLOUT_PATH = "/api/training-rollouts"
_ADVANTAGE_PATH = "/api/advantage-distributions"
_METRIC_POINTS_PATH = "/api/metric-points"
MAX_METRIC_POINTS_PER_BATCH = 5000
# Status and rollout writes rewrite the run record and take seconds. Waiting for
# them keeps a write from landing after the run's own final save.
_PHASE_TIMEOUT_SECONDS = 30.0
_STEP_EVENT_TIMEOUT_SECONDS = 5.0
_ROLLOUT_TIMEOUT_SECONDS = 30.0
REPORT_DRAIN_RETRY_DELAY_SECONDS = 0.1
REPORT_DRAIN_TIMEOUT_SECONDS = 60.0
_REPORT_DRAIN_DEADLINE: float | None = None


class _Lane:
    """One dashboard endpoint's queue and the worker thread that posts it."""

    def __init__(self) -> None:
        self.queue: Queue[dict[str, Any] | None] = Queue(maxsize=512)
        self.thread: threading.Thread | None = None
        self.sentinel_queued = False


_LANES: dict[str, _Lane] = {}


def _lane(url: str) -> _Lane:
    with _REPORTER_LOCK:
        return _LANES.setdefault(urlsplit(url).path.rstrip("/"), _Lane())


def register_pre_drain_hook(hook: Callable[[], None]) -> None:
    with _REPORTER_LOCK:
        if hook not in _PRE_DRAIN_HOOKS:
            _PRE_DRAIN_HOOKS.append(hook)


def _run_pre_drain_hooks() -> None:
    with _REPORTER_LOCK:
        hooks = tuple(_PRE_DRAIN_HOOKS)
    for hook in hooks:
        try:
            hook()
        except Exception:
            pass


def _arg_value(args: Any, key: str) -> Any:
    value = getattr(args, key, None)
    if value not in (None, ""):
        return value

    for container_name in ("extra_config", "custom_config"):
        container = getattr(args, container_name, None)
        if isinstance(container, dict):
            value = container.get(key)
            if value not in (None, ""):
                return value
    return None


def _run_context(args: Any) -> dict[str, Any]:
    return {
        "training_run_id": _arg_value(args, "training_run_id")
        or _arg_value(args, "modal_dojo_training_run_id")
        or os.environ.get("MODAL_DOJO_TRAINING_RUN_ID", "")
        or "",
        "app_name": _arg_value(args, "app_name")
        or _arg_value(args, "modal_dojo_app_name")
        or os.environ.get("MODAL_DOJO_APP_NAME", "")
        or "",
        "modal_app_id": os.environ.get("MODAL_APP_ID", ""),
    }


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _total_steps(args: Any) -> int | None:
    for key in ("num_rollout", "modal_dojo_total_steps"):
        total = _positive_int(_arg_value(args, key))
        if total is not None:
            return total
    return _positive_int(os.environ.get("MODAL_DOJO_TOTAL_STEPS"))


def _step_progress(args: Any, rollout_id: int | None = None) -> dict[str, Any]:
    total = _total_steps(args)
    if rollout_id is None:
        current = 0
    else:
        current = max(0, int(rollout_id) + 1)
    if total is not None:
        current = min(current, total)
    return {
        "progress_current": current,
        "progress_total": total,
        "progress_unit": "step",
    }


def _phase_url() -> str:
    return (
        os.environ.get("MODAL_DOJO_FRAMEWORK_STATUS_URL", "")
        or os.environ.get(PHASE_REPORT_URL_ENV, "")
    ).strip()


def _derive_url(path: str) -> str:
    base = _phase_url()
    if not base:
        return ""
    if base.endswith(_PHASE_PATH):
        return base[: -len(_PHASE_PATH)] + path
    return base.rstrip("/") + path


def _rollout_url() -> str:
    return _derive_url(_ROLLOUT_PATH)


def _advantage_url() -> str:
    return _derive_url(_ADVANTAGE_PATH)


def _report_token() -> str:
    return (
        os.environ.get("MODAL_DOJO_FRAMEWORK_STATUS_TOKEN", "")
        or os.environ.get(PHASE_REPORT_TOKEN_ENV, "")
    ).strip()


def _ensure_worker(lane: _Lane, *, allow_during_drain: bool = False) -> None:
    if lane.thread is not None or (_REPORTER_DRAINING and not allow_during_drain):
        return
    with _REPORTER_LOCK:
        if lane.thread is not None or (_REPORTER_DRAINING and not allow_during_drain):
            return
        lane.thread = threading.Thread(
            target=_worker,
            args=(lane.queue,),
            name="modal-dojo-phase-reporter",
            daemon=True,
        )
        lane.thread.start()


def _coalesce_status(
    queue: "Queue[dict[str, Any] | None]", item: dict[str, Any]
) -> bool:
    """Fold a plain status update into the run's pending one.

    The dashboard merges status updates into one snapshot, so only the latest
    pending one is worth a seconds-long write. Step events are never folded.
    """
    with queue.mutex:
        last = queue.queue[-1] if queue.queue else None
        if (
            last is None
            or "step_event" in last
            or "step_event" in item
            or last.get("training_run_id") != item.get("training_run_id")
        ):
            return False
        last.update(item)
        return True


def _enqueue(
    payload: dict[str, Any],
    *,
    timeout_seconds: float = _PHASE_TIMEOUT_SECONDS,
) -> None:
    """Enqueue a framework-status payload with its request timeout."""
    if _REPORTER_DRAINING:
        return
    url = _phase_url()
    if not url:
        return
    lane = _lane(url)
    _ensure_worker(lane)
    item = {"_url": url, "_timeout": timeout_seconds, **payload}
    if _coalesce_status(lane.queue, item):
        return
    try:
        lane.queue.put_nowait(item)
    except Exception:
        pass


def _enqueue_rollout(payload: dict[str, Any]) -> None:
    """Enqueue a rollout-data payload (large, longer timeout)."""
    if _REPORTER_DRAINING:
        return
    url = _rollout_url()
    if not url:
        return
    lane = _lane(url)
    _ensure_worker(lane)
    item = {"_url": url, "_timeout": _ROLLOUT_TIMEOUT_SECONDS, **payload}
    try:
        lane.queue.put_nowait(item)
    except Exception:
        pass


def _enqueue_advantage(payload: dict[str, Any]) -> None:
    """Enqueue an advantage-distribution payload (longer timeout like rollouts)."""
    if _REPORTER_DRAINING:
        return
    url = _advantage_url()
    if not url:
        return
    lane = _lane(url)
    _ensure_worker(lane)
    item = {"_url": url, "_timeout": _ROLLOUT_TIMEOUT_SECONDS, **payload}
    try:
        lane.queue.put_nowait(item)
    except Exception:
        pass


def _enqueue_timing(payload: dict[str, Any], *, final: bool = False) -> None:
    """Enqueue a timing snapshot on the shared reporting worker."""
    if _REPORTER_DRAINING and not final:
        return
    url = _derive_url("/api/timing-events")
    if not url:
        return
    lane = _lane(url)
    _ensure_worker(lane, allow_during_drain=_REPORTER_DRAINING and final)
    item = {
        "_url": url,
        "_timeout": _ROLLOUT_TIMEOUT_SECONDS,
        "_retry_count": 3 if final else 0,
        "_retry_delay": 1.0,
        "_timing_debug": os.environ.get("MODAL_DOJO_TIMING_DEBUG") == "1",
        **payload,
    }
    key = (str(payload.get("training_run_id", "")), str(payload.get("storage_key", "")))
    if final:
        with _UNACKNOWLEDGED_TIMING_LOCK:
            if len(_UNACKNOWLEDGED_TIMING_FINALS) >= _MAX_UNACKNOWLEDGED_TIMING_FINALS:
                _UNACKNOWLEDGED_TIMING_FINALS.pop(
                    next(iter(_UNACKNOWLEDGED_TIMING_FINALS))
                )
            _UNACKNOWLEDGED_TIMING_FINALS[key] = item
    try:
        lane.queue.put_nowait(item)
    except Full:
        if final:
            print(
                f"[modal-dojo] timing final queue full; retaining {key} for "
                "process-exit retry",
                flush=True,
            )
    except Exception:
        if final:
            print(
                f"[modal-dojo] failed to enqueue timing final {key}; retaining "
                "it for process-exit retry",
                flush=True,
            )


def _enqueue_metric_points(payload: dict[str, Any], *, final: bool = False) -> None:
    """Enqueue mirrored scalars; ``final`` is the process-exit flush."""
    if _REPORTER_DRAINING and not final:
        return
    url = _derive_url(_METRIC_POINTS_PATH)
    if not url:
        return
    lane = _lane(url)
    _ensure_worker(lane, allow_during_drain=final)
    item = {
        "_url": url,
        "_timeout": _STEP_EVENT_TIMEOUT_SECONDS,
        "_retry_count": 3 if final else 1,
        "_retry_delay": 1.0,
        **payload,
        "final": final,
    }
    try:  # the exit flush waits for room; the worker is still draining
        lane.queue.put(item, block=final, timeout=_STEP_EVENT_TIMEOUT_SECONDS)
    except Exception:
        pass


def _requeue_timing_retry(payload: dict[str, Any], retries: int) -> None:
    if _REPORTER_DRAINING:
        return
    payload["_retry_count"] = retries
    try:
        _lane(payload["_url"]).queue.put_nowait(payload)
    except Full:
        if payload.get("final", False):
            print(
                "[modal-dojo] timing retry queue full; retaining final for "
                "process-exit retry",
                flush=True,
            )
    except Exception:
        pass


def _schedule_timing_retry(payload: dict[str, Any], retries: int) -> None:
    delay = float(payload.get("_retry_delay", 1.0) or 1.0)
    payload["_retry_delay"] = min(delay * 2.0, 4.0)
    timer = threading.Timer(delay, _requeue_timing_retry, (payload, retries))
    timer.daemon = True
    timer.start()


def _is_final_timing(payload: dict[str, Any]) -> bool:
    """Process-exit flush whose delivery is worth retrying while draining."""
    url = str(payload.get("_url", "")).rstrip("/")
    return bool(payload.get("final")) and (
        url.endswith(("/api/timing-events", _METRIC_POINTS_PATH))
    )


def _retry_timing_final_during_drain(payload: dict[str, Any], retries: int) -> bool:
    deadline = _REPORT_DRAIN_DEADLINE
    while retries > 0 and deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        delay = min(
            float(payload.get("_retry_delay", REPORT_DRAIN_RETRY_DELAY_SECONDS) or 0),
            REPORT_DRAIN_RETRY_DELAY_SECONDS,
            remaining,
        )
        if delay:
            time.sleep(delay)
        if time.monotonic() >= deadline:
            return False
        retries -= 1
        payload["_retry_count"] = retries
        if _post(payload):
            return True
    return False


def _worker(queue: "Queue[dict[str, Any] | None]") -> None:
    while True:
        try:
            payload = queue.get()
        except Exception:
            continue
        if payload is None:
            try:
                deadline = _REPORT_DRAIN_DEADLINE
                remaining = deadline is None or time.monotonic() < deadline
                with queue.mutex:
                    has_remaining = any(item is not None for item in queue.queue)
                    if has_remaining and remaining:
                        queue.queue.append(None)
                        queue.unfinished_tasks += 1
                        queue.not_empty.notify()
                        continue
                    discarded_sentinels = len(queue.queue)
                    if discarded_sentinels:
                        queue.queue.clear()
                        queue.unfinished_tasks -= discarded_sentinels
                        if queue.unfinished_tasks == 0:
                            queue.all_tasks_done.notify_all()
                return
            finally:
                queue.task_done()
        try:
            try:
                delivered = _post(payload)
                retries = int(payload.get("_retry_count", 0) or 0)
                if delivered and payload.get("final", False):
                    key = (
                        str(payload.get("training_run_id", "")),
                        str(payload.get("storage_key", "")),
                    )
                    with _UNACKNOWLEDGED_TIMING_LOCK:
                        _UNACKNOWLEDGED_TIMING_FINALS.pop(key, None)
                elif not delivered and retries > 0:
                    if _REPORTER_DRAINING and _is_final_timing(payload):
                        delivered = _retry_timing_final_during_drain(payload, retries)
                        if delivered:
                            key = (
                                str(payload.get("training_run_id", "")),
                                str(payload.get("storage_key", "")),
                            )
                            with _UNACKNOWLEDGED_TIMING_LOCK:
                                _UNACKNOWLEDGED_TIMING_FINALS.pop(key, None)
                    elif not _REPORTER_DRAINING:
                        _schedule_timing_retry(payload, retries - 1)
            except Exception as exc:
                if payload.get("_timing_debug"):
                    debug_payload = {
                        "event": "post_attempt",
                        "lane": payload.get("storage_key"),
                        "final": payload.get("final"),
                        "phases": sorted(payload.get("phases", {})),
                        "result": "failed",
                        "retry_count": payload.get("_retry_count"),
                        "failure_reason": {
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                    print(
                        "[timing-debug] " + json.dumps(debug_payload, sort_keys=True),
                        flush=True,
                    )
        finally:
            queue.task_done()


def _post(item: dict[str, Any]) -> bool:
    url = item.get("_url", "")
    timeout = float(
        item.get("_timeout", _PHASE_TIMEOUT_SECONDS) or _PHASE_TIMEOUT_SECONDS
    )
    if _REPORTER_DRAINING and _REPORT_DRAIN_DEADLINE is not None:
        timeout = min(timeout, max(0.001, _REPORT_DRAIN_DEADLINE - time.monotonic()))
    retry_count = item.get("_retry_count")
    timing_debug = item.get("_timing_debug", False)
    if not url:
        return False

    body = json.dumps(
        {key: value for key, value in item.items() if not key.startswith("_")},
        default=str,
    ).encode("utf-8")

    from modal_dojo.common.config import modal_proxy_auth_headers

    headers = {
        "Content-Type": "application/json",
        **modal_proxy_auth_headers(),
    }

    token = _report_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    result = "failed"
    failure_reason: dict[str, Any] | None = None
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read()
        result = "ok"
    except (OSError, URLError) as exc:
        failure_reason = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        if isinstance(exc, HTTPError):
            failure_reason["http_status"] = exc.code
    if timing_debug:
        debug_payload = {
            "event": "post_attempt",
            "lane": item.get("storage_key"),
            "final": item.get("final"),
            "phases": sorted(item.get("phases", {})),
            "result": result,
        }
        if result == "failed":
            debug_payload["retry_count"] = retry_count
            debug_payload["failure_reason"] = failure_reason
        print(
            "[timing-debug] " + json.dumps(debug_payload, sort_keys=True),
            flush=True,
        )
    return result == "ok"


def _retry_unacknowledged_timing_finals() -> None:
    with _UNACKNOWLEDGED_TIMING_LOCK:
        pending = tuple(_UNACKNOWLEDGED_TIMING_FINALS.values())
    for item in pending:
        key = (str(item.get("training_run_id", "")), str(item.get("storage_key", "")))
        queue = _lane(item["_url"]).queue
        with queue.mutex:
            already_queued = any(
                queued is not None
                and queued.get("final", False)
                and (
                    str(queued.get("training_run_id", "")),
                    str(queued.get("storage_key", "")),
                )
                == key
                for queued in queue.queue
            )
        if already_queued:
            continue
        try:
            queue.put_nowait(item)
        except Full:
            print(
                f"[modal-dojo] timing final retry queue full for {key}",
                flush=True,
            )


register_pre_drain_hook(_retry_unacknowledged_timing_finals)


def _compact_report_queue(queue: "Queue[dict[str, Any] | None]") -> None:
    with queue.mutex:
        queued = list(queue.queue)
        final_priority: list[dict[str, Any] | None] = []
        metrics_by_run: dict[str, dict[str, Any]] = {}
        status_priority: list[dict[str, Any] | None] = []
        remaining: list[dict[str, Any] | None] = []
        discarded = 0
        for item in queued:
            if item is None:
                remaining.append(item)
                continue
            url = str(item.get("_url", "")).rstrip("/")
            if url.endswith(_METRIC_POINTS_PATH):
                # Fold the run's queued batches into one, in arrival order, so
                # the drain deadline can't strand the final values behind a backlog.
                merged = metrics_by_run.setdefault(
                    str(item.get("training_run_id")), {**item, "points": {}}
                )
                for point in item.get("points", []):
                    folded = merged["points"].setdefault(point["step"], {"metrics": {}})
                    folded.update(
                        point, metrics={**folded["metrics"], **point["metrics"]}
                    )
                merged["final"] = merged.get("final", False) or item.get("final", False)
                merged["_retry_count"] = max(
                    merged.get("_retry_count", 0), item.get("_retry_count", 0)
                )
                discarded += 1
            elif _is_final_timing(item):
                final_priority.append(item)
            elif url.endswith("/api/timing-events"):
                discarded += 1
            elif url.endswith("/api/framework-status"):
                status_priority.append(item)
            else:
                remaining.append(item)
        for merged in metrics_by_run.values():
            points = [point for _, point in sorted(merged["points"].items())]
            chunks = [
                points[i : i + MAX_METRIC_POINTS_PER_BATCH]
                for i in range(0, len(points), MAX_METRIC_POINTS_PER_BATCH)
            ] or [[]]
            for chunk in chunks:
                final_priority.append({**merged, "points": chunk})
                discarded -= 1
        queue.queue.clear()
        queue.queue.extend(final_priority)
        queue.queue.extend(status_priority)
        queue.queue.extend(remaining)
        queue.unfinished_tasks -= discarded
        queue.not_empty.notify_all()


def drain() -> None:
    """Deliver queued reports, then stop accepting new ones.

    Waits up to ``REPORT_DRAIN_TIMEOUT_SECONDS`` across all endpoints. Runs at
    process exit; call it earlier to land every report before a final write.
    """
    global _REPORTER_DRAINING, _REPORT_DRAIN_DEADLINE
    _REPORT_DRAIN_DEADLINE = time.monotonic() + REPORT_DRAIN_TIMEOUT_SECONDS
    with _REPORTER_LOCK:
        _REPORTER_DRAINING = True
    for lane in _lanes():
        _compact_report_queue(lane.queue)
    _run_pre_drain_hooks()
    lanes = _lanes()
    for lane in lanes:
        _compact_report_queue(lane.queue)
    deadline = _REPORT_DRAIN_DEADLINE
    for lane in lanes:
        if lane.thread is None or lane.sentinel_queued:
            continue
        try:
            lane.queue.put(None, timeout=max(0.0, deadline - time.monotonic()))
        except Full:
            continue
        lane.sentinel_queued = True
    for lane in lanes:
        if lane.thread is not None:
            lane.thread.join(timeout=max(0.0, deadline - time.monotonic()))


def _lanes() -> list[_Lane]:
    with _REPORTER_LOCK:
        return list(_LANES.values())


atexit.register(drain)


def _advantage_samples_payload(
    sample_sums: list[float],
    sample_counts: list[float],
    sample_indices: list[int | None],
    raw_rewards: list[object],
    n_samples_per_prompt: int,
) -> list[dict[str, float | int | None]]:
    """Build the per-sample advantage rows from masked ``(sum, count)`` pairs.

    Pure (no torch / mpu) so the group-index and divide-by-count logic is
    unit-testable. ``advantage = sum / count`` is the mask-weighted mean over
    the sample's response tokens; ``group_index`` is the GRPO prompt group the
    sample belongs to (``sample_index // n_samples_per_prompt``).
    """
    n_per = max(1, int(n_samples_per_prompt or 1))
    out: list[dict[str, float | int | None]] = []
    for i in range(len(sample_sums)):
        count = sample_counts[i] if i < len(sample_counts) else 0.0
        advantage = (sample_sums[i] / count) if count else 0.0
        if i < len(sample_indices) and sample_indices[i] is not None:
            idx = int(sample_indices[i])
        else:
            idx = i
        raw = raw_rewards[i] if i < len(raw_rewards) else None
        out.append(
            {
                "sample_index": idx,
                "group_index": idx // n_per,
                "advantage": float(advantage),
                "raw_reward": _coerce_float(raw) if raw is not None else None,
            }
        )
    return out
