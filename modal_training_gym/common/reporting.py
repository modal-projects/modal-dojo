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
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from modal_training_gym.common.sample_extraction import _coerce_float

# Legacy env names kept for backwards compatibility with older images.
PHASE_REPORT_URL_ENV = "SLIME_PHASE_REPORT_URL"
PHASE_REPORT_TOKEN_ENV = "SLIME_PHASE_REPORT_TOKEN"

# Internal queue entry: each item is {"_url": str, "_timeout": float, **payload}.
# Status reports use the framework-status URL with a short 1s timeout;
# rollout-data reports derive a /api/training-rollouts URL from the same base
# with a longer timeout because payloads can be 100KB+.
_REPORT_QUEUE: "Queue[dict[str, Any] | None]" = Queue(maxsize=512)
_REPORTER_STARTED = False
_REPORTER_THREAD: threading.Thread | None = None
_REPORTER_DRAINING = False
_REPORT_DRAIN_SENTINEL_QUEUED = False
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
_PHASE_TIMEOUT_SECONDS = 1.0
_STEP_EVENT_TIMEOUT_SECONDS = 5.0
_ROLLOUT_TIMEOUT_SECONDS = 10.0
REPORT_DRAIN_POST_TIMEOUT_SECONDS = 1.0
REPORT_DRAIN_FINAL_POST_TIMEOUT_SECONDS = _ROLLOUT_TIMEOUT_SECONDS
REPORT_DRAIN_FINAL_POST_COUNT = 4
REPORT_DRAIN_RETRY_DELAY_SECONDS = 0.1
REPORT_DRAIN_TIMEOUT_SECONDS = (
    REPORT_DRAIN_FINAL_POST_TIMEOUT_SECONDS * REPORT_DRAIN_FINAL_POST_COUNT
)
_REPORT_DRAIN_DEADLINE: float | None = None


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
        or _arg_value(args, "training_gym_training_run_id")
        or os.environ.get("TRAINING_GYM_TRAINING_RUN_ID", "")
        or "",
        "app_name": _arg_value(args, "app_name")
        or _arg_value(args, "training_gym_app_name")
        or os.environ.get("TRAINING_GYM_APP_NAME", "")
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
    for key in ("num_rollout", "training_gym_total_steps"):
        total = _positive_int(_arg_value(args, key))
        if total is not None:
            return total
    return _positive_int(os.environ.get("TRAINING_GYM_TOTAL_STEPS"))


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
        os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_URL", "")
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
        os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_TOKEN", "")
        or os.environ.get(PHASE_REPORT_TOKEN_ENV, "")
    ).strip()


def _ensure_worker(*, allow_during_drain: bool = False) -> None:
    global _REPORTER_STARTED, _REPORTER_THREAD
    if _REPORTER_STARTED or (_REPORTER_DRAINING and not allow_during_drain):
        return
    with _REPORTER_LOCK:
        if _REPORTER_STARTED or (_REPORTER_DRAINING and not allow_during_drain):
            return
        thread = threading.Thread(
            target=_worker,
            name="training-gym-phase-reporter",
            daemon=True,
        )
        thread.start()
        _REPORTER_THREAD = thread
        _REPORTER_STARTED = True


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
    _ensure_worker()
    item = {"_url": url, "_timeout": timeout_seconds, **payload}
    try:
        _REPORT_QUEUE.put_nowait(item)
    except Exception:
        pass


def _enqueue_rollout(payload: dict[str, Any]) -> None:
    """Enqueue a rollout-data payload (large, longer timeout)."""
    if _REPORTER_DRAINING:
        return
    url = _rollout_url()
    if not url:
        return
    _ensure_worker()
    item = {"_url": url, "_timeout": _ROLLOUT_TIMEOUT_SECONDS, **payload}
    try:
        _REPORT_QUEUE.put_nowait(item)
    except Exception:
        pass


def _enqueue_advantage(payload: dict[str, Any]) -> None:
    """Enqueue an advantage-distribution payload (longer timeout like rollouts)."""
    if _REPORTER_DRAINING:
        return
    url = _advantage_url()
    if not url:
        return
    _ensure_worker()
    item = {"_url": url, "_timeout": _ROLLOUT_TIMEOUT_SECONDS, **payload}
    try:
        _REPORT_QUEUE.put_nowait(item)
    except Exception:
        pass


def _enqueue_timing(payload: dict[str, Any], *, final: bool = False) -> None:
    """Enqueue a timing snapshot on the shared reporting worker."""
    if _REPORTER_DRAINING and not final:
        return
    url = _derive_url("/api/timing-events")
    if not url:
        return
    if _REPORTER_DRAINING and final:
        _ensure_worker(allow_during_drain=True)
    else:
        _ensure_worker()
    item = {
        "_url": url,
        "_timeout": _ROLLOUT_TIMEOUT_SECONDS,
        "_retry_count": 3 if final else 0,
        "_retry_delay": 1.0,
        "_timing_debug": os.environ.get("TRAINING_GYM_TIMING_DEBUG") == "1",
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
        _REPORT_QUEUE.put_nowait(item)
    except Full:
        if final:
            print(
                f"[training-gym] timing final queue full; retaining {key} for "
                "process-exit retry",
                flush=True,
            )
    except Exception:
        if final:
            print(
                f"[training-gym] failed to enqueue timing final {key}; retaining "
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
    _ensure_worker(allow_during_drain=final)
    item = {
        "_url": url,
        "_timeout": _STEP_EVENT_TIMEOUT_SECONDS,
        "_retry_count": 3 if final else 1,
        "_retry_delay": 1.0,
        **payload,
        "final": final,
    }
    try:  # the exit flush waits for room; the worker is still draining
        _REPORT_QUEUE.put(item, block=final, timeout=_STEP_EVENT_TIMEOUT_SECONDS)
    except Exception:
        pass


def _requeue_timing_retry(payload: dict[str, Any], retries: int) -> None:
    if _REPORTER_DRAINING:
        return
    payload["_retry_count"] = retries
    try:
        _REPORT_QUEUE.put_nowait(payload)
    except Full:
        if payload.get("final", False):
            print(
                "[training-gym] timing retry queue full; retaining final for "
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
        url.endswith("/api/timing-events") or url.endswith(_METRIC_POINTS_PATH)
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


def _worker() -> None:
    while True:
        try:
            payload = _REPORT_QUEUE.get()
        except Exception:
            continue
        if payload is None:
            try:
                deadline = _REPORT_DRAIN_DEADLINE
                remaining = deadline is None or time.monotonic() < deadline
                with _REPORT_QUEUE.mutex:
                    has_remaining = any(
                        item is not None for item in _REPORT_QUEUE.queue
                    )
                    if has_remaining and remaining:
                        _REPORT_QUEUE.queue.append(None)
                        _REPORT_QUEUE.unfinished_tasks += 1
                        _REPORT_QUEUE.not_empty.notify()
                        continue
                    discarded_sentinels = len(_REPORT_QUEUE.queue)
                    if discarded_sentinels:
                        _REPORT_QUEUE.queue.clear()
                        _REPORT_QUEUE.unfinished_tasks -= discarded_sentinels
                        if _REPORT_QUEUE.unfinished_tasks == 0:
                            _REPORT_QUEUE.all_tasks_done.notify_all()
                return
            finally:
                _REPORT_QUEUE.task_done()
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
            _REPORT_QUEUE.task_done()


def _post(item: dict[str, Any]) -> bool:
    url = item.get("_url", "")
    timeout = float(
        item.get("_timeout", _PHASE_TIMEOUT_SECONDS) or _PHASE_TIMEOUT_SECONDS
    )
    if _REPORTER_DRAINING:
        if _is_final_timing(item):
            remaining = (
                _REPORT_DRAIN_DEADLINE - time.monotonic()
                if _REPORT_DRAIN_DEADLINE is not None
                else REPORT_DRAIN_FINAL_POST_TIMEOUT_SECONDS
            )
            timeout = min(timeout, max(0.001, remaining))
        else:
            timeout = min(timeout, REPORT_DRAIN_POST_TIMEOUT_SECONDS)
    retry_count = item.get("_retry_count")
    timing_debug = item.get("_timing_debug", False)
    if not url:
        return False

    body = json.dumps(
        {key: value for key, value in item.items() if not key.startswith("_")},
        default=str,
    ).encode("utf-8")

    from modal_training_gym.common.config import modal_proxy_auth_headers

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
        with _REPORT_QUEUE.mutex:
            already_queued = any(
                queued is not None
                and queued.get("final", False)
                and (
                    str(queued.get("training_run_id", "")),
                    str(queued.get("storage_key", "")),
                )
                == key
                for queued in _REPORT_QUEUE.queue
            )
        if already_queued:
            continue
        try:
            _REPORT_QUEUE.put_nowait(item)
        except Full:
            print(
                f"[training-gym] timing final retry queue full for {key}",
                flush=True,
            )


register_pre_drain_hook(_retry_unacknowledged_timing_finals)


def _compact_report_queue() -> None:
    with _REPORT_QUEUE.mutex:
        queued = list(_REPORT_QUEUE.queue)
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
        _REPORT_QUEUE.queue.clear()
        _REPORT_QUEUE.queue.extend(final_priority)
        _REPORT_QUEUE.queue.extend(status_priority)
        _REPORT_QUEUE.queue.extend(remaining)
        _REPORT_QUEUE.unfinished_tasks -= discarded
        _REPORT_QUEUE.not_empty.notify_all()


def _drain_report_queue() -> None:
    global _REPORTER_DRAINING, _REPORT_DRAIN_DEADLINE, _REPORT_DRAIN_SENTINEL_QUEUED
    _REPORT_DRAIN_DEADLINE = time.monotonic() + REPORT_DRAIN_TIMEOUT_SECONDS
    with _REPORTER_LOCK:
        _REPORTER_DRAINING = True
    _compact_report_queue()
    _run_pre_drain_hooks()
    _compact_report_queue()
    with _REPORTER_LOCK:
        thread = _REPORTER_THREAD
        started = _REPORTER_STARTED
        sentinel_queued = _REPORT_DRAIN_SENTINEL_QUEUED
    if thread is None or not started:
        return
    deadline = _REPORT_DRAIN_DEADLINE
    assert deadline is not None
    if not sentinel_queued:
        try:
            _REPORT_QUEUE.put(None, timeout=max(0.0, deadline - time.monotonic()))
        except Full:
            return
        with _REPORTER_LOCK:
            _REPORT_DRAIN_SENTINEL_QUEUED = True
    thread.join(timeout=max(0.0, deadline - time.monotonic()))


atexit.register(_drain_report_queue)


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
