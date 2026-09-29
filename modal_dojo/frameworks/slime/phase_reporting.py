"""Stable import surface for slime's in-container dashboard reporting.

The build-time patches (``modal_helpers/patches/patch_rollout_status_reporting.py``
and ``patch_advantage_distribution.py``) and the recipe's default custom-function
paths import from this module *inside the training container*, so everything
they reference must stay importable here. The implementation is split across:

- :mod:`modal_dojo.common.reporting` — HTTP queue/URL/token plumbing +
  run-context helpers (shared with miles)
- :mod:`modal_dojo.common.sample_extraction` — trace/image/trajectory
  extraction from Samples (shared with miles)
- :mod:`.advantage_reporting` — torch/megatron advantage-distribution math

This module keeps the reporting entry points (``report_*``, ``log_*``,
``before_*``) and re-exports the split internals for compatibility.
"""

from __future__ import annotations

import time
from typing import Any

from modal_dojo.common.status import SlimeStatus
from modal_dojo.common.timing_recorder import (
    RoleRecorder as RoleRecorder,
    recording_lane as recording_lane,
    recording_lane_on_reporting_rank as recording_lane_on_reporting_rank,
    time_phase as time_phase,
)

from .advantage_reporting import (
    _advantage_samples_payload as _advantage_samples_payload,
    report_advantage_distribution as report_advantage_distribution,
)
from modal_dojo.common.reporting import (
    _STEP_EVENT_TIMEOUT_SECONDS,
    _enqueue,
    _enqueue_rollout,
    _run_context,
    _step_progress,
)
from modal_dojo.common.reporting import (
    _advantage_url as _advantage_url,
    _arg_value as _arg_value,
    _derive_url as _derive_url,
    _enqueue_advantage as _enqueue_advantage,
    _phase_url as _phase_url,
    _positive_int as _positive_int,
    _report_token as _report_token,
    _rollout_url as _rollout_url,
    _total_steps as _total_steps,
)
from modal_dojo.common.sample_extraction import (
    RolloutImageStore,
    _image_limit,
    _metrics_to_dict,
    _resolve_hook,
    _response_parser,
    _sample_to_dict,
    _trace_enabled,
    _trace_sample_limit,
    _trajectory_sample_limit,
)
from modal_dojo.common.sample_extraction import (
    CAPTURE_TRACE_ENV as CAPTURE_TRACE_ENV,
    IMAGE_SAMPLE_LIMIT_ENV as IMAGE_SAMPLE_LIMIT_ENV,
    RESPONSE_PARSER_PATH_ENV as RESPONSE_PARSER_PATH_ENV,
    TRACE_SAMPLE_LIMIT_ENV as TRACE_SAMPLE_LIMIT_ENV,
    TRAJECTORY_SAMPLE_LIMIT_ENV as TRAJECTORY_SAMPLE_LIMIT_ENV,
    _IMAGE_MAX_BYTES as _IMAGE_MAX_BYTES,
    _IMAGE_MAX_DIM as _IMAGE_MAX_DIM,
    _IMAGE_LIMIT_DEFAULT as _IMAGE_LIMIT_DEFAULT,
    _IMAGE_REF_CHARS as _IMAGE_REF_CHARS,
    _TRACE_ATTR_STR_MAX as _TRACE_ATTR_STR_MAX,
    _TRACE_MAX_SPANS as _TRACE_MAX_SPANS,
    _TRACE_SAMPLE_LIMIT_DEFAULT as _TRACE_SAMPLE_LIMIT_DEFAULT,
    _TRAJECTORY_MAX_MESSAGES as _TRAJECTORY_MAX_MESSAGES,
    _TRAJECTORY_MSG_CHARS_MAX as _TRAJECTORY_MSG_CHARS_MAX,
    _TRAJECTORY_SAMPLE_LIMIT_DEFAULT as _TRAJECTORY_SAMPLE_LIMIT_DEFAULT,
    _coerce_float as _coerce_float,
    _sample_score as _sample_score,
    _coerce_text as _coerce_text,
    _compact_trajectory_messages as _compact_trajectory_messages,
    _extract_audio_from_prompt as _extract_audio_from_prompt,
    _extract_trace as _extract_trace,
    _image_candidates as _image_candidates,
    _image_to_data_uri as _image_to_data_uri,
    _normalize_span as _normalize_span,
    _normalize_trace as _normalize_trace,
    _parsed_response_dict as _parsed_response_dict,
    _trace_attrs as _trace_attrs,
    _trace_scalar as _trace_scalar,
)

CUSTOM_ROLLOUT_LOG_FUNCTION_PATH_KEY = "training_gym_custom_rollout_log_function_path"
CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH_KEY = (
    "training_gym_custom_eval_rollout_log_function_path"
)
CUSTOM_BEFORE_LOG_PROB_HOOK_PATH_KEY = (
    "training_gym_custom_megatron_before_log_prob_hook_path"
)
CUSTOM_BEFORE_TRAIN_STEP_HOOK_PATH_KEY = (
    "training_gym_custom_megatron_before_train_step_hook_path"
)


def _hook_path_from_args(args: Any, path_key: str) -> str | None:
    direct = getattr(args, path_key, None)
    if isinstance(direct, str) and direct.strip():
        return direct

    for container_name in ("extra_config", "custom_config"):
        container = getattr(args, container_name, None)
        if isinstance(container, dict):
            value = container.get(path_key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def report_phase(
    status: SlimeStatus,
    args: Any = None,
    **extra: Any,
) -> None:
    _enqueue(
        {**_run_context(args), "phase": status.value, "event_ts": time.time(), **extra}
    )


def report_rollout_samples(
    rollout_id: int,
    args: Any,
    samples: Any,
    rollout_extra_metrics: Any,
    rollout_time: Any,
) -> None:
    """Post one TrainingRolloutResult-shaped payload to the dashboard."""
    if samples is None:
        return
    parser = _response_parser()
    # Trace/trajectory only the first N samples (traces also gated by an enable flag)
    # so the payload stays small — the caps keep volume growth well under 1%. Images
    # are capped by distinct content instead.
    trace_limit = _trace_sample_limit() if _trace_enabled() else 0
    trajectory_limit = _trajectory_sample_limit()
    image_store = RolloutImageStore(_image_limit())
    n_per = _positive_int(_arg_value(args, "n_samples_per_prompt")) or 1
    try:
        sample_dicts = [
            _sample_to_dict(
                s,
                parser,
                include_trace=(i < trace_limit),
                image_store=image_store,
                include_trajectory=(i < trajectory_limit),
                n_samples_per_prompt=n_per,
            )
            for i, s in enumerate(samples)
        ]
    except TypeError:
        return
    payload = {
        **_run_context(args),
        "rollout_id": int(rollout_id),
        "created_at": int(time.time()),
        "n_samples_per_prompt": n_per,
        "samples": sample_dicts,
        "metrics": _metrics_to_dict(rollout_extra_metrics),
    }
    if rollout_time is not None:
        try:
            payload["rollout_time"] = float(rollout_time)
        except (TypeError, ValueError):
            pass
    _enqueue_rollout(payload)


def _call_hook(path_key: str, args: Any, *hook_args: Any, **hook_kwargs: Any) -> Any:
    hook = _resolve_hook(_hook_path_from_args(args, path_key))
    if hook is None:
        return None
    return hook(*hook_args, **hook_kwargs)


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: Any,
    rollout_extra_metrics: Any,
    rollout_time: Any,
) -> bool:
    progress = _step_progress(args, rollout_id)
    report_phase(
        SlimeStatus.ROLLOUT_LOGGING,
        args,
        **progress,
        rollout_id=rollout_id,
        sample_count=len(samples) if hasattr(samples, "__len__") else None,
        metrics=rollout_extra_metrics,
        rollout_time=rollout_time,
    )
    report_rollout_samples(
        rollout_id, args, samples, rollout_extra_metrics, rollout_time
    )
    result = _call_hook(
        CUSTOM_ROLLOUT_LOG_FUNCTION_PATH_KEY,
        args,
        rollout_id,
        args,
        samples,
        rollout_extra_metrics,
        rollout_time,
    )
    if result is None:
        return False
    return bool(result)


def log_eval_rollout_data(
    rollout_id: int,
    args: Any,
    data: Any,
    extra_metrics: Any,
) -> bool:
    report_phase(
        SlimeStatus.EVAL_ROLLOUT_LOGGING,
        args,
        **_step_progress(args, rollout_id),
        rollout_id=rollout_id,
        sample_count=len(data) if hasattr(data, "__len__") else None,
        metrics=extra_metrics,
    )
    result = _call_hook(
        CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH_KEY,
        args,
        rollout_id,
        args,
        data,
        extra_metrics,
    )
    if result is None:
        return False
    return bool(result)


def before_log_prob_hook(args: Any, model: Any, store_prefix: str) -> None:
    # NOTE: this hook runs in the Megatron train actor, which has no current
    # rollout_id, so the compute_log_probs substep is reported (id-tagged) from
    # the driver loop right before actor_model.async_train(); see the patch.
    _call_hook(
        CUSTOM_BEFORE_LOG_PROB_HOOK_PATH_KEY,
        args,
        args,
        model,
        store_prefix,
    )


def before_train_step_hook(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    report_phase(
        SlimeStatus.OPTIMIZER_STEP,
        args,
        **_step_progress(args, rollout_id),
        rollout_id=rollout_id,
        step_id=step_id,
    )
    _call_hook(
        CUSTOM_BEFORE_TRAIN_STEP_HOOK_PATH_KEY,
        args,
        args,
        rollout_id,
        step_id,
        model,
        optimizer,
        opt_param_scheduler,
    )


def report_step_event(
    status: SlimeStatus | str,
    args: Any = None,
    rollout_id: int | None = None,
    step_event: str = "",
) -> None:
    """Report one step/substep event tagged with the ``status`` phase.

    ``status`` may be a plain string — the patched slime train.py passes phase
    names as literals so the injected code stays stdlib-only.
    """
    payload = {
        **_run_context(args),
        "phase": status.value if isinstance(status, SlimeStatus) else str(status),
        **_step_progress(args, rollout_id),
        "rollout_id": rollout_id,
        "event_ts": time.time(),
    }
    if step_event:
        payload["step_event"] = step_event
    match step_event:
        case "start":
            _enqueue(payload, timeout_seconds=_STEP_EVENT_TIMEOUT_SECONDS)
        case "finish":
            if rollout_id is not None:
                _enqueue(payload, timeout_seconds=_STEP_EVENT_TIMEOUT_SECONDS)
        case _:
            _enqueue(payload)


__all__ = [
    "before_log_prob_hook",
    "before_train_step_hook",
    "report_advantage_distribution",
    "report_phase",
    "report_rollout_samples",
    "report_step_event",
    "log_eval_rollout_data",
    "log_rollout_data",
    "RoleRecorder",
    "recording_lane",
    "recording_lane_on_reporting_rank",
    "time_phase",
]
