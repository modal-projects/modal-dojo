"""Stable import surface for miles' in-container dashboard reporting.

The build-time patches (``modal_helpers/patches/patch_rollout_status_reporting.py``
and ``patch_advantage_distribution.py``) and the recipe's default custom-function
paths import from this module *inside the training container*, so everything
they reference must stay importable here. The implementation is split across:

- :mod:`modal_dojo.common.reporting` — HTTP queue/URL/token plumbing +
  run-context helpers (shared with slime)
- :mod:`modal_dojo.common.sample_extraction` — trace/image/trajectory
  extraction from Samples (shared with slime; miles' Sample is the same shape)
- :mod:`.advantage_reporting` — torch/megatron advantage-distribution math

Mirror of :mod:`modal_dojo.frameworks.slime.phase_reporting` with two
differences: statuses come from :class:`MilesStatus`, and miles has no step
timing at all — no ``step_event`` markers are emitted, and the dashboard skips
step-time recording for miles runs entirely.
"""

from __future__ import annotations

import time
from typing import Any

from modal_dojo.common.reporting import (
    _enqueue,
    _enqueue_rollout,
    _run_context,
    _step_progress,
)
from modal_dojo.common.reporting import (
    _arg_value as _arg_value,
    _positive_int as _positive_int,
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
from modal_dojo.common.status import MilesStatus

from .advantage_reporting import (
    report_advantage_distribution as report_advantage_distribution,
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

    native_key = path_key.removeprefix("training_gym_")
    for container_name in ("extra_config", "custom_config"):
        container = getattr(args, container_name, None)
        if not isinstance(container, dict):
            continue
        for key in (path_key, native_key):
            value = container.get(key)
            if (
                isinstance(value, str)
                and value.strip()
                and not value.startswith("modal_dojo.")
            ):
                return value
    return None


def report_phase(
    status: MilesStatus,
    args: Any = None,
    **extra: Any,
) -> None:
    _enqueue(
        {**_run_context(args), "phase": status.value, "event_ts": time.time(), **extra}
    )


def report_step_event(
    status: MilesStatus | str,
    args: Any = None,
    rollout_id: int | None = None,
) -> None:
    """Report one training-loop phase transition tagged with ``status``.

    ``status`` may be a plain string — the patched miles train.py passes phase
    names as literals so the injected code stays stdlib-only.

    Unlike slime, miles does not track step times: there is no ``step_event``
    parameter, so the dashboard gets phase + progress updates without
    start/finish timing events.
    """
    payload = {
        **_run_context(args),
        "phase": status.value if isinstance(status, MilesStatus) else str(status),
        **_step_progress(args, rollout_id),
        "rollout_id": rollout_id,
        "event_ts": time.time(),
    }
    _enqueue(payload)


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
    report_phase(
        MilesStatus.ROLLOUT_LOGGING,
        args,
        **_step_progress(args, rollout_id),
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
    extra_metrics: Any = None,
) -> bool:
    report_phase(
        MilesStatus.EVAL_ROLLOUT_LOGGING,
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
    # rollout_id, so the compute_log_probs phase is reported from the driver
    # loop right before actor_model.train(); see the rollout-status patch.
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
        MilesStatus.OPTIMIZER_STEP,
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


__all__ = [
    "before_log_prob_hook",
    "before_train_step_hook",
    "report_advantage_distribution",
    "report_phase",
    "report_rollout_samples",
    "report_step_event",
    "log_eval_rollout_data",
    "log_rollout_data",
]
