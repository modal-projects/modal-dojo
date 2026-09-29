"""Mirror the scalars Slime and Miles log through ``wandb`` into the dashboard.

A ``.pth`` in the training image imports this module in every interpreter, and
``bootstrap()`` picks the ``wandb`` surface from ``TRAINING_GYM_METRIC_PROVIDER``:

- ``dashboard`` (``DashboardMetricConfig``): a W&B-shaped shim whose only sink
  is the dashboard.
- ``trackio``: Trackio's adapter (``trackio.py``), which also calls ``mirror_log``.
- ``wandb``: the real package, with ``Run.log`` patched to also call ``mirror_log``.

Points are coalesced per step and shipped on the reporting queue, so they carry
the per-run bearer token, never block training and drain at process exit.
"""

from __future__ import annotations

import math
import os
import shlex
import sys
import threading
import time
import types
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from typing import Any, ClassVar

from modal_training_gym.common.metrics import MetricConfig

PROVIDER_ENV = "TRAINING_GYM_METRIC_PROVIDER"
FLUSH_INTERVAL_SECONDS = 2.0

_PTH_FILE = "_training_gym_metric_mirror.pth"
_PTH_LINE = (
    "import os; not os.environ.get('TRAINING_GYM_METRIC_PROVIDER') "
    "or __import__('modal_training_gym.common.metric_mirror', "
    "fromlist=['bootstrap']).bootstrap()\n"
)


@dataclass
class DashboardMetricConfig(MetricConfig):
    """Log framework metrics to the Training Gym dashboard only.

    Slime and Miles keep calling ``wandb.*``; a W&B-shaped shim routes the
    scalars to the dashboard's Metrics tab. No W&B account, API key or Trackio
    server is involved.

    Attributes:
        project: Metric project name.
        group: Group tag for related runs.
        exp_name: Run display name.
    """

    project: str = "training-gym"
    group: str = ""
    exp_name: str = ""
    disable_random_suffix: bool = True

    provider: ClassVar[str] = "dashboard"  # pyright: ignore[reportIncompatibleMethodOverride]


def pth_install_command() -> str:
    """Shell command writing the bootstrap ``.pth`` into the image's site-packages."""
    code = (
        "import pathlib, site; "
        f"pathlib.Path(site.getsitepackages()[0], {_PTH_FILE!r}).write_text({_PTH_LINE!r})"
    )
    return f"python3 -c {shlex.quote(code)}"


def bootstrap() -> None:
    provider = os.environ.get(PROVIDER_ENV)
    if provider == DashboardMetricConfig.provider:
        install_wandb_shim()
    elif provider == "trackio":
        from modal_training_gym.common.trackio import install_wandb_shim as trackio_shim

        trackio_shim()
    elif provider == "wandb":
        install_wandb_tee()


# ── Client-side buffer ─────


def flatten_numeric(data: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """``{"train": {"loss": 0.4}}`` -> ``{"train/loss": 0.4}``, dropping what
    W&B would not chart (bools, strings, media, NaN/inf)."""
    out: dict[str, float] = {}
    for raw_key, value in data.items():
        key = f"{prefix}/{raw_key}" if prefix else str(raw_key)
        if isinstance(value, Mapping):
            out.update(flatten_numeric(value, key))
            continue
        if not isinstance(value, (int, float)):
            try:
                value = value.item()  # 0-d tensors and numpy scalars
            except Exception:
                continue
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            out[key] = float(value)
    return out


class MetricMirror:
    """Coalesces ``log`` calls per step and posts them every few seconds.

    Step semantics follow ``wandb.log``: ``step=None`` lands on an implicit
    counter that advances unless ``commit=False``; an explicit step moves the
    counter forward, never back.
    """

    def __init__(self, training_run_id: str) -> None:
        self.training_run_id = training_run_id
        self._pending: dict[int, dict[str, Any]] = {}
        self._next_step = 0
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def log(
        self,
        data: Mapping[str, Any],
        *,
        step: int | None = None,
        commit: bool | None = None,
    ) -> None:
        metrics = flatten_numeric(data)
        with self._lock:
            if step is None:
                step = self._next_step
                if commit is not False:
                    self._next_step += 1
            elif (step := int(step)) < 0:
                return
            else:
                self._next_step = max(self._next_step, step)
            if not metrics:
                return
            point = self._pending.setdefault(step, {"step": step, "metrics": {}})
            point["metrics"].update(metrics)
            point["time"] = time.time()
            if self._timer is None:
                self._timer = threading.Timer(FLUSH_INTERVAL_SECONDS, self.flush)
                self._timer.daemon = True
                self._timer.start()

    def flush(self, final: bool = False) -> None:
        with self._lock:
            if (
                self._timer is not None
                and self._timer is not threading.current_thread()
            ):
                self._timer.cancel()
            self._timer = None
            points = [point for _, point in sorted(self._pending.items())]
            self._pending.clear()
        if points:
            from modal_training_gym.common.reporting import _enqueue_metric_points

            _enqueue_metric_points(
                {"training_run_id": self.training_run_id, "points": points},
                final=final,
            )


_MIRROR: MetricMirror | None = None
_MIRROR_LOCK = threading.Lock()


def mirror_log(
    data: Any, *, step: int | None = None, commit: bool | None = None
) -> None:
    """Best-effort mirror of one ``wandb.log`` call; never raises."""
    global _MIRROR
    try:
        with _MIRROR_LOCK:
            if _MIRROR is None:
                if not (run_id := os.environ.get("TRAINING_GYM_TRAINING_RUN_ID")):
                    return
                from modal_training_gym.common.reporting import register_pre_drain_hook

                mirror = _MIRROR = MetricMirror(run_id)
                register_pre_drain_hook(lambda: mirror.flush(final=True))
        if isinstance(data, Mapping):
            _MIRROR.log(data, step=step, commit=commit)
    except Exception:
        pass


# ── wandb: patch the real package as it is imported ─────


class _WandbTeeFinder:
    """Wraps ``wandb``'s loader so the patch lands on import, not at startup."""

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> ModuleSpec | None:
        if fullname != "wandb":
            return None
        sys.meta_path.remove(self)
        spec = find_spec("wandb")
        loader: Any = spec.loader if spec is not None else None
        if loader is not None:
            exec_module = loader.exec_module

            def exec_and_patch(module: Any) -> None:
                exec_module(module)
                patch_wandb_module()

            loader.exec_module = exec_and_patch
        return spec


def install_wandb_tee() -> None:
    if "wandb" in sys.modules:
        patch_wandb_module()
    else:
        sys.meta_path.insert(0, _WandbTeeFinder())


def patch_wandb_module() -> None:
    """Tee ``Run.log``, which ``wandb.log`` dispatches to after ``init``."""
    run_cls = import_module("wandb.sdk.wandb_run").Run
    if "_training_gym_mirror" in vars(run_cls):
        return
    run_cls._training_gym_mirror = True
    original_log = run_cls.log

    def log(
        self: Any, data: Any, step: Any = None, commit: Any = None, *a: Any, **k: Any
    ) -> Any:
        result = original_log(self, data, step, commit, *a, **k)
        mirror_log(data, step=step, commit=commit)
        return result

    run_cls.log = log


# ── dashboard: a W&B-shaped module over the dashboard ─────


class _Config(dict):
    """``wandb.config``: attribute access plus ``update(namespace)``."""

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        kwargs.pop("allow_val_change", None)
        for arg in args:
            if arg is not None:
                super().update(arg if isinstance(arg, Mapping) else vars(arg))
        super().update(kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__


class _Media:
    """Stand-in for ``wandb.Image``/``Table``/``Settings``; dropped by the flattener."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class DashboardRun:
    """What the shim's ``wandb.init`` returns."""

    def __init__(self, run_id: str, name: str, project: str, config: Any) -> None:
        self.id, self.name, self.project, self.url = run_id, name, project, ""
        self.summary: dict[str, Any] = {}
        self.config = _Config()
        self.config.update(config)

    def log(
        self, data: Any, step: Any = None, commit: Any = None, *_a: Any, **_k: Any
    ) -> None:
        mirror_log(data, step=step, commit=commit)

    def __getattr__(self, name: str) -> Any:
        # finish, define_metric, save, watch, alert, ...: accepted and ignored.
        return lambda *args, **kwargs: None


def generate_id() -> str:
    return uuid.uuid4().hex[:8]


def install_wandb_shim() -> None:
    """Expose the dashboard as the W&B subset used by Slime and Miles."""
    modules: dict[str, Any] = {}
    for name in (
        "wandb",
        "wandb.util",
        "wandb.sdk",
        "wandb.sdk.lib",
        "wandb.sdk.lib.runid",
    ):
        module: Any = types.ModuleType(name)
        modules[name] = module
        module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        module.__path__ = []
        module.generate_id = generate_id
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
    sys.modules.update(modules)

    shim = modules["wandb"]
    shim.run = None
    shim.config = _Config()
    shim.Settings = _Media

    def init(*args: Any, **kwargs: Any) -> DashboardRun:
        run_id = str(kwargs.get("id") or kwargs.get("name") or generate_id())
        shim.run = DashboardRun(
            run_id,
            str(kwargs.get("name") or run_id),
            str(kwargs.get("project", args[0] if args else "") or "training-gym"),
            kwargs.get("config"),
        )
        shim.config = shim.run.config
        return shim.run

    def finish(*_args: Any, **_kwargs: Any) -> None:
        shim.run = None

    shim.init = init
    shim.log = lambda data, step=None, commit=None, *a, **k: mirror_log(
        data, step=step, commit=commit
    )
    shim.finish = finish
    shim.save = lambda *args, **kwargs: []
    shim.login = lambda *args, **kwargs: True
    shim.define_metric = lambda *args, **kwargs: None

    def missing(name: str) -> Any:
        if name[:1].isupper():
            return _Media  # wandb.Image, wandb.Table, wandb.Histogram, ...
        raise AttributeError(name)

    shim.__getattr__ = missing
