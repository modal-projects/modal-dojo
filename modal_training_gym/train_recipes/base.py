import dataclasses as _dc
import hashlib
import json
import os
import re
import uuid
from abc import ABC
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.train_recipes.gpu_allocation import (
    GpuAllocation,
    resolve_gpu_allocation,
)

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.metrics import MetricConfig
    from modal_training_gym.common.models import ModelConfig

# ── Volume mount paths (shared by every framework) ───────────────────────────

HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")


def _safe_data_key(cache_key: str) -> str:
    """Return a readable filename component with collision-resistant identity."""
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "-", cache_key).strip("._-")
    prefix = prefix[:48] or "dataset"
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    return f"{prefix}-{digest}"


# Recipe fields whose dict values are emitted as JSON CLI arguments.
JSON_CONFIG_FIELDS = ("train_env_vars", "apply_chat_template_kwargs", "multimodal_keys")

_SFT_CLI_OVERRIDES: dict[str, Any] = {
    "colocate": False,
    "rollout_num_gpus": None,
    "calculate_per_token_loss": True,
    "disable_compute_advantages_and_returns": True,
    "debug_train_only": True,
    "n_samples_per_prompt": 1,
    "use_fault_tolerance": False,
    "use_kl_loss": False,
    "kl_coef": 0.0,
    "entropy_coef": 0.0,
    "apply_chat_template": False,
}


def _apply_loss_type_fields(
    fields: dict[str, Any], *, sft_rollout_function: str
) -> None:
    if fields["num_epoch"] is not None:
        fields["num_rollout"] = None
    if fields["loss_type"] == "policy_loss":
        fields["loss_type"] = None
        fields["loss_mask_type"] = None
        return
    fields.update(_SFT_CLI_OVERRIDES)
    global_batch_size = fields.get("global_batch_size")
    rollout_batch_size = fields.get("rollout_batch_size")
    if global_batch_size is not None:
        fields["rollout_batch_size"] = global_batch_size
    elif rollout_batch_size is not None:
        fields["global_batch_size"] = rollout_batch_size
    fields["rollout_function"] = sft_rollout_function
    if fields.get("loss_mask_type") == "qwen":
        fields["loss_mask_type"] = None
    if fields["advantage_estimator"] == "ppo":
        fields["advantage_estimator"] = "grpo"
    if "num_steps_per_rollout" in fields:
        fields["num_steps_per_rollout"] = 1


class BaseTrainRecipe(ABC):
    model_config_class: ClassVar["type[ModelConfig] | None"] = None

    # Fields consumed by the Modal launcher (image build, cluster topology,
    # callable shipping) and never forwarded to the framework CLI. Every
    # framework recipe overrides this; a name missing from it leaks onto the
    # command line and the framework's argparse aborts with "unrecognized
    # arguments".
    _SKIP_FIELDS: ClassVar[frozenset[str]] = frozenset()

    # Field holding the inline-YAML escape-hatch dict. ``_emit_fields`` renames
    # it to ``_ESCAPE_HATCH_FLAG`` on the command line, and keys inside it always
    # win over same-named top-level fields.
    _ESCAPE_HATCH_FIELD: ClassVar[str] = "extra_config"
    _ESCAPE_HATCH_FLAG: ClassVar[str] = "custom_config_path"

    # ── Callable → import path ────────────────────────────────────────────────

    @staticmethod
    def _callable_path(fn: Callable) -> str:
        """Dotted import path a container can resolve this callable by.

        A function defined in a ``__main__`` tutorial script has no importable
        module name, so fall back to its file stem — which is what the launcher
        mounts it as. ``__pending__`` marks a callable with no readable source
        file, for the launcher to fill in when it ships it.
        """
        mod = getattr(fn, "__module__", None) or ""
        name = getattr(fn, "__qualname__", None) or fn.__name__
        if mod == "__main__":
            import inspect

            try:
                src_file = inspect.getfile(fn)
                mod = Path(src_file).stem if os.path.isfile(src_file) else "__pending__"
            except (TypeError, OSError):
                mod = "__pending__"
        return f"{mod}.{name}"

    @classmethod
    def _path_or_callable_path(cls, value: "Callable | str | None") -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return cls._callable_path(value)

    # ── Model presets ─────────────────────────────────────────────────────────

    @classmethod
    def get_base_recipe(cls, model_config: "ModelConfig") -> "BaseTrainRecipe | None":
        """Return the model preset for ``model_config``.

        Call this explicitly. ``TrainConfig`` uses the recipe it receives
        without applying a preset.

        Returns:
            The model preset, or ``None`` when no preset is registered.

        Raises:
            TrainingGymConfigError:
                The recipe accepts only registered models and ``model_config`` is not
                registered.
        """
        return None

    def validate_model_parallelism(self, model: "ModelConfig") -> None:
        """Validate the model's parallelism settings."""
        return None

    # ── Container → framework flag converters ────────────────────────────────

    @staticmethod
    def _resolve_data_paths(
        ds: "DatasetConfig",
    ) -> str:
        """Return the materialized path for one dataset instance."""
        cache_key = ds.cache_key()
        if cache_key is None:
            cache_key = str(uuid.uuid4())
        safe_key = _safe_data_key(str(cache_key))
        return f"{DATA_PATH}/{safe_key}.{ds.output_format()}"

    @staticmethod
    def _validate_datasets(
        ds: "DatasetConfig",
        eval_ds: "DatasetConfig | None" = None,
        *,
        loss_type: str = "policy_loss",
    ) -> None:
        if loss_type == "sft_loss" and eval_ds is not None:
            raise TrainingGymConfigError(
                "eval_dataset is not supported with loss_type='sft_loss'"
            )
        if eval_ds is None:
            return
        for dataset_method in ("input_key", "label_key", "apply_chat_template"):
            train_value = getattr(ds, dataset_method)()
            eval_value = getattr(eval_ds, dataset_method)()
            if train_value != eval_value:
                raise TrainingGymConfigError(
                    f"Training and evaluation datasets must use the same "
                    f"{dataset_method}(): got {train_value!r} and {eval_value!r}."
                )

    @classmethod
    def _dataset_to_fields(
        cls,
        ds: "DatasetConfig",
        eval_ds: "DatasetConfig | None" = None,
        *,
        dataset_path: str | None = None,
        eval_dataset_path: str | None = None,
    ) -> dict[str, Any]:
        cls._validate_datasets(ds, eval_ds)
        return {
            "prompt_data": dataset_path,
            "eval_prompt_data": (
                ["eval", eval_dataset_path]
                if eval_ds is not None and eval_dataset_path is not None
                else None
            ),
            "input_key": ds.input_key(),
            "label_key": ds.label_key(),
            "apply_chat_template": ds.apply_chat_template(),
        }

    @staticmethod
    def _metrics_to_fields(metric: "MetricConfig") -> dict[str, Any]:
        from modal_training_gym.common.metrics import metric_cli_fields

        return metric_cli_fields(metric)

    # ── CLI serialization ─────────────────────────────────────────────────────

    def _field_values(self) -> dict[str, Any]:
        """Every declared field's current value, as the starting point for CLI emission."""
        return {f.name: getattr(self, f.name) for f in _dc.fields(self)}

    def _escape_hatch_keys(self) -> tuple[str, ...]:
        """Keys set in the inline-YAML escape hatch, on either side of materialization.

        ``prepare_launch_config`` replaces the dict with a path to the YAML it
        wrote *before* the launcher calls ``cli_args``, so reading the field
        alone would see a ``str`` and silently skip the precedence rule in
        ``_emit_fields``. It records the keys on the recipe as it materializes;
        fall back to those.
        """
        val = getattr(self, self._ESCAPE_HATCH_FIELD, None)
        if isinstance(val, dict):
            return tuple(val)
        return tuple(getattr(self, "_materialized_config_keys", ()) or ())

    def _escape_hatch_values(self) -> dict[str, Any]:
        """The escape-hatch mapping, before or after it is written to YAML."""
        val = getattr(self, self._ESCAPE_HATCH_FIELD, None)
        if isinstance(val, dict):
            return val
        stored = getattr(self, "_materialized_config", None)
        return stored if isinstance(stored, dict) else {}

    def _emit_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Drop launcher-only fields and let the escape hatch win over same-named flags.

        The escape hatch is an explicit per-recipe override, but the frameworks
        resolve ``--<flag>`` CLI args *over* the YAML custom-config — so a key a
        recipe sets in both places would be clobbered by the field's own flag
        (e.g. a ``qkv_format="thd"`` default overriding an ``extra_config``
        ``"bshd"``). Drop any such flag so the YAML value stands.
        """
        out = {k: v for k, v in fields.items() if k not in self._SKIP_FIELDS}
        hatch = self._ESCAPE_HATCH_FIELD
        for key in self._escape_hatch_keys():
            if key != hatch:
                out.pop(key, None)
        if hatch in out:
            out[self._ESCAPE_HATCH_FLAG] = out.pop(hatch)
        return out

    def _fields(
        self,
        dataset: "DatasetConfig | None" = None,
        eval_dataset: "DatasetConfig | None" = None,
        dataset_path: str | None = None,
        eval_dataset_path: str | None = None,
        model: "ModelConfig | None" = None,
    ) -> dict[str, Any]:
        """Recipe fields to emit as CLI flags, merged with dataset/model/wandb.

        Not abstract: lightweight subclasses (e.g. test doubles) may skip it,
        in which case ``cli_args`` is unavailable.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _fields() to use cli_args()"
        )

    def cli_args(
        self,
        dataset: "DatasetConfig | None" = None,
        eval_dataset: "DatasetConfig | None" = None,
        dataset_path: str | None = None,
        eval_dataset_path: str | None = None,
        model: "ModelConfig | None" = None,
    ) -> list[str]:
        out: list[str] = []
        for key, val in self._fields(
            dataset=dataset,
            eval_dataset=eval_dataset,
            dataset_path=dataset_path,
            eval_dataset_path=eval_dataset_path,
            model=model,
        ).items():
            if val is None or val is False or val == "":
                continue
            flag = f"--{key.replace('_', '-')}"
            if val is True:
                out.append(flag)
            elif isinstance(val, dict) and key in JSON_CONFIG_FIELDS:
                out += [flag, json.dumps(val)]
            elif isinstance(val, list):
                out += [flag] + [str(v) for v in val]
            else:
                out += [flag, str(val)]
        return out

    # ── Cluster topology ──────────────────────────────────────────────────────

    @property
    def total_nodes(self) -> int:
        return self.gpu_allocation.total_nodes

    @property
    def gpu_allocation(self) -> GpuAllocation:
        return resolve_gpu_allocation(self, warn=False)
