import dataclasses as _dc
from collections.abc import Mapping
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass
from pydantic_core import ArgsKwargs

from modal_dojo.common.errors import TrainingGymConfigError
from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

if TYPE_CHECKING:
    from modal_dojo.common.models import ModelConfig

_PATCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "frameworks"
    / "miles"
    / "modal_helpers"
    / "patches"
)

_PATCHES = (
    "patch_router_startup_timeout",
    "patch_gemma4_vl_rollout_text",
)


def _image_patches() -> list[str]:
    return [
        f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
        for name in _PATCHES
    ]


_EPHEMERAL_DISK_MIB = 1_048_576


_VISION_MODE: dict[str, Any] = {
    "rollout_top_p": 0.95,
    "rollout_top_k": 64,
    "sglang_max_running_requests": 8,
}


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Gemma4_26B_A4B_Recipe(MilesRecipe):
    """Gemma-4-26B-A4B recipe."""

    _SKIP_FIELDS: ClassVar[frozenset[str]] = MilesRecipe._SKIP_FIELDS | {"modality"}

    modality: Literal["text", "vision"] = "text"

    gpu_type: str = "B300"
    colocate: bool = False
    rollout_num_gpus: int | None = 1
    image_run_commands: list[str] = field(default_factory=_image_patches)

    hf_checkpoint: str = "google/gemma-4-26B-A4B-it"
    ref_load: str = "google/gemma-4-26B-A4B-it"
    miles_model_name: str = "gemma-4-26b-a4b-it"
    train_function_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"ephemeral_disk": _EPHEMERAL_DISK_MIB}
    )

    sglang_ep_size: int | None = 1
    moe_enable_deepep: bool = False
    moe_token_dispatcher_type: str = "alltoall"

    use_dynamic_batch_size: bool = False
    micro_batch_size: int = 1
    max_tokens_per_gpu: int = 1024

    balance_data: bool = True
    rollout_top_p: float | None = None
    rollout_stop_token_ids: list[int] | None = field(
        default_factory=lambda: [1, 106, 50]
    )

    rollout_health_check_first_wait: int = 300
    # Gemma-4's global head_dim=512 exceeds FlashAttention's 256 cap.
    sglang_attention_backend: str = "triton"
    sglang_moe_runner_backend: str = "triton"
    sglang_disable_custom_all_reduce: bool = True
    sglang_disable_cuda_graph: bool = True
    sglang_disable_overlap_schedule: bool = True
    sglang_disable_radix_cache: bool = True
    no_offload_train: bool = True
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    attention_backend: str = "unfused"
    qkv_format: str = "bshd"
    no_gradient_accumulation_fusion: bool = True
    no_check_for_nan_in_loss_and_grad: bool = True

    @model_validator(mode="before")
    @classmethod
    def _apply_vision_defaults(cls, data: Any) -> Any:
        """Fill unsupplied ``_VISION_MODE`` fields when ``modality="vision"``.

        Applied to the incoming arguments rather than to the built recipe, so a
        value the caller supplied is never overwritten and there is nothing to
        disambiguate afterwards. Rebuilding a resolved recipe from all of its
        fields is therefore a no-op.
        """
        if isinstance(data, ArgsKwargs):
            args, kwargs = data.args or (), dict(data.kwargs or {})
        elif isinstance(data, Mapping):
            args, kwargs = (), dict(data)
        else:
            return data

        names = [f.name for f in _dc.fields(cls) if f.init is not False]
        positional = set(names[: len(args)])
        if "modality" in positional:
            modality = args[names.index("modality")]
        else:
            modality = kwargs.get("modality", "text")
        if modality != "vision":
            return data

        for name, value in _VISION_MODE.items():
            if name not in positional and name not in kwargs:
                kwargs[name] = value
        return ArgsKwargs(args, kwargs) if isinstance(data, ArgsKwargs) else kwargs

    @model_validator(mode="after")
    def _keep_image_patches(self) -> "Gemma4_26B_A4B_Recipe":
        """Keep the build-time patches at the head of ``image_run_commands``.

        The field is replaced wholesale, so a caller adding their own command
        would otherwise drop the patches — and losing the VL one shows up as a
        blind model rather than an error.
        """
        patches = _image_patches()
        current = list(self.image_run_commands or [])
        if current[: len(patches)] != patches:
            object.__setattr__(
                self,
                "image_run_commands",
                [*patches, *(c for c in current if c not in patches)],
            )
        return self

    @model_validator(mode="after")
    def _keep_disk_reservation(self) -> "Gemma4_26B_A4B_Recipe":
        """Keep the disk reservation when a caller supplies their own kwargs.

        Same reasoning as ``_keep_image_patches``: passing ``{"secrets": [...]}``
        would otherwise drop the reservation and the run would die part-way through
        the checkpoint download. A caller who names ``ephemeral_disk`` still wins.
        """
        kwargs = self.train_function_kwargs or {}
        if "ephemeral_disk" not in kwargs:
            object.__setattr__(
                self,
                "train_function_kwargs",
                {"ephemeral_disk": _EPHEMERAL_DISK_MIB, **kwargs},
            )
        return self

    def _brings_own_reward(self) -> bool:
        if self.custom_rm_function is not None or self.rollout_function is not None:
            return True
        if self.rm_type:
            return True
        extra = self.extra_config
        if isinstance(extra, str) and extra:
            return True
        return isinstance(extra, dict) and bool(extra.get("custom_rm_path"))

    @model_validator(mode="after")
    def _require_vision_reward(self) -> "Gemma4_26B_A4B_Recipe":
        if self.modality == "vision" and not self._brings_own_reward():
            raise TrainingGymConfigError(
                f"{type(self).__name__}(modality='vision') needs its own reward. "
                "Pass custom_rm_function=... or rm_type=... to choose a built-in."
            )
        return self

    def validate_model_parallelism(self, model: "ModelConfig") -> None:
        super().validate_model_parallelism(model)
        if self.pipeline_model_parallel_size != 1:
            raise TrainingGymConfigError(
                f"{type(self).__name__} needs pipeline_model_parallel_size=1: the "
                "Megatron bridge loads the vision tower and the tied input/output "
                "embedding onto a single pipeline stage, so a split only fails once "
                f"Megatron builds the model. Got {self.pipeline_model_parallel_size}; "
                "scale with tensor_model_parallel_size or expert_model_parallel_size."
            )
