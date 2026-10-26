from dataclasses import field
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import ModelConfig, Qwen3_VL_8B
from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

_VL_PATCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "frameworks"
    / "miles"
    / "modal_helpers"
    / "patches"
)
_VL_PATCHES = (
    "patch_qwen3_vl_export",
    "patch_qwen3_vl_torch_dist",
)


def _vl_image_run_commands() -> list[str]:
    return [
        f"echo {encode_patch(name, _VL_PATCH_DIR)} | base64 -d | python3"
        for name in _VL_PATCHES
    ]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_VL_8B_Recipe(MilesRecipe):
    """Qwen3-VL-8B recipe."""

    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_VL_8B
    sft_supported: ClassVar[bool] = False

    recompute_granularity: str | None = "full"

    gpu_type: str = "H200"
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True
    use_dynamic_batch_size: bool = False
    extra_config: dict | None = field(
        default_factory=lambda: {"qkv_format": "bshd", "micro_batch_size": 1}
    )

    sglang_mem_fraction_static: float = 0.55

    image_run_commands: list[str] = field(default_factory=_vl_image_run_commands)
