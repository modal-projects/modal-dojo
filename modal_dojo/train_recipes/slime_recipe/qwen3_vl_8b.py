from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_VL_8B_Recipe(SlimeRecipe):
    """Qwen3-VL-8B recipe."""

    gpu_type: str = "H200"
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True
    use_dynamic_batch_size: bool = False
    extra_config: dict | None = field(
        default_factory=lambda: {"qkv_format": "bshd", "micro_batch_size": 1}
    )

    megatron_to_hf_mode: str = "bridge"

    sglang_mem_fraction_static: float = 0.55
    freeze_params_name_list: list[str] | None = field(
        default_factory=lambda: ["vision_model"]
    )
