from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.models import ModelConfig, Qwen3_5_4B
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_5_4B_Miles_Recipe(MilesRecipe):
    """Qwen3.5-4B recipe."""

    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_5_4B
    sft_supported: ClassVar[bool] = False

    miles_model_name: str = "qwen3.5-4B"
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "CONVERT_KEEP_PP1": "1",
        }
    )
    ref_load: str = "/checkpoints/Qwen3.5-4B_torch_dist"
    megatron_to_hf_mode: str = ""

    balance_data: bool = True

    recompute_granularity: str | None = "full"
    recompute_method: str | None = "uniform"
    recompute_num_layers: int | None = 1

    use_kl_loss: bool = True

    sglang_mem_fraction_static: float = 0.7

    attention_backend: str | None = "flash"
