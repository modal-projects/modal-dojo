from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import ModelConfig, Moonlight_16B_A3B_Instruct
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Moonlight_16B_A3B_Recipe(MilesRecipe):
    """Moonlight-16B-A3B recipe."""

    model_config_class: ClassVar[type[ModelConfig]] = Moonlight_16B_A3B_Instruct

    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "CONVERT_KEEP_PP1": "1",
        }
    )
    ref_load: str = "/checkpoints/Moonlight-16B-A3B-Instruct_torch_dist"
    megatron_to_hf_mode: str = ""

    use_miles_router: bool = True

    global_batch_size: int | None = None
    balance_data: bool = True

    gpu_type: str = "H200"

    recompute_granularity: str | None = "full"
    recompute_method: str | None = "uniform"
    recompute_num_layers: int | None = 1
    max_tokens_per_gpu: int = 8192

    use_kl_loss: bool = True

    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    sglang_mem_fraction_static: float = 0.7
    sglang_cuda_graph_bs: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8] + list(range(16, 257, 8))
    )

    moe_enable_deepep: bool = False
    moe_token_dispatcher_type: str = "alltoall"
