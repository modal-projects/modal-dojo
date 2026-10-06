"""GPT-OSS-120B LoRA GRPO recipe, derived from upstream's ``run_gpt_oss_20b.py``."""

from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import GPT_OSS_120B, ModelConfig
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

# Includes GPT-OSS attention sinks and sliding windows in Megatron and SGLang.
_DOCKER_IMAGE = "radixark/miles:dev-202609251434"


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class GPT_OSS_120B_LoRA_Recipe(MilesRecipe):
    """GPT-OSS-120B rank-32 LoRA recipe for one node of eight H100 GPUs.

    Bridge mode trains from the MXFP4 release directly, so there is no
    torch_dist conversion and no Megatron reference model (no KL loss).
    """

    model_config_class: ClassVar[type[ModelConfig]] = GPT_OSS_120B

    docker_image: str = _DOCKER_IMAGE
    # Each of the eight bridge ranks mmaps the whole 65 GB MXFP4 checkpoint and
    # the sandbox charges those file pages to the container, on top of the BF16
    # host backups of the frozen base; a 512 GiB request was killed mid-load.
    # 1.5 TiB is the largest request that still schedules on 1.8-2 TiB H100
    # hosts (90% of host RAM is allocatable); 1792 GiB only fits 2 TiB hosts.
    memory: tuple[int, int] = (1536 * 1024, 2048 * 1024)

    # ── Cluster: TP8 x EP8 on one node (DP1), as in upstream's 20B run ──────
    actor_num_gpus_per_node: int = 8
    tensor_model_parallel_size: int = 8
    sequence_parallel: bool = True
    expert_model_parallel_size: int = 8

    # ── Attention: sinks need bshd, which rules out dynamic batching ─────────
    qkv_format: str = "bshd"
    attention_backend: str | None = "fused"
    use_dynamic_batch_size: bool = False
    micro_batch_size: int | None = 1
    recompute_granularity: str | None = "full"
    recompute_method: str | None = "uniform"
    recompute_num_layers: int | None = 1

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_rank: int | None = 32
    lora_alpha: int | None = 32
    lora_dropout: float | None = 0.0
    # Resolves to GPT-OSS's HF targets: q/k/v/o and the packed experts.
    target_modules: str | None = "all-linear"
    lora_base_cpu_backup: bool = True
    no_gradient_accumulation_fusion: bool = True

    # ── Rollout and sampling ─────────────────────────────────────────────────
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    rollout_max_response_len: int = 8192

    # ── Optimizer ────────────────────────────────────────────────────────────
    lr: float = 1e-5

    # ── SGLang: one TP8 engine serving the MXFP4 release ─────────────────────
    rollout_num_gpus_per_engine: int = 8
    sglang_mem_fraction_static: float = 0.7
    sglang_dtype: str = "bfloat16"
    # Marlin supports MXFP4 experts with LoRA.
    sglang_moe_runner_backend: str | None = "marlin"
    sglang_lora_backend: str | None = "triton"
    sglang_max_lora_rank: int = 32
    sglang_max_loras_per_batch: int = 1
    no_sglang_lora_use_virtual_experts: bool = True
    # The engine holds MXFP4 experts; the trainer holds their BF16 dequant.
    check_weight_update_allow_quant_error: bool = True
