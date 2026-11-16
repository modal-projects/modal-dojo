"""DeepSeek V4.1 Flash merged LoRA with rollout on Stitch."""

from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models.base import ModelConfig
from modal_dojo.common.models.deepseek_v41_flash import DeepSeek_V4_1_Flash_LoRA
from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.stitch_lora import (
    MILES_ENVIRONMENT,
    _Stitch_LoRA_Recipe,
    stitch_lora_extra_config,
)


def _environment() -> dict[str, str]:
    return {
        **MILES_ENVIRONMENT,
        # The CuTe DSL NVFP4 runner in W4A16 with its deterministic finalize; four-over-six weights reproduce
        # the release's MXFP4 experts; Miles' fused hyper-connection kernels.
        "SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16": "1",
        "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE": "0",
        "NVTE_NVFP4_4OVER6": "weights",
        "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0",
        "SGLANG_OPT_FUSE_WQA_WKV": "0",
        "MILES_DSV41_HC_FUSED": "1",
    }


def _extra_config() -> dict:
    return stitch_lora_extra_config(
        log_probs_chunk_size=2048,
        moe_router_freeze_gate=True,
        freeze_e_score_correction_bias=True,
        moe_router_dtype="fp32",
        grad_reduce_in_bf16=True,
        transformer_impl="transformer_engine",
        sglang_attn_dp_size=4,
        sglang_quantization="modelopt_fp4",
        sglang_dsv4_attn_backend="flashmla",
        sglang_enable_symm_mem=True,
        sglang_reasoning_parser="deepseek-v41",
    )


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class DeepSeek_V4_1_Flash_LoRA_Stitch_Recipe(_Stitch_LoRA_Recipe):
    """DeepSeek V4.1 Flash, rank-32 merged LoRA on every linear (routed experts shared-outer), fully async on Stitch.

    The trainer is one 8-GPU B300 node holding the frozen BF16 base (TP1, CP4 with all-gathered KV, EP8) and packs
    up to 8K tokens per GPU. Rollout runs on two 4-GPU B300 engines serving NVFP4 (attention DP4, EP4), so a
    replaced engine never empties the pool.

    Both bases are made once with Miles' tools, from the release's ``model_name`` revision; Stitch's
    ``recipes/prep.py`` and ``tests/e2e/checkpoints.py`` record the chain:

    - ``ref_load``: the release dequantized to BF16 (``tools/convert_deepseek_v4_to_bf16.py``), with the tensors
      the trainer keeps in FP32 stored as FP32.
    - ``hf_checkpoint``, the engines' base: ``tools/convert_hf_to_nvfp4.py`` on that BF16 copy, on a B200, in this
      recipe's environment (four-over-six weights reproduce the release's MXFP4 experts).
    """

    model_config_class: ClassVar[type[ModelConfig]] = DeepSeek_V4_1_Flash_LoRA
    hf_checkpoint: str = "/checkpoints/deepseek-v4-1-flash-1017-nvfp4-miles"
    ref_load: str = "/checkpoints/deepseek-v4-1-flash-1017-bf16-miles"
    miles_model_name: str = "deepseek-v4.1"
    model_name: str = "deepseekv41"
    environment: dict = field(default_factory=_environment)
    # The node's budget: eight training ranks, the rollout executor and the weight export.
    cpu: float | tuple[float, float] | None = 64.0
    memory: int | tuple[int, int] | None = (1792 * 1024, 1792 * 1024)

    # 64 attention heads stay whole per GPU: CP splits the sequence, never the heads.
    context_parallel_size: int = 4
    allgather_cp: bool = True
    dsv4_impl: str | None = "miles"
    expert_model_parallel_size: int = 8
    max_tokens_per_gpu: MicroBatchSize = 8192
    accumulate_allreduce_grads_in_fp32: bool = False

    rollout_num_gpus: int | None = 8
    rollout_num_gpus_per_engine: int = 4
    sglang_ep_size: int | None = 4
    sglang_mem_fraction_static: float = 0.8
    sglang_moe_runner_backend: str | None = "flashinfer_cutedsl"
    extra_config: dict | None = field(default_factory=_extra_config)
