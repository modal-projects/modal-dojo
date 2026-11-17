"""GLM-5.3-Flash merged LoRA, disaggregated: rollout on a Stitch engine pool outside the training cluster."""

from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models.base import ModelConfig
from modal_dojo.common.models.glm_5_3_flash import GLM_5_3_Flash_LoRA
from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.disaggregated_lora import (
    MILES_ENVIRONMENT,
    _Disaggregated_LoRA_Recipe,
    disaggregated_lora_extra_config,
)


def _environment() -> dict[str, str]:
    return {
        **MILES_ENVIRONMENT,
        # The CuTe DSL NVFP4 runner in W4A16 with its deterministic finalize; TE's four-over-six quantizer for the
        # weight export, as the engines' base was converted.
        "SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16": "1",
        "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE": "0",
        "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
        "NVTE_NVFP4_DISABLE_RHT": "1",
        "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
        "NVTE_USE_FAST_MATH": "0",
        "NVTE_NVFP4_4OVER6": "all",
        "NVTE_NVFP4_4OVER6_E4M3_USE_256": "none",
        "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
        "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
    }


def _extra_config() -> dict:
    return disaggregated_lora_extra_config(
        # The absorbed DSA never runs kv_b_proj as a linear, so it takes no adapter.
        exclude_modules="*.self_attn.kv_b_proj",
        log_probs_chunk_size=2048,
        transformer_impl="transformer_engine",
        sglang_attn_dp_size=4,
        sglang_quantization="modelopt_fp4",
        sglang_dsa_prefill_backend="tilelang",
        sglang_dsa_decode_backend="tilelang",
        sglang_kv_cache_dtype="bfloat16",
        sglang_chunked_prefill_size=8192,
        sglang_disable_radix_cache=True,
    )


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class GLM_5_3_Flash_LoRA_Disaggregated_Recipe(_Disaggregated_LoRA_Recipe):
    """GLM-5.3-Flash, rank-32 merged LoRA on every linear but ``kv_b_proj`` (routed experts shared-outer), fully async on Stitch.

    The trainer is one 8-GPU B300 node holding the frozen BF16 base (TP2 with sequence parallelism, EP8). Rollout
    runs on two 4-GPU B300 engines serving NVFP4 (attention DP4, EP4, TileLang DSA), so a replaced engine never
    empties the pool.

    - ``ref_load``: the BF16 release, ``zai-org/GLM-5.3-Flash-BF16``, in the Hugging Face cache; Miles reads it
      directly.
    - ``hf_checkpoint``, the engines' base: Miles' ``tools/convert_hf_to_nvfp4.py`` on that release, in this
      recipe's environment, keeping the tensors the trainer holds in FP32 as FP32. Stitch's ``recipes/prep.py`` and
      ``tests/e2e/checkpoints.py`` record the conversion.
    """

    model_config_class: ClassVar[type[ModelConfig]] = GLM_5_3_Flash_LoRA
    hf_checkpoint: str = "/checkpoints/glm5-3-flash-nvfp4-trainer"
    ref_load: str = (
        "/root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash-BF16/snapshots/"
        "f9fed3bbc82291073373d67452e9f25317ef7577"
    )
    miles_model_name: str = "glm5.3-flash"
    model_name: str = "glm5_next"
    environment: dict = field(default_factory=_environment)

    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True
    expert_model_parallel_size: int = 8
    use_dynamic_batch_size: bool = False
    micro_batch_size: MicroBatchSize | None = 1
    max_tokens_per_gpu: MicroBatchSize = 32768

    rollout_num_gpus: int | None = 8
    rollout_num_gpus_per_engine: int = 4
    sglang_ep_size: int | None = 4
    sglang_mem_fraction_static: float = 0.7
    sglang_moe_runner_backend: str | None = "flashinfer_cutedsl"
    extra_config: dict | None = field(default_factory=_extra_config)
