"""Muse Spark Open merged LoRA, disaggregated: rollout on a Stitch engine pool outside the training cluster."""

from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models.base import ModelConfig
from modal_dojo.common.models.muse_spark import Muse_Spark_Open
from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.disaggregated_lora import (
    _Disaggregated_LoRA_Recipe,
    disaggregated_lora_extra_config,
)


def _extra_config() -> dict:
    return disaggregated_lora_extra_config(
        log_probs_chunk_size=4096,
        rollout_max_prompt_len=2048,
        # Megatron takes it as a list; Miles sets ["p2p"] before applying this config.
        cp_comm_type=["a2a"],
        use_distributed_optimizer=True,
        # DP attention 8 with EP8, BF16 weights and KV, the language model only; the running-request and
        # decode-graph caps keep SGLang's sliding-window KV pool able to hold a request.
        sglang_attn_dp_size=8,
        sglang_dtype="bfloat16",
        sglang_kv_cache_dtype="bfloat16",
        sglang_language_model_only=True,
        sglang_cuda_graph_max_bs_decode=32,
        sglang_context_length=32768,
    )


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Muse_Spark_Open_LoRA_Disaggregated_Recipe(_Disaggregated_LoRA_Recipe):
    """Muse Spark Open, rank-32 merged LoRA on every linear (routed experts shared-outer), fully async on Stitch.

    The trainer is two 8-GPU B300 nodes holding the frozen BF16 base (PP2, CP8 all-to-all, EP8; ~113 GiB of base per
    GPU). Rollout runs on two 8-GPU B300 engines serving BF16, language model only (attention DP8, EP8), so a
    replaced engine never empties the pool.

    - ``ref_load``: the release, stacked as published; Miles' Muse Spark bridge reads it directly.
    - ``hf_checkpoint``, the engines' base: the release with one tensor per expert under the names Miles exports
      (``tools/convert_hf_to_raw_export.py``); Stitch's ``recipes/prep.py`` records it.
    """

    model_config_class: ClassVar[type[ModelConfig]] = Muse_Spark_Open
    hf_checkpoint: str = "/checkpoints/muse-spark-open-816b-a42b-bf16-per-expert"
    ref_load: str = "/checkpoints/muse-spark-open-816b-a42b-bf16"
    miles_model_name: str = "muse-spark-full"
    model_name: str = "muse_spark"
    # Weight delta keeps the last published bytes of each exported tensor on the host: ~0.82 TB per stage, plus
    # the release's page cache while loading.
    memory: int | tuple[int, int] | None = (2048 * 1024, 2560 * 1024)

    actor_num_nodes: int = 2
    pipeline_model_parallel_size: int = 2
    context_parallel_size: int = 8
    expert_model_parallel_size: int = 8
    attention_backend: str | None = "flash"
    max_tokens_per_gpu: MicroBatchSize = 4096
    rollout_max_response_len: int = 30720

    rollout_num_gpus: int | None = 16
    rollout_num_gpus_per_engine: int = 8
    sglang_ep_size: int | None = 8
    sglang_mem_fraction_static: float = 0.93
    sglang_max_running_requests: int | None = 256
    extra_config: dict | None = field(default_factory=_extra_config)
