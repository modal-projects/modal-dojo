"""Experimental 256k Beam LoRA; full-model long-context validation is pending."""

from typing import ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.train_recipes.base import MicroBatchSize
from .beam import Beam_1_0_501B_A23B_LoRA_Recipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Beam_1_0_501B_A23B_LoRA_256k_Recipe(Beam_1_0_501B_A23B_LoRA_Recipe):
    """256k total context on 16 training + 16 rollout H200s (unvalidated).

    Uses the requested 32k response budget; preserves batch size and optimizer.
    FlashAttention avoids quadratic score storage; CP2 shards training sequences.
    A 5k DAPO proof does not validate this preset at the advertised context length.
    """

    _BEAM_ATTENTION_BACKENDS: ClassVar[tuple[str, ...]] = ("flash",)
    attention_backend: str = "flash"
    context_parallel_size: int = 2
    cp_comm_type: str = "all_gather"
    rollout_max_context_len: int = 262144
    rollout_max_prompt_len: int = 229376
    rollout_max_response_len: int = 32768
    eval_max_response_len: int = 32768
    seq_length: int = 262144
    max_tokens_per_gpu: MicroBatchSize = 262144
    # The pinned worker subtracts one slot, then the scheduler reserves another.
    # Keep the actual prompt+response at 262144 while leaving both internal slots.
    sglang_context_length: int = 262146
    # Limit concurrent long KV caches, without changing samples per optimizer step.
    sglang_max_running_requests: int | None = 4
    sglang_cuda_graph_max_bs_decode: int = 4
    sglang_chunked_prefill_size: int = 4096

    @model_validator(mode="after")
    def _check_long_context(self) -> "Beam_1_0_501B_A23B_LoRA_256k_Recipe":
        if self.cp_comm_type != "all_gather":
            raise ValueError(
                "Beam sliding-window CP requires all_gather in this preset"
            )
        if self.context_parallel_size < 2:
            raise ValueError("The 256k Beam preset requires context_parallel_size >= 2")
        if (
            self.rollout_max_prompt_len + self.rollout_max_response_len
            > self.rollout_max_context_len
        ):
            raise ValueError(
                "Prompt and response budgets exceed the Beam context window"
            )
        if (
            self.seq_length < self.rollout_max_context_len
            or self.sglang_context_length < self.rollout_max_context_len + 2
        ):
            raise ValueError(
                "Trainer and rollout engine must cover the Beam context window"
            )
        return self
