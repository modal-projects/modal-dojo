"""Kimi-K3 LoRA at 64k context on eight B300:8 nodes."""

from collections.abc import Callable
from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.train_recipes.miles_recipe.kimi_k3 import Kimi_K3_LoRA_Recipe


async def generate_with_context_limit(input):
    """Bound single-turn generation and resumed samples by remaining context."""
    from miles.rollout.base_types import GenerateFnOutput
    from miles.rollout.sglang_rollout import generate
    from miles.utils.types import Sample

    sample = input.sample
    context_length = input.args.sglang_context_length
    prompt_length = len(
        input.state.tokenizer.encode(sample.prompt, add_special_tokens=False)
    )
    if prompt_length >= context_length - 1:
        raise ValueError("Prompt leaves no generation space in the K3 context window")
    params = dict(input.sampling_params)
    # Reserve the pinned scheduler's one-token boundary margin. Miles subtracts
    # previously generated tokens itself when resuming an aborted sample.
    params["max_new_tokens"] = min(
        params["max_new_tokens"], context_length - prompt_length - 1
    )
    if (
        sample.response
        and len(sample.tokens) - prompt_length >= params["max_new_tokens"]
    ):
        sample.status = Sample.Status.TRUNCATED
        return GenerateFnOutput(samples=sample)
    sample = await generate(input.args, sample, params, evaluation=input.evaluation)
    return GenerateFnOutput(samples=sample)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Kimi_K3_Long_Context_Recipe(Kimi_K3_LoRA_Recipe):
    """Synchronous 64k LoRA preset; inherits the base K3 image and patches.

    Validated with two updates over near-64k sequences containing masked
    synthetic observations. Custom agents must bound the complete trajectory
    and mask tool observations themselves.
    """

    # Offload the active model before restoring the other to avoid GPU overlap.
    colocate_memory_peak_device: str = "cpu"
    memory: tuple[int, int] = (2560 * 1024, 3072 * 1024)

    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    global_batch_size: int = 8
    rollout_max_response_len: int = 65536
    eval_max_response_len: int = 65536
    custom_generate_function: Callable | None = generate_with_context_limit

    seq_length: int = 65536
    max_position_embeddings: int = 65536
    # CP2 packs one full 64k sequence into a 32k per-GPU token budget.
    max_tokens_per_gpu: int = 32768
    log_probs_max_tokens_per_gpu: int = 32768

    sglang_context_length: int = 65536
    sglang_chunked_prefill_size: int = 4096
    sglang_server_concurrency: int | None = 1
    sglang_max_running_requests: int | None = 1
    sglang_max_mamba_cache_size: int = 5
    # One active sequence plus one context window of cache headroom.
    sglang_max_total_tokens: int = 131072
    sglang_cuda_graph_bs_decode: list[int] | None = field(default_factory=lambda: [1])

    # A YAML null disables Miles' initial prompt filter; None CLI fields would
    # be omitted and leave the upstream default in effect.
    extra_config: dict | None = field(
        default_factory=lambda: {"rollout_max_prompt_len": None}
    )
