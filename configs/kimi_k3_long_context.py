"""64k Kimi-K3 configuration on the existing 8 x B300:8 allocation.

Use ``build_recipe()`` with TrainConfig and your agent's dataset/reward. The
agent must cap the *whole* trajectory (including tool observations), preserve
generated token IDs/logprobs, and mask observations out of the loss. A response
ceiling alone does not implement that contract.

This is an explicit workload configuration, not a new model/default recipe.
It retains the base recipe's image patches and checkpoint-volume identity.
"""

from typing import Any

from modal_dojo import Kimi_K3_LoRA_Recipe


def remaining_response_tokens(
    context_length: int, current_length: int, response_limit: int
) -> int:
    """Reserve the one token the pinned SGLang scheduler keeps at the boundary."""
    if context_length < 2 or current_length < 0 or response_limit < 0:
        raise ValueError("Invalid context, current sequence length, or response limit")
    return max(0, min(response_limit, context_length - current_length - 1))


async def generate_with_context_limit(input):
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
    params["max_new_tokens"] = remaining_response_tokens(
        context_length, prompt_length, params["max_new_tokens"]
    )
    if (
        sample.response
        and len(sample.tokens) - prompt_length >= params["max_new_tokens"]
    ):
        sample.status = Sample.Status.TRUNCATED
        return GenerateFnOutput(samples=sample)
    # Miles subtracts already generated tokens when resuming an aborted sample.
    sample = await generate(input.args, sample, params, evaluation=input.evaluation)
    return GenerateFnOutput(samples=sample)


def build_recipe(
    *,
    context_length: int = 65536,
    concurrency_per_engine: int = 1,
    **overrides: Any,
) -> Kimi_K3_LoRA_Recipe:
    """Build a synchronous long-context recipe; overrides select task/horizon.

    CP2 packs up to ``2 * max_tokens_per_gpu`` tokens per microbatch. One
    full-length sample therefore needs a budget of context_length / 2, not
    context_length. This is a packing target, not an OOM guarantee.
    """
    if context_length not in (65536, 131072):
        raise ValueError("Choose a target shape: 65536 or 131072 tokens")
    if concurrency_per_engine not in (1, 2, 4, 8):
        raise ValueError("concurrency_per_engine must be one of 1, 2, 4, 8")
    extra = {
        "rollout_max_prompt_len": None,
        "sglang_context_length": context_length,
        "sglang_chunked_prefill_size": 4096,
        "seq_length": context_length,
        # Miles derives this before applying YAML overrides, so changing only
        # seq_length leaves the 4096 placeholder and fails Megatron validation.
        "max_position_embeddings": context_length,
        "log_probs_max_tokens_per_gpu": context_length // 2,
    }
    extra.update(overrides.pop("extra_config", None) or {})
    settings = {
        "colocate_memory_peak_device": "cpu",
        "memory": (2560 * 1024, 3072 * 1024),
        "custom_generate_function": generate_with_context_limit,
        "rollout_max_response_len": context_length,
        "eval_max_response_len": context_length,
        "max_tokens_per_gpu": context_length // 2,
        # Two prompts x four completions is the initial capacity-proof batch.
        "rollout_batch_size": 2,
        "n_samples_per_prompt": 4,
        "global_batch_size": 8,
        "sglang_server_concurrency": concurrency_per_engine,
        "sglang_max_running_requests": concurrency_per_engine,
        "sglang_max_mamba_cache_size": 5 * concurrency_per_engine,
        # One additional context of cache headroom; this is per engine, not
        # the per-request context limit. Verify actual allocation at startup.
        "sglang_max_total_tokens": (concurrency_per_engine + 1) * context_length,
        "sglang_cuda_graph_bs_decode": [
            n for n in (1, 2, 4, 8) if n <= concurrency_per_engine
        ],
        "extra_config": extra,
    }
    settings.update(overrides)
    recipe = Kimi_K3_LoRA_Recipe(**settings)
    if recipe.context_parallel_size != 2:
        raise ValueError("Recalculate the packing budget before changing CP2")
    return recipe
