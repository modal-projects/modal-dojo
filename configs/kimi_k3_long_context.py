"""Shared context budgeting for the separate B300 and H200 K3 recipes."""


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


def context_settings(context_length: int, concurrency_per_engine: int) -> dict:
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
    return {
        "colocate_memory_peak_device": "cpu",
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
