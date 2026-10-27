"""Opt-in full-window Beam stress generation with measured-token assertions."""

from functools import lru_cache


@lru_cache(maxsize=32)
def pad_prompt(tokenizer, prompt: str, target_tokens: int) -> str:
    """Insert synthetic padding inside the user message, preserving chat boundaries."""
    prefix, marker, suffix = prompt.partition("<|start|>user<|message|>")
    if not marker:
        raise ValueError("Beam stress prompt must contain a rendered user message")
    head = prefix + marker + "Synthetic throughput stress context (padding follows):\n"
    tail = "\nEnd of padding.\n" + suffix
    repeats = target_tokens - len(
        tokenizer.encode(head + tail, add_special_tokens=False)
    )
    for _ in range(4):
        if repeats < 0:
            raise ValueError("Stress prompt exceeds the requested input budget")
        padded = head + " x" * repeats + tail
        count = len(tokenizer.encode(padded, add_special_tokens=False))
        if count == target_tokens:
            return padded
        repeats += target_tokens - count
    raise ValueError("Could not construct an exact-length Beam stress prompt")


def validate_sample(sample, prompt_tokens: int, response_tokens: int) -> dict:
    """Reject short/aborted/missing-logprob outputs before they can train or score."""
    actual_response = sample.response_length
    actual_total = len(sample.tokens or [])
    actual_prompt = actual_total - actual_response
    status = getattr(sample.status, "name", str(sample.status)).upper()
    if status not in {"COMPLETED", "TRUNCATED"}:
        raise ValueError(f"Incomplete Beam stress sample: {status}")
    if (actual_prompt, actual_response) != (prompt_tokens, response_tokens):
        raise ValueError(
            f"Beam stress length mismatch: input={actual_prompt}, output={actual_response}; "
            f"expected {prompt_tokens}+{response_tokens}"
        )
    if len(sample.rollout_log_probs or []) != response_tokens:
        raise ValueError(
            "Beam stress sample is missing generated-token log probabilities"
        )
    return {
        "benchmark": "beam-full-window-forced",
        "sample_index": getattr(sample, "index", None),
        "prompt_tokens": actual_prompt,
        "generated_tokens": actual_response,
        "total_tokens": actual_total,
        "status": status,
        "ignore_eos": True,
    }


async def force_full_window(input):
    """Generate every token genuinely; never pad, copy or fabricate model outputs."""
    import json
    from dataclasses import replace

    from miles.rollout.generate_hub.single_turn import generate

    if input.evaluation:
        return await generate(input)
    args, sample = input.args, input.sample
    if getattr(args, "partial_rollout", False) or getattr(args, "fully_async", False):
        raise ValueError(
            "Full-window stress requires complete, non-partial rollout batches"
        )
    if sample.response_length or sample.tokens:
        raise ValueError("Full-window stress requires a fresh sample")
    response_tokens = args.rollout_max_response_len
    prompt_tokens = args.rollout_max_context_len - response_tokens
    if (prompt_tokens, response_tokens) != (229376, 32768):
        raise ValueError("Beam full-window stress requires 224k input + 32k output")
    sample.prompt = pad_prompt(input.state.tokenizer, sample.prompt, prompt_tokens)
    params = {
        **input.sampling_params,
        "ignore_eos": True,
        "min_new_tokens": response_tokens,
        "max_new_tokens": response_tokens,
        "stop": [],
        "stop_token_ids": [],
    }
    output = await generate(replace(input, sampling_params=params))
    report = validate_sample(output.samples, prompt_tokens, response_tokens)
    output.samples.metadata = {**(output.samples.metadata or {}), **report}
    print("BEAM_FULL_WINDOW_SAMPLE " + json.dumps(report), flush=True)
    return output


def audit_full_batch(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """Check the entire batch, not just the dashboard's sample preview."""
    import json

    expected = args.rollout_batch_size * args.n_samples_per_prompt
    if len(samples) != expected:
        raise ValueError(
            f"Incomplete Beam stress batch: {len(samples)} of {expected} samples"
        )
    indices = [sample.index for sample in samples]
    if None in indices or len(set(indices)) != expected:
        raise ValueError("Beam stress batch must contain distinct sample indices")
    reports = [validate_sample(s, 229376, 32768) for s in samples]
    print(
        "BEAM_FULL_WINDOW_BATCH "
        + json.dumps(
            {
                "rollout_id": rollout_id,
                "samples": expected,
                "generated_tokens": sum(r["generated_tokens"] for r in reports),
                "prompt_tokens": sum(r["prompt_tokens"] for r in reports),
                "all_full_window": True,
            }
        ),
        flush=True,
    )
    return False  # Keep the framework's normal metrics logging enabled.
