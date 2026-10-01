"""Exercise the K3 long-context configuration on the full 64-B300 model.

Run with ``uv run -m scripts.validate_kimi_k3_long_context --mode capacity``.
Defaults to two updates at 64k context; pass ``--launch`` to allocate GPUs.
Capacity mode uses a large masked observation between two real generations.
It tests long-sequence training, not long autonomous assistant generation.
Decode mode generates a long assistant response from a short task prompt; the
minimum-token constraint is a stress test, not a production sampling setting.
Math mode runs ordinary DAPO rollouts with the configured response ceiling.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def generation_budget(context_length, current_length, response_limit):
    return max(0, min(response_limit, context_length - current_length - 1))


def append_tokens(sample, tokens, logprobs, *, generated):
    if len(tokens) != len(logprobs):
        raise ValueError("Every response token must have one logprob entry")
    if any(not math.isfinite(p) for p in logprobs):
        raise ValueError("Non-finite rollout logprob")
    sample.tokens.extend(tokens)
    sample.response_length += len(tokens)
    sample.rollout_log_probs.extend(logprobs)
    sample.loss_mask.extend([int(generated)] * len(tokens))


def observation_prefix(rendered_suffix, generated_text):
    """Close a truncated K3 message using masked environment boundary tokens."""
    user_start = '<|open|>message role="user"'
    closing, user = rendered_suffix.split(user_start, 1)
    if "<|end_of_msg|>" in generated_text:
        closing = ""
    elif "<|close|>message" in generated_text:
        closing = "<|end_of_msg|>"
    elif "<|close|>response" in generated_text:
        closing = "<|close|>message<|sep|><|end_of_msg|>"
    elif "<|close|>think" not in generated_text:
        closing = "<|close|>think<|sep|><|open|>response<|sep|>" + closing
    return closing + user_start + user


async def generate_probe(input):
    from miles.rollout.base_types import GenerateFnOutput
    from miles.rollout.generate_utils.generate_endpoint_utils import (
        compute_routing_headers,
    )
    from miles.utils.http_utils import post
    from miles.utils.lora.utils import LORA_ADAPTER_NAME
    from miles.utils.types import Sample

    args, sample, tok = input.args, input.sample, input.state.tokenizer
    limit = args.sglang_context_length
    mode = args.k3_validation_mode
    sample.tokens = tok.encode(sample.prompt, add_special_tokens=False)
    prompt_len = len(sample.tokens)
    sample.response_length = 0
    sample.rollout_log_probs = []
    sample.loss_mask = []
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    headers = compute_routing_headers(args, sample)
    turns = []

    async def generate_turn(requested, *, force_length=False):
        budget = generation_budget(limit, len(sample.tokens), requested)
        if not budget:
            raise ValueError("No generation space remains")
        params = dict(input.sampling_params)
        params["max_new_tokens"] = budget
        if force_length:
            params["min_new_tokens"] = budget
            params["ignore_eos"] = True
        output = await post(
            url,
            {
                "input_ids": sample.tokens,
                "sampling_params": params,
                "return_logprob": True,
                "lora_path": LORA_ADAPTER_NAME,
            },
            headers=headers,
        )
        info = output["meta_info"]
        entries = info.get("output_token_logprobs") or []
        if not entries:
            raise ValueError(f"No generated tokens: {info}")
        tokens = [entry[1] for entry in entries]
        probs = [entry[0] for entry in entries]
        append_tokens(sample, tokens, probs, generated=True)
        sample.update_from_meta_info(args, info)
        turns.append(
            {
                "generated_tokens": len(tokens),
                "finish_reason": info.get("finish_reason"),
            }
        )
        print(
            "K3_CONTEXT_TURN "
            + json.dumps(
                {"index": sample.index, "total_tokens": len(sample.tokens), **turns[-1]}
            ),
            flush=True,
        )
        return output["text"]

    if mode in ("decode", "math"):
        requested = (
            args.k3_decode_tokens
            if mode == "decode"
            else input.sampling_params["max_new_tokens"]
        )
        final_response = await generate_turn(requested, force_length=mode == "decode")
    else:
        first_response = await generate_turn(256)
        # Never re-tokenize the generated prefix. A truncated thinking channel
        # must be closed before inserting the synthetic environment message.
        marker = "K3_ASSISTANT_BOUNDARY_6d1982"
        pre = tok.apply_chat_template(
            [
                {"role": "user", "content": "Earlier task"},
                {"role": "assistant", "content": marker},
                {"role": "user", "content": "OBSERVATION_START"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        suffix = observation_prefix(pre.split(marker, 1)[1], first_response)
        prefix_text, tail_text = suffix.split("OBSERVATION_START", 1)
        prefix = tok.encode(
            prefix_text + "Diagnostic observation follows.\n", add_special_tokens=False
        )
        user_marker = '<|open|>message role="user"<|sep|>'
        original_problem = sample.prompt.split(user_marker, 1)[1].split(
            "<|close|>message", 1
        )[0]
        tail = tok.encode(
            "\nEnd diagnostic observation. Original problem, repeated for reference:\n"
            + original_problem
            + "\nSolve this problem and give the final answer in \\boxed{}."
            + tail_text,
            add_special_tokens=False,
        )
        # Keep every trajectory within 1024 tokens of the selected context.
        # Observation is synthetic input, never labeled as generated output.
        fill_count = limit - 1024 - len(sample.tokens) - len(prefix) - len(tail)
        if fill_count < 0:
            raise ValueError("Prompt leaves insufficient capacity-probe space")
        unit = tok.encode(
            "Diagnostic record: all checks completed.\n", add_special_tokens=False
        )
        filler = (unit * ((fill_count + len(unit) - 1) // len(unit)))[:fill_count]
        observation = prefix + filler + tail
        append_tokens(sample, observation, [0.0] * len(observation), generated=False)
        final_response = await generate_turn(1023)

    sample.response = tok.decode(sample.tokens[prompt_len:], skip_special_tokens=False)
    sample.metadata = dict(sample.metadata or {})
    sample.metadata.update(
        {
            "k3_validation_mode": mode,
            "k3_final_response": final_response,
            "k3_total_tokens": len(sample.tokens),
            "k3_generated_tokens": sum(sample.loss_mask),
            "k3_observation_tokens": len(sample.loss_mask) - sum(sample.loss_mask),
            "k3_turns": turns,
        }
    )
    assert len(sample.tokens) <= limit
    assert len(sample.tokens) - prompt_len == sample.response_length
    assert (
        sample.response_length == len(sample.loss_mask) == len(sample.rollout_log_probs)
    )
    if mode == "capacity":
        assert len(sample.tokens) >= limit - 1024
    if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
        raise RuntimeError(f"Unexpected final sample status: {sample.status}")
    print(
        "K3_CONTEXT_SAMPLE "
        + json.dumps(
            {
                k: v
                for k, v in sample.metadata.items()
                if k.startswith("k3_") and k != "k3_final_response"
            }
        ),
        flush=True,
    )
    return GenerateFnOutput(samples=sample)


async def score_final_response(args, sample, **kwargs):
    from miles.rollout.rm_hub import get_deepscaler_rule_based_reward

    # Only the last real generation is an answer, never the observation/prompt.
    response = (sample.metadata or {}).get("k3_final_response", "")
    return get_deepscaler_rule_based_reward(response, sample.label)


def report_memory(stage):
    from pathlib import Path

    import psutil
    import torch

    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    host = {"process_rss_gib": psutil.Process().memory_info().rss / 2**30}
    for name, paths in {
        "host_used_gib": (
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
        "host_peak_gib": (
            "/sys/fs/cgroup/memory.peak",
            "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
        ),
        "host_limit_gib": (
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ),
    }.items():
        for path in paths:
            try:
                host[name] = int(Path(path).read_text().strip()) / 2**30
                break
            except (OSError, ValueError):
                continue
    print(
        "K3_CONTEXT_MEMORY "
        + json.dumps(
            {
                "stage": stage,
                "rank": torch.distributed.get_rank(),
                "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                "device_free_gib": free / 2**30,
                "device_total_gib": total / 2**30,
                **host,
            }
        ),
        flush=True,
    )


def before_log_prob(args, model, store_prefix):
    report_memory("before_log_prob:" + store_prefix)


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    report_memory(f"before_train:{rollout_id}:{step_id}")


def build_config(
    mode="capacity", context_length=65536, rollouts=2, decode_tokens=57344
):
    from configs.kimi_k3_long_context import build_recipe
    from modal_dojo import HuggingFaceDataset, Kimi_K3, TrainConfig

    recipe = build_recipe(
        context_length=context_length,
        num_rollout=rollouts,
        rm_type="deepscaler",
        # Even ordinary math must subtract the prompt from the response
        # ceiling: the pinned builtin generator does not cap total context.
        custom_generate_function=generate_probe,
        custom_rm_function=score_final_response,
        custom_megatron_before_log_prob_hook=before_log_prob,
        custom_megatron_before_train_step_hook=before_train_step,
        capture_trace=True,
        trace_sample_limit=4,
        save_interval=1,
        max_retries=0,
        # Capacity mode measures sequence memory; keep the final answer short
        # enough to score within its reserved 1023-token completion window.
        apply_chat_template_kwargs={"thinking_effort": "low"}
        if mode == "capacity"
        else {},
        extra_config={
            "k3_validation_mode": mode,
            "k3_decode_tokens": decode_tokens,
        },
    )
    return TrainConfig(
        model=Kimi_K3(),
        recipe=recipe,
        dataset=HuggingFaceDataset(
            "zhuzilin/dapo-math-17k",
            hf_split=f"train[:{2 * rollouts}]",
            input_column="prompt",
            output_column="label",
            input_format="messages",
            always_download=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["capacity", "decode", "math"], default="capacity"
    )
    parser.add_argument(
        "--context-length", type=int, choices=[65536, 131072], default=65536
    )
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=57344)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    config = build_config(
        args.mode, args.context_length, args.rollouts, args.decode_tokens
    )
    print(config.recipe.gpu_allocation.summary())
    if args.launch:
        run = config.launch()
        record = {
            "run_id": run.training_run_id,
            "mode": args.mode,
            "context_length": args.context_length,
            "rollouts": args.rollouts,
            "decode_tokens": args.decode_tokens,
        }
        dest = Path(".gym/new_models/Kimi_K3_Long_Context")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{run.training_run_id}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(json.dumps(record), flush=True)
    else:
        print(json.dumps(config.recipe.extra_config, indent=2))


if __name__ == "__main__":
    main()
