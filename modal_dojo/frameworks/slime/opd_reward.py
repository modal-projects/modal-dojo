"""Drop-in OPD reward and post-process functions that support extra SGLang params.

These are functionally identical to slime's built-in
``slime.rollout.on_policy_distillation.reward_func`` and
``post_process_rewards``, but route the HTTP call through
:func:`modal_dojo.common.sglang_client.generate` so
that arbitrary SGLang ``/generate`` parameters (e.g. ``top_logprobs_num``)
can be injected via ``args.sglang_request_params``.

When ``sglang_request_params`` is not set on the recipe (or is empty),
the payload is identical to slime's built-in.
"""

from __future__ import annotations

from typing import Any

import torch

from modal_dojo.common.sglang_client import (
    generate as sglang_generate,
)


def _encode_images(sample: Any) -> list[Any] | None:
    """Encode multimodal image inputs the same way slime's built-in does."""
    if not getattr(sample, "multimodal_inputs", None):
        return None
    images = sample.multimodal_inputs.get("images")
    if not images:
        return None
    from slime.utils.processing_utils import encode_image_for_rollout_engine

    return [encode_image_for_rollout_engine(img) for img in images]


async def reward_func(args: Any, sample: Any, **kwargs: Any) -> dict[str, Any]:
    """Collect teacher log-probs via SGLang ``/generate``.

    Reads ``args.sglang_request_params`` (populated from ``extra_config``
    YAML) and merges those params into the request payload, allowing
    callers to request ``top_logprobs_num``, ``return_text_in_logprobs``,
    etc. without modifying this function.
    """
    extra: dict[str, Any] = getattr(args, "sglang_request_params", None) or {}

    return await sglang_generate(
        args.rm_url,
        input_ids=sample.tokens,
        sampling_params={
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        return_logprob=True,
        logprob_start_len=0,
        image_data=_encode_images(sample),
        **extra,
    )


def post_process_rewards(
    args: Any,
    samples: list[Any],
    **kwargs: Any,
) -> tuple[list[float], list[float]]:
    """Extract teacher log-probs and return scalar rewards.

    Mirrors ``slime.rollout.on_policy_distillation.post_process_rewards``:

    1. Extracts per-token teacher log-probs from the SGLang response.
    2. Trims them to the response span.
    3. Stores them as ``sample.teacher_log_probs`` for the OPD KL penalty.
    4. Returns ``0.0`` scalar rewards (pure distillation — the learning
       signal comes from the KL penalty, not a task reward).
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    teacher_log_probs = [
        torch.tensor(
            [item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]],
            dtype=torch.float32,
        )
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(
            teacher_log_probs, response_lengths, strict=False
        )
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
