"""Torch / Megatron advantage-distribution math for slime's dashboard
reporting.

Split out of :mod:`.phase_reporting` (which re-exports these). The pure payload
builder (:func:`_advantage_samples_payload`, shared with miles) lives in
:mod:`modal_dojo.common.reporting`;
:func:`report_advantage_distribution` lazily imports torch / megatron so this
module stays importable outside the training container.
"""

from __future__ import annotations

import time

from modal_dojo.common.reporting import (
    _advantage_samples_payload as _advantage_samples_payload,
    _arg_value,
    _enqueue_advantage,
    _positive_int,
    _run_context,
)


def report_advantage_distribution(
    rollout_id: int,
    args: object,
    rollout_data: object,
) -> None:
    """Emit per-sample advantages (tagged with their GRPO group) for one step.

    ``args`` is slime's argparse namespace and ``rollout_data`` its per-step
    dict of tensors — neither type is importable outside the training
    container, so both are taken as ``object`` and duck-typed.

    Injected into slime's ``log_rollout_data`` so it fires right after
    ``compute_advantages_and_returns``. slime itself only logs the *mean*
    advantage per step; this captures the full per-sample distribution.

    Runs on every actor rank but only the TP-rank-0 / last-PP-stage ranks hold
    the reduced advantages, and within those only CP-rank-0 posts (after a CP
    all-reduce makes each sample's mean cover its full response). Each surviving
    rank covers its own data-parallel shard of the step's samples; the dashboard
    merges shards into per-group distributions.
    """
    if not isinstance(rollout_data, dict):
        return
    try:
        import torch
        import torch.distributed as dist
        from megatron.core import mpu
    except Exception:
        return

    try:
        if not (
            mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage()
        ):
            return
    except Exception:
        return

    advantages = rollout_data.get("advantages")
    loss_masks = rollout_data.get("loss_masks")
    response_lengths = rollout_data.get("response_lengths")
    total_lengths = rollout_data.get("total_lengths")
    if not advantages or loss_masks is None:
        return
    if response_lengths is None or total_lengths is None:
        return

    n = len(advantages)
    try:
        device = advantages[0].device
        sums = torch.zeros(n, dtype=torch.float64, device=device)
        counts = torch.zeros(n, dtype=torch.float64, device=device)
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()

        if cp_size == 1:
            for i in range(n):
                adv = advantages[i].to(torch.float64)
                mask = loss_masks[i].to(torch.float64)
                m = min(adv.numel(), mask.numel())
                sums[i] = (adv[:m] * mask[:m]).sum()
                counts[i] = mask[:m].sum()
        else:
            from slime.backends.megatron_utils.cp_utils import (
                get_logits_and_tokens_offset_with_cp,
            )

            for i in range(n):
                total_len = int(total_lengths[i])
                resp_len = int(response_lengths[i])
                prompt_len = total_len - resp_len
                _, _, _, toff = get_logits_and_tokens_offset_with_cp(
                    total_len, resp_len
                )
                mask = loss_masks[i]
                m0 = mask[toff[0][0] - prompt_len : toff[0][1] - prompt_len]
                m1 = mask[toff[1][0] - prompt_len : toff[1][1] - prompt_len]
                chunked = torch.cat([m0, m1]).to(torch.float64)
                adv = advantages[i].to(torch.float64)
                m = min(adv.numel(), chunked.numel())
                sums[i] = (adv[:m] * chunked[:m]).sum()
                counts[i] = chunked[:m].sum()
            # Every CP rank holds a token-shard of the same samples; reduce so
            # each sample's mean is taken over its full response.
            cp_group = mpu.get_context_parallel_group()
            dist.all_reduce(sums, group=cp_group)
            dist.all_reduce(counts, group=cp_group)
    except Exception:
        return

    if cp_rank != 0:
        return

    raw_rewards = list(rollout_data.get("raw_reward") or [])
    sample_indices = list(rollout_data.get("sample_indices") or range(n))
    n_per = _positive_int(_arg_value(args, "n_samples_per_prompt")) or 1

    samples = _advantage_samples_payload(
        sums.tolist(),
        counts.tolist(),
        sample_indices,
        raw_rewards,
        n_per,
    )
    if not samples:
        return

    try:
        dp_rank = int(mpu.get_data_parallel_rank(with_context_parallel=False))
    except Exception:
        dp_rank = 0

    _enqueue_advantage(
        {
            **_run_context(args),
            "rollout_id": int(rollout_id),
            "created_at": int(time.time()),
            "dp_rank": dp_rank,
            "n_samples_per_prompt": int(n_per),
            "samples": samples,
        }
    )
