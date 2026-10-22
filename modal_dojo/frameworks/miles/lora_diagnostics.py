"""Observed policy/LoRA movement, independent of reward and gradient computation.

The Dojo Miles integration captures logits during its existing pre-update forward, then repeats
that forward once after the update. Full vocabulary distributions are retained only
at a deterministic sample of assistant-loss positions, in CPU memory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from itertools import pairwise
from pathlib import Path

import torch
import torch.distributed as dist

_ACTIVE = None


def select_positions(masks, budget, seed):
    """Uniform token sample without replacement; never include masked context."""
    lengths = [len(x) for x in masks]
    valid = (
        torch.cat([torch.as_tensor(x, device="cpu").bool() for x in masks])
        .nonzero()
        .flatten()
    )
    generator = torch.Generator().manual_seed(seed)
    selected = (
        valid[torch.randperm(len(valid), generator=generator)[:budget]].sort().values
    )
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return [
        selected[(selected >= a) & (selected < b)] - a for a, b in pairwise(offsets)
    ]


def shard_log_probs(logits, vocab_size, tp_rank=0, tp_group=None):
    """Exact softmax over all real vocabulary entries, distributed across TP."""
    x = logits.float().clone()
    start = tp_rank * x.shape[-1]
    x[:, max(0, vocab_size - start) :] = -torch.inf
    maximum = x.max(-1).values
    distributed = dist.is_initialized() and tp_group is not None
    if distributed:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp_group)
    shifted = x - maximum[:, None]
    denominator = shifted.exp().sum(-1)
    if distributed:
        dist.all_reduce(denominator, group=tp_group)
    return shifted - denominator.log()[:, None]


def kl_contribution(before, after):
    # Padded entries are -inf on both sides: exclude rather than form 0 * NaN.
    finite = torch.isfinite(before)
    difference = torch.where(finite, before - after, 0)
    return (before.exp() * difference).sum(-1)


def effective_squares(base, old_a, old_b, new_a, new_b, scale, rows=128):
    """Exact dense effective-weight norms in bounded row blocks, FP64 reduction."""
    if base.ndim != 2 or (old_b.shape[0], old_a.shape[1]) != base.shape:
        raise ValueError("LoRA factors do not match the local base-weight shard")
    delta2 = torch.zeros((), dtype=torch.float64, device=base.device)
    weight2 = delta2.clone()
    adapter2 = delta2.clone()
    old_a, new_a = old_a.double(), new_a.double()
    for start in range(0, base.shape[0], rows):
        stop = start + rows
        old = (old_b[start:stop].double() @ old_a) * scale
        new = (new_b[start:stop].double() @ new_a) * scale
        delta2 += (new - old).double().square().sum()
        weight2 += (base[start:stop].double() + old).double().square().sum()
        adapter2 += old.double().square().sum()
    return torch.stack([delta2, weight2, adapter2])


def _key(tokens, mask):
    digest = hashlib.sha256(
        tokens.detach().to(device="cpu", dtype=torch.int64).numpy().tobytes()
    )
    digest.update(
        torch.as_tensor(mask, device="cpu", dtype=torch.uint8).numpy().tobytes()
    )
    return digest.hexdigest()


def _aligned(a, b, base, adapter, group=None):
    if group is None:
        group = getattr(adapter, "tp_group", getattr(adapter, "expert_tp_group", None))
    size = (
        dist.get_world_size(group) if group is not None and dist.is_initialized() else 1
    )

    def gather(x):
        parts = [torch.empty_like(x) for _ in range(size)]
        dist.all_gather(parts, x.contiguous(), group=group)
        return torch.cat(parts, dim=0)

    if size > 1 and a.shape[0] * size == adapter.dim and b.shape[1] == adapter.dim:
        a = gather(a)
    # Nonparallel base linears can still have both adapter factors TP-sharded:
    # Bridge gathers A's intermediate and B's output during their forward pass.
    # These gathers are independent, not mutually exclusive.
    if size > 1 and b.shape[0] * size == base.shape[0]:
        b = gather(b)
    if b.shape[1] != a.shape[0] or (b.shape[0], a.shape[1]) != base.shape:
        raise ValueError(
            f"Unsupported LoRA shard layout: A={tuple(a.shape)}, B={tuple(b.shape)}, "
            f"base={tuple(base.shape)}, TP={size}, adapter={type(adapter).__name__}; "
            "refusing misleading update metric"
        )
    return a, b


def _layers(models, parallel):
    covered = set()
    for chunk, model in enumerate(models):
        for name, module in model.named_modules():
            if module.__class__.__name__ in {
                "LoRALinear",
                "LoRAUnreducedRowLinear",
            } and hasattr(module, "exports"):
                exports = list(module.exports())
                hosts = list(module.weight_deltas())
                if len(exports) != 1 or len(hosts) != 1:
                    raise ValueError(
                        "Native linear diagnostics require one projection and one host"
                    )
                projection, (weight, _) = exports[0], hosts[0]
                a, b = projection.a, projection.b
                if (b.shape[0], a.shape[1]) != weight.shape or b.shape[1] != a.shape[0]:
                    raise ValueError(
                        "Native LoRA factors do not match the local host shard"
                    )
                covered.update([id(a), id(b)])
                replicas = parallel.effective_dp.size
                if not getattr(weight, "tensor_model_parallel", False):
                    replicas *= parallel.tp.size
                yield (
                    f"{chunk}:{name}:0",
                    weight,
                    a,
                    b,
                    float(module.context.lora.scale),
                    replicas,
                )
                continue
            adapter, wrapped = (
                getattr(module, "adapter", None),
                getattr(module, "to_wrap", None),
            )
            if adapter is None or wrapped is None or not hasattr(adapter, "linear_in"):
                continue
            a, b = adapter.linear_in.weight, adapter.linear_out.weight
            covered.update([id(a), id(b)])
            if adapter.__class__.__name__ == "SharedOuterGroupedExpertAdapter":
                raise ValueError("Shared-outer expert LoRA diagnostics not implemented")
            base = getattr(wrapped, "weight", None)
            bases = (
                [base]
                if base is not None
                else [
                    p
                    for key, p in sorted(
                        wrapped.named_parameters(recurse=False),
                        key=lambda item: (
                            int(item[0][6:])
                            if item[0].startswith("weight") and item[0][6:].isdigit()
                            else -1
                        ),
                    )
                    if key.startswith("weight")
                ]
            )
            if not bases or any(x.ndim != 2 for x in bases):
                raise ValueError(f"Unsupported base layout: {name}")
            if a.ndim == 3 and len(bases) != a.shape[0]:
                raise ValueError("Grouped expert count differs from base weights")
            scale = (
                float(adapter.alpha / adapter.dim)
                if hasattr(adapter, "alpha")
                else float(adapter.scale)
            )
            expert_parameter = not getattr(a, "allreduce", True)
            tensor_parallel = parallel.etp if expert_parameter else parallel.tp
            for expert, weight in enumerate(bases):
                ai, bi = (a[expert], b[expert]) if a.ndim == 3 else (a, b)
                try:
                    ai, bi = _aligned(ai, bi, weight, adapter, tensor_parallel.group)
                except ValueError as error:
                    raise ValueError(f"{name} expert={expert}: {error}") from error
                # Count unique model-weight shards, not copies on DP ranks.
                replicas = parallel.effective_dp.size
                if expert_parameter:
                    # Pinned Bridge uses global Megatron groups and exposes no
                    # adapter.expert_dp_group. CP=PP=1 is required by this probe.
                    from megatron.core import parallel_state as megatron_parallel

                    replicas = megatron_parallel.get_expert_data_parallel_world_size()
                tp_size = parallel.etp.size if expert_parameter else parallel.tp.size
                if not getattr(weight, "tensor_model_parallel", False):
                    replicas *= tp_size
                yield f"{chunk}:{name}:{expert}", weight, ai, bi, scale, replicas
    expected = {id(p) for m in models for p in m.parameters() if p.requires_grad}
    if expected != covered:
        raise ValueError(
            "Update diagnostics require all trainable parameters to be supported LoRA factors"
        )


@torch.no_grad()
def validate_layout(models):
    """Check every adapter before spending time on initial evaluation/rollouts."""
    if os.environ.get("DOJO_UPDATE_DIAGNOSTICS", "0") != "1":
        return
    from miles.backends.training_utils.parallel import get_parallel_state

    count = sum(1 for _ in _layers(models, get_parallel_state()))
    logging.getLogger(__name__).info(
        "Validated update-diagnostic layout: %s layer shards", count
    )


class UpdateProbe:
    def __init__(self, args, data, rollout_id):
        from miles.backends.training_utils.parallel import get_parallel_state

        self.parallel = get_parallel_state()
        p = self.parallel
        if p.cp.size != 1 or p.pp.size != 1 or p.indep_dp.size != 1:
            raise ValueError(
                "Update diagnostics currently require CP=PP=1 and synchronous DP"
            )
        if getattr(args, "true_on_policy_mode", False) or args.rollout_temperature != 1:
            raise ValueError(
                "Update diagnostic currently measures untempered policy (temperature=1)"
            )
        self.args, self.rollout_id = args, rollout_id
        self.started = time.monotonic()
        budget = math.ceil(
            int(os.environ.get("DOJO_UPDATE_PROBE_TOKENS", "1536"))
            / p.effective_dp.size
        )
        positions = select_positions(data["loss_masks"], budget, 1731 + rollout_id)
        self.positions = {}
        for tokens, mask, indices in zip(
            data["tokens"], data["loss_masks"], positions, strict=True
        ):
            key = _key(tokens, mask)
            # Repeated identical prefixes need one fixed set, not overwritten sets.
            self.positions[key] = torch.unique(
                torch.cat(
                    [self.positions.get(key, torch.empty(0, dtype=torch.long)), indices]
                )
            )
        self.before = {}
        self.values = {}
        self.phase = "before"
        self.weights = {}

    @torch.no_grad()
    def capture(self, logits, batch):
        try:
            from miles.backends.training_utils.loss.hub.logit_processors import (
                get_responses,
            )
        except ImportError:
            from miles.backends.training_utils.loss_hub.logit_processors import (
                get_responses,
            )

        chunks = get_responses(
            logits,
            args=self.args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens"),
        )
        p = self.parallel
        for tokens, mask, (chunk, _) in zip(
            batch["unconcat_tokens"], batch["loss_masks"], chunks, strict=True
        ):
            key = _key(tokens, mask)
            indices = self.positions[key]
            if not len(indices):
                continue
            logp = shard_log_probs(
                chunk[indices.to(chunk.device)],
                self.args.vocab_size,
                p.tp.rank,
                p.tp.group,
            )
            if self.phase == "before":
                self.before[key] = logp.cpu()
            else:
                old = self.before[key].to(logp.device)
                kl = kl_contribution(old, logp)
                if p.tp.size > 1:
                    dist.all_reduce(kl, group=p.tp.group)
                self.values[key] = kl.cpu()

    @torch.no_grad()
    def snapshot_weights(self, models):
        for name, base, a, b, scale, replicas in _layers(models, self.parallel):
            self.weights[name] = (a.detach().cpu().clone(), b.detach().cpu().clone())

    @torch.no_grad()
    def finish(self, models):
        from miles.utils.tracking_utils import tracking

        if self.before.keys() != self.values.keys() or not self.values:
            raise ValueError("Incomplete before/after KL probe coverage")
        rows, totals = [], torch.zeros(3, device="cuda", dtype=torch.float64)
        for name, base, a, b, scale, replicas in _layers(models, self.parallel):
            old_a, old_b = self.weights.pop(name)
            values = effective_squares(
                base, old_a.to(a.device), old_b.to(b.device), a, b, scale
            )
            totals += values / replicas
            d2, w2, a2 = values.tolist()
            rows.append(
                {
                    "layer_shard": name,
                    "delta_norm": math.sqrt(d2),
                    "weight_norm": math.sqrt(w2),
                    "relative_update": math.sqrt(d2 / w2) if w2 else None,
                    "relative_adapter_update": math.sqrt(d2 / a2) if a2 else None,
                }
            )
        dist.all_reduce(totals)
        local = (
            torch.cat(list(self.values.values())).tolist()
            if self.parallel.tp.rank == 0
            else []
        )
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        values = torch.tensor(
            [v for part in gathered for v in part], dtype=torch.float64
        )
        if not torch.isfinite(values).all() or not torch.isfinite(totals).all():
            raise ValueError("Non-finite update diagnostics")
        rank = dist.get_rank()
        directory = Path(self.args.save) / "update_diagnostics"
        directory.mkdir(parents=True, exist_ok=True)
        # Contains indices/hashes and numeric metrics, never conversation text.
        record = {
            "rollout_id": self.rollout_id,
            "rank": rank,
            "layers": rows,
            "probe_positions": {k: v.tolist() for k, v in self.positions.items()},
            "kl": {k: v.tolist() for k, v in self.values.items()},
        }
        (directory / f"step{self.rollout_id:07d}-rank{rank}.json").write_text(
            json.dumps(record)
        )
        if rank == 0:
            d2, w2, a2 = totals.tolist()
            metrics = {
                "train/step": self.rollout_id,
                "train/post_update_kl_mean": values.mean().item(),
                "train/post_update_kl_p95": values.quantile(0.95).item(),
                "train/post_update_kl_positions": len(values),
                "train/effective_lora_delta_norm": math.sqrt(d2),
                "train/effective_weight_relative_update": math.sqrt(d2 / w2),
                "train/update_diagnostics_window_seconds": time.monotonic()
                - self.started,
            }
            if a2:
                metrics["train/effective_adapter_relative_update"] = math.sqrt(d2 / a2)
            tracking.log(self.args, metrics, step_key="train/step")


def begin(args, data, rollout_id):
    global _ACTIVE
    _ACTIVE = None
    if os.environ.get("DOJO_UPDATE_DIAGNOSTICS", "0") == "1":
        if (
            getattr(args, "skip_train", False)
            or getattr(args, "skip_actor_forward_only", False)
            or not args.compute_advantages_and_returns
            or args.keep_old_actor
            or (args.use_rollout_logprobs and not args.get_mismatch_metrics)
        ):
            raise ValueError("Diagnostics require the pre-update training forward")
        _ACTIVE = UpdateProbe(args, data, rollout_id)
    return _ACTIVE


def capture(logits, batch):
    if _ACTIVE is not None:
        _ACTIVE.capture(logits, batch)


def end():
    global _ACTIVE
    _ACTIVE = None


def retain_replay_data(data, keys):
    """Keep references before Miles fill_replay_data deletes these dictionary keys."""
    if os.environ.get("DOJO_UPDATE_DIAGNOSTICS", "0") != "1":
        return {}
    return {key: data[key] for key in keys}
