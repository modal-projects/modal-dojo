"""Memory-bounded Inkling 2 training on the image pinned by its recipe.

The model retains the upstream parameters, kernels, rounding and policy loss.
Only the lifetime/size of intermediates changes. Imported on training workers;
torch and Miles are deliberately not local package dependencies.
"""

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache, partial
import math
from types import MethodType


def stream_token_scores(
    hidden_states,
    input_ids,
    project,
    scale_logits,
    *,
    temperature=1.0,
    chunk_size=256,
    vocab_size=None,
):
    """[T, 1, H] -> [1, T, 2] next-token logprob/entropy, without [T, V].

    Checkpoint projection *and* normalization together, so backward also holds
    at most one vocabulary chunk. The final row predicts outside this packed
    batch; Miles' response layout never selects it.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
        raise ValueError("Inkling streaming scores require packed thd inputs")
    if tuple(input_ids.shape) != (1, hidden_states.shape[0]):
        raise ValueError("Input tokens must align with decoder hidden states")
    if hidden_states.shape[0] < 2:
        raise ValueError("A training sample needs a prompt and a response token")
    if chunk_size <= 0 or temperature <= 0:
        raise ValueError("Chunk size and temperature must be positive")

    def score(hidden, targets):
        logits = scale_logits(project(hidden))[:, 0, :vocab_size].float() / temperature
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs.gather(-1, targets[:, None]).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        return torch.stack((selected, entropy), dim=-1)

    chunks = []
    for start in range(0, hidden_states.shape[0] - 1, chunk_size):
        end = min(start + chunk_size, hidden_states.shape[0] - 1)
        inputs = (hidden_states[start:end], input_ids[0, start + 1 : end + 1])
        if torch.is_grad_enabled():
            chunks.append(checkpoint(score, *inputs, use_reentrant=False))
        else:
            chunks.append(score(*inputs))
    final = hidden_states[-1:, 0, :1].float().expand(-1, 2) * 0
    result = torch.cat([*chunks, final], dim=0).unsqueeze(0)
    result._modal_dojo_inkling2_scores = True
    return result


def output_processor(
    *,
    hidden_states,
    input_ids,
    output_layer,
    output_weight,
    scale_logits,
    temperature,
    config,
    labels=None,
    runtime_gather_output=None,
    **kwargs,
):
    if labels is not None or runtime_gather_output:
        raise ValueError("Inkling streaming scores are for RL training only")

    def project(hidden):
        # Keep the native LoRA-wrapped head, including its fp32 projection.
        logits, _ = output_layer(
            hidden, weight=output_weight, runtime_gather_output=False
        )
        return logits

    return stream_token_scores(
        hidden_states,
        input_ids,
        project,
        scale_logits,
        temperature=temperature,
        vocab_size=config.inkling2.unpadded_vocab_size,
    )


def select_token_scores(
    scores,
    *,
    args,
    unconcat_tokens,
    total_lengths,
    response_lengths,
    with_entropy=False,
    entropy_requires_grad=True,
    max_seq_lens=None,
    rollout_sampling_mask=None,
):
    """Use Miles' own response offsets and keep its policy loss/metrics intact."""
    import torch
    from miles.backends.training_utils.loss.hub.logit_processors import (
        _local_response_layout,
    )

    if rollout_sampling_mask is not None or args.true_on_policy_mode:
        raise ValueError("Inkling streaming scores do not support sampling-mask replay")
    flat, layout = _local_response_layout(
        scores,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        include_response_indices=False,
    )
    result = {"log_probs": []}
    if with_entropy:
        result["entropy"] = []
    for ranges, _tokens, _positions in layout:
        rows = torch.cat([flat[start:end] for start, end in ranges], dim=0)
        result["log_probs"].append(rows[:, 0])
        if with_entropy:
            entropy = rows[:, 1]
            result["entropy"].append(
                entropy if entropy_requires_grad else entropy.detach()
            )
    return result


def chunked_residual_mix(
    original,
    sources,
    queries,
    *,
    normalize_values,
    partial_addend=None,
    chunk_size=1024,
):
    """Bound fp32 depth-mixing intermediates; never split sequence attention."""
    import torch
    from torch.utils.checkpoint import checkpoint

    def mix(*tensors):
        return tuple(
            original(
                list(tensors[:-2]),
                tensors[-2],
                normalize_values=normalize_values,
                partial_addend=tensors[-1],
            )
        )

    outputs = []
    for start in range(0, sources[0].shape[0], chunk_size):
        end = start + chunk_size
        tensors = (
            *(source[start:end] for source in sources),
            queries,
            None if partial_addend is None else partial_addend[start:end],
        )
        chunk = (
            checkpoint(mix, *tensors, use_reentrant=False)
            if torch.is_grad_enabled()
            else mix(*tensors)
        )
        outputs.append(chunk)
    return [torch.cat(parts, dim=0) for parts in zip(*outputs, strict=True)]


def attn_res_mix(sources, queries, *, normalize_values, partial_addend=None):
    from miles_plugins.models.inkling2.ops import attn_res_mix as original

    return chunked_residual_mix(
        original,
        sources,
        queries,
        normalize_values=normalize_values,
        partial_addend=partial_addend,
    )


@lru_cache(maxsize=1)
def _flex_compiled():
    import torch
    from torch.nn.attention.flex_attention import flex_attention

    # The pinned image's dynamic flex compiler generates an invalid launcher
    # for unequal query/key lengths. Fixed chunk shapes avoid that path.
    # The full-model backward hit the default eight-specialization limit even
    # after setting the global limit during wrapper construction. Set the limit
    # on this compile region, so surrounding config scopes cannot reset it.
    # Fullgraph makes exhaustion an error rather than a dense attention fallback.
    compiled = torch.compile(
        flex_attention,
        dynamic=False,
        fullgraph=True,
        recompile_limit=1024,
        mode="max-autotune-no-cudagraphs",
    )

    def run(*args, **kwargs):
        with torch._dynamo.config.patch(
            accumulated_recompile_limit=max(
                1024, torch._dynamo.config.accumulated_recompile_limit
            )
        ):
            return compiled(*args, **kwargs)

    return run


def relative_attention(
    q, k, v, r, rel_proj, seqlens, *, window, block_mask_cache, chunk_size=4096
):
    """Chunk queries, retaining all keys and the original absolute positions.

    The upstream [T, 48, 1024] relative-bias projection and its fp32 copy alone
    exceed the available memory at 256k. Recompute one query chunk at a time;
    this does not truncate attention or introduce a sliding window.
    """
    import torch
    from torch.nn.attention.flex_attention import AuxRequest, create_block_mask
    from torch.utils.checkpoint import checkpoint

    segment = torch.repeat_interleave(
        torch.arange(len(seqlens), device=q.device),
        torch.tensor(seqlens, device=q.device),
    )
    layout = (tuple(seqlens), window)
    total = q.shape[0]
    if block_mask_cache.get("layout") != layout:
        block_mask_cache.clear()
        block_mask_cache["layout"] = layout

    def attend(query, keys, values, relative, projection, offset, block_mask):
        bias = (
            torch.einsum("thd,de->the", relative, projection)
            .permute(1, 0, 2)
            .float()
            .contiguous()
        )
        extent = bias.shape[-1]
        query_len = query.shape[0]

        def score_mod(score, b, h, q_idx, kv_idx):
            distance = q_idx + offset - kv_idx
            rel = bias[h, q_idx.clamp(max=query_len - 1), distance.clamp(0, extent - 1)]
            return torch.where(distance < extent, score + rel, score)

        output, aux = _flex_compiled()(
            query.permute(1, 0, 2).unsqueeze(0),
            keys.permute(1, 0, 2).unsqueeze(0),
            values.permute(1, 0, 2).unsqueeze(0),
            score_mod=score_mod,
            block_mask=block_mask,
            scale=1.0 / math.sqrt(query.shape[-1]),
            enable_gqa=True,
            return_aux=AuxRequest(lse=True),
        )
        return output.squeeze(0).permute(1, 0, 2), aux.lse.squeeze(0).permute(1, 0)

    def make_mask(offset, query_len):
        def mask_mod(b, h, q_idx, kv_idx):
            absolute = q_idx + offset
            keep = (q_idx < query_len) & (absolute >= kv_idx)
            keep = keep & (
                segment[absolute.clamp(max=total - 1)]
                == segment[kv_idx.clamp(max=total - 1)]
            )
            if window is not None:
                keep = keep & (absolute - kv_idx <= window)
            return keep

        return mask_mod

    outputs, lses = [], []
    for start in range(0, q.shape[0], chunk_size):
        end = min(start + chunk_size, q.shape[0])
        offset = torch.tensor(start, device=q.device)
        key = (start, end)
        if key not in block_mask_cache:
            block_mask_cache[key] = create_block_mask(
                make_mask(offset, end - start),
                B=None,
                H=None,
                Q_LEN=end - start,
                KV_LEN=k.shape[0],
                device=q.device,
                _compile=True,
            )
        inputs = (
            q[start:end],
            k,
            v,
            r[start:end],
            rel_proj,
            offset,
            block_mask_cache[key],
        )
        output, lse = (
            checkpoint(attend, *inputs, use_reentrant=False)
            if torch.is_grad_enabled()
            else attend(*inputs)
        )
        outputs.append(output)
        lses.append(lse)
    return torch.cat(outputs), torch.cat(lses)


def _cpu_copy(tensor):
    import torch

    result = torch.empty_like(tensor, device="cpu", pin_memory=tensor.is_cuda)
    result.copy_(tensor.detach())
    return result


@dataclass
class CheckpointStore:
    """One forward's immutable completed residual blocks, stored once on CPU.

    Inkling appends completed blocks; previously completed blocks never change.
    Saving every stacked prefix separately would consume several TiB per node.
    Each checkpoint instead owns references to the CPU blocks in its prefix.
    """

    blocks: list = field(default_factory=list)

    def pack_context(self, tensor):
        if tensor.ndim != 4:
            raise ValueError("Expected Inkling context [T, 1, blocks, hidden]")
        count = tensor.shape[-2]
        if count < len(self.blocks):
            raise ValueError("Inkling residual context must only append blocks")
        self.blocks.extend(
            _cpu_copy(tensor[:, :, i]) for i in range(len(self.blocks), count)
        )
        return tensor.device, tuple(self.blocks), tensor.shape

    @staticmethod
    def unpack_context(packed):
        import torch

        device, blocks, shape = packed
        # Copy directly into the destination to avoid a second GPU-sized stack.
        result = torch.empty(shape, device=device, dtype=blocks[0].dtype)
        for i, block in enumerate(blocks):
            result[:, :, i].copy_(block)
        return result


_checkpoint_store = ContextVar("inkling2_checkpoint_store", default=None)


def activation_checkpoint(function, distribute_saved_activations, *args):
    """CPU-save outer layer checkpoints; nested MoE checkpoints remain on GPU."""
    import torch
    from megatron.core import tensor_parallel

    store = _checkpoint_store.get()
    if store is None:
        return tensor_parallel.checkpoint(function, distribute_saved_activations, *args)
    if distribute_saved_activations:
        raise ValueError(
            "Inkling long context requires TP=1 without activation sharding"
        )
    # Autograd retains the hook closures until backward. Capturing the tensor
    # here keeps every original GPU prefix alive despite copying it to CPU.
    # Packing happens synchronously while args still owns the input tensor.
    context_id = id(args[2])

    def pack(tensor):
        if id(tensor) == context_id:
            return True, store.pack_context(tensor)
        return False, (tensor.device, _cpu_copy(tensor))

    def unpack(packed):
        is_context, value = packed
        if is_context:
            return store.unpack_context(value)
        device, tensor = value
        return tensor.to(device)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        return tensor_parallel.checkpoint(function, False, *args)


def model_provider(pre_process=True, post_process=True, vp_stage=None, *, args):
    """Retain the upstream model/checkpoint schema and replace only its output."""
    from miles_plugins.models.inkling2.model import inkling2_model_provider

    if (
        args.loss_type != "policy_loss"
        or args.qkv_format != "thd"
        or args.recompute_loss_function
        or args.true_on_policy_mode
        or args.use_sampling_support_replay
    ):
        raise ValueError(
            "Inkling 256k requires packed policy_loss without loss/sampling replay"
        )
    model = inkling2_model_provider(
        pre_process,
        post_process,
        vp_stage,
        args=args,
    )
    original_forward = model.forward

    def forward(self, *inputs, **kwargs):
        token = _checkpoint_store.set(CheckpointStore())
        try:
            return original_forward(
                *inputs,
                output_processor=partial(
                    output_processor, temperature=args.rollout_temperature
                ),
                **kwargs,
            )
        finally:
            _checkpoint_store.reset(token)

    model.forward = MethodType(forward, model)
    return model
