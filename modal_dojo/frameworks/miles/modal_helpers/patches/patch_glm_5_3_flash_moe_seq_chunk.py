"""
image: radixark/miles:glm53next
file: megatron/core/transformer/moe/moe_layer.py (image Megatron-LM)

Run allgather/alltoall-dispatched MoE layers over a fixed number of token chunks
after routing, so dispatch buffers, permuted expert rows, and unpermuted outputs
on a hot EP rank are bounded by the chunk instead of the full long-context
sequence. The chunk count is fixed so every rank issues the same collectives.
With grad enabled each chunk is checkpointed, so backward holds one chunk's
expert activations at a time.

TODO: Revisit upstream chunked EP once it supports BF16 Bridge LoRA and full
recompute; the current MXFP8/flex implementation rejects full recompute:
https://github.com/NVIDIA/Megatron-LM/pull/7331
"""

from pathlib import Path

MARKER = "PATCHED_MODAL_DOJO_GLM53_MOE_SEQ_CHUNK"

OLD = """                    hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)

                    if intermediate_tensors is not None:
                        return hidden_states, probs, shared_expert_output
"""

NEW = """                    if intermediate_tensors is None and _modal_dojo_moe_chunks(self, hidden_states) > 1:
                        return (
                            _modal_dojo_chunked_moe(
                                self, hidden_states, probs, routing_map, shared_expert_output
                            ),
                            None,
                        )
                    hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)

                    if intermediate_tensors is not None:
                        return hidden_states, probs, shared_expert_output
"""

APPENDIX = """

_MODAL_DOJO_MOE_CHUNKS = 8


def _modal_dojo_moe_chunks(layer, hidden_states):
    dispatcher = layer.token_dispatcher
    chunks = _MODAL_DOJO_MOE_CHUNKS
    if (
        not isinstance(dispatcher, (MoEAllGatherTokenDispatcher, MoEAlltoAllTokenDispatcher))
        or getattr(dispatcher, "drop_and_pad", False)
        or layer.fwd_execution_map != ["route", "expert_compute", "postprocess"]
        or layer.config.moe_latent_size
        or layer.config.overlap_dispatch_backward_with_experts_wgrad
        or layer.config.cuda_graph_impl != "none"
        or (hasattr(layer, "_inference_token_dispatcher") and InferenceMode.is_active())
    ):
        chunks = 1
    if not getattr(layer, "_modal_dojo_moe_chunks_logged", False):
        layer._modal_dojo_moe_chunks_logged = True
        print(
            f"GLM256K_MOE_SEQ_CHUNK layer={layer.layer_number} chunks={chunks} "
            f"shape={tuple(hidden_states.shape)} dispatcher={type(dispatcher).__name__} "
            f"shared_overlap={layer.shared_expert_overlap}",
            flush=True,
        )
    return chunks


def _modal_dojo_moe_chunk(layer, hidden_states, probs, routing_map):
    hidden_states, probs = layer.preprocess(hidden_states, probs, routing_map)
    hidden_states, probs = layer.dispatch(hidden_states, probs)
    hidden_states, _ = layer.routed_experts_compute(hidden_states, probs)
    return layer.token_dispatcher.combine_postprocess(layer.combine(hidden_states))


def _modal_dojo_chunked_moe(layer, hidden_states, probs, routing_map, shared_expert_output):
    shape = hidden_states.shape
    flat = hidden_states.reshape(-1, shape[-1])
    rows = flat.shape[0]
    chunks = _modal_dojo_moe_chunks(layer, hidden_states)
    output = None
    for idx in range(chunks):
        begin, end = rows * idx // chunks, rows * (idx + 1) // chunks
        args = (flat[begin:end], probs[begin:end], routing_map[begin:end])
        if torch.is_grad_enabled():
            chunk = tensor_parallel.checkpoint(
                lambda h, p, r: _modal_dojo_moe_chunk(layer, h, p, r), False, *args
            )
        else:
            chunk = _modal_dojo_moe_chunk(layer, *args)
        if output is None:
            output = chunk.new_empty((rows, *chunk.shape[1:]))
        output[begin:end] = chunk
        del chunk
    output = output.view(*shape[:-1], output.shape[-1])
    if shared_expert_output is not None:
        output = output + shared_expert_output
    return output
"""


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    for needle in (
        OLD,
        "import math\n",
        "InferenceMode\n",
        "MoEAllGatherTokenDispatcher,\n",
        "MoEAlltoAllTokenDispatcher,\n",
        "tensor_parallel.checkpoint(\n",
    ):
        if needle not in source:
            raise ValueError(f"Megatron moe_layer changed: missing {needle!r}")
    return f"# {MARKER}\n" + source.replace(OLD, NEW, 1) + APPENDIX


def main() -> None:
    from importlib.util import find_spec

    path = (
        Path(find_spec("megatron.core").origin).parent / "transformer/moe/moe_layer.py"
    )
    path.write_text(patch_source(path.read_text()))
    print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
