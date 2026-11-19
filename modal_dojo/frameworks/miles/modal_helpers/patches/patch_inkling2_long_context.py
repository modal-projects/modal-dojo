"""Wire bounded Inkling intermediates into the PR #847 image, fail closed.

Image: radixark/miles:inkling-2-small@sha256:a2e3c49d5a266d9374d67ca4b706767b7bdf7d798958dd51520f61c29597e9a5
The recipe also applies the existing, generic MoE sequence-chunk patch.
"""

from pathlib import Path

MARKER = "PATCHED_MODAL_DOJO_INKLING2_LONG_CONTEXT"
_MODULE = "modal_dojo.frameworks.miles.inkling2_long_context"
_CHECKPOINT = """                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    *rotary_pos_emb,
                    padding_mask,
                )"""
_SCORES = """    assert non_loss_data
    if rollout_sampling_mask is not None:
"""
_SCORES_NEW = f"""    assert non_loss_data
    if getattr(logits, "_modal_dojo_inkling2_scores", False):
        from {_MODULE} import select_token_scores

        return select_token_scores(
            logits, args=args, unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths, response_lengths=response_lengths,
            with_entropy=with_entropy, entropy_requires_grad=entropy_requires_grad,
            max_seq_lens=max_seq_lens, rollout_sampling_mask=rollout_sampling_mask,
        )
    if rollout_sampling_mask is not None:
"""
_LAYERS = "\nclass Inkling2TransformerLayer(TransformerLayer):"
_ATTENTION = """        rel_logits = torch.einsum("thd,de->the", r, self.rel_proj)
        out, lse = relative_attention(
            q, k, v, rel_logits, seqlens, window=self.window, block_mask_cache=self._block_masks
        )"""


def patch_source(source: str, target: str) -> str:
    if MARKER in source:
        return source
    match target:
        case "transformer_block.py":
            old = _CHECKPOINT
            new = (
                f"                from {_MODULE} import activation_checkpoint\n\n"
                + _CHECKPOINT.replace(
                    "tensor_parallel.checkpoint(", "activation_checkpoint(", 1
                )
            )
        case "logit_processors.py":
            old, new = _SCORES, _SCORES_NEW
        case "layers.py":
            if source.count(_ATTENTION) != 1:
                raise ValueError("Pinned Inkling relative-attention source changed")
            source = source.replace(
                _ATTENTION,
                """        out, lse = relative_attention(
            q, k, v, r, self.rel_proj, seqlens,
            window=self.window, block_mask_cache=self._block_masks,
        )""",
                1,
            )
            old = _LAYERS
            new = (
                f"\nfrom {_MODULE} import attn_res_mix, relative_attention\n\n"
                + _LAYERS
            )
        case _:
            raise ValueError(f"Unknown Inkling patch target: {target}")
    if source.count(old) != 1:
        raise ValueError(f"Pinned Inkling source changed: {target}")
    return f"# {MARKER}\n" + source.replace(old, new, 1)


def main() -> None:
    from importlib.util import find_spec

    megatron = Path(find_spec("megatron.core").origin).parent
    paths = (
        megatron / "transformer/transformer_block.py",
        Path("/root/miles/miles/backends/training_utils/loss/hub/logit_processors.py"),
        Path("/root/miles/miles_plugins/models/inkling2/layers.py"),
    )
    # Validate every target before modifying any of them.
    patched = [(path, patch_source(path.read_text(), path.name)) for path in paths]
    for path, source in patched:
        path.write_text(source)
        print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
