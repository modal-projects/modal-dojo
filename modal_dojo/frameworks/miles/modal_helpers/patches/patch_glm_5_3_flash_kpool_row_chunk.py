"""
image: radixark/miles:glm53next
file: megatron/bridge/models/glm5_next/kpool_indexer.py (Megatron-Bridge 6527b18)

Score kpool indexer logits in query-row chunks so long sequences do not
materialize a full fp32 [tokens, pools] logits tensor.
"""

from pathlib import Path

MARKER = "PATCHED_MODAL_DOJO_GLM53_KPOOL_ROW_CHUNK"

OLD = '    if pooled_k.shape[0] > 0:\n        # TileLang is only needed for the scoring kernel; keep it out of module import so the\n        # packed-sequence helpers (pool_boundaries, ...) stay importable without it.\n        from megatron.bridge.models.glm5.tilelang.tilelang_indexer_fwd import indexer_fwd_interface\n\n        with torch.no_grad():\n            pool_logits = indexer_fwd_interface(\n                index_q,\n                pooled_k,\n                head_weights,\n                pool_base.to(torch.int32),\n                (pool_base + eligible_pools).to(torch.int32),\n                clean_logits=True,\n            )\n    else:\n        pool_logits = torch.full((num_tokens, 1), float("-inf"), dtype=torch.float32, device=device)\n\n    replay = _indexer_replay_manager()\n    if replay is not None and replay.enabled:\n        topk_fn = replay.get_topk_fn(\n            _pool_topk_to_token_fn(seq_token_base, pool_base, local_positions, kpool),\n            return_probs=False,\n        )\n        tokens = topk_fn(pool_logits, index_topk)\n        shortcut = (local_positions + 1) <= index_topk\n        tokens = append_tail_and_pad(tokens, seq_token_base, local_positions, shortcut, kpool)\n    else:\n        tokens = _select_expand_tail(pool_logits, seq_token_base, pool_base, local_positions, index_topk, kpool)\n    return tokens.unsqueeze(1)\n\n'

NEW = '    def pool_logits_for(rows: slice) -> torch.Tensor:\n        if pooled_k.shape[0] == 0:\n            n = rows.stop - rows.start\n            return torch.full((n, 1), float("-inf"), dtype=torch.float32, device=device)\n        from megatron.bridge.models.glm5.tilelang.tilelang_indexer_fwd import indexer_fwd_interface\n\n        with torch.no_grad():\n            return indexer_fwd_interface(\n                index_q[rows],\n                pooled_k,\n                head_weights[rows],\n                pool_base[rows].to(torch.int32),\n                (pool_base + eligible_pools)[rows].to(torch.int32),\n                clean_logits=True,\n            )\n\n    replay = _indexer_replay_manager()\n    if replay is not None and replay.enabled:\n        pool_logits = pool_logits_for(slice(0, num_tokens))\n        topk_fn = replay.get_topk_fn(\n            _pool_topk_to_token_fn(seq_token_base, pool_base, local_positions, kpool),\n            return_probs=False,\n        )\n        tokens = topk_fn(pool_logits, index_topk)\n        shortcut = (local_positions + 1) <= index_topk\n        tokens = append_tail_and_pad(tokens, seq_token_base, local_positions, shortcut, kpool)\n    else:\n        tokens = torch.cat(\n            [\n                _select_expand_tail(\n                    pool_logits_for(slice(s, min(s + ROW_CHUNK, num_tokens))),\n                    seq_token_base[s : s + ROW_CHUNK],\n                    pool_base[s : s + ROW_CHUNK],\n                    local_positions[s : s + ROW_CHUNK],\n                    index_topk,\n                    kpool,\n                )\n                for s in range(0, num_tokens, ROW_CHUNK)\n            ]\n        )\n    return tokens.unsqueeze(1)\n'


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    if source.count(OLD) != 1:
        raise ValueError("GLM kpool indexer changed: expected one selection block")
    source = source.replace(OLD, NEW, 1)
    anchor = "\ndef kpool_select_topk("
    if source.count(anchor) != 1:
        raise ValueError("GLM kpool indexer changed: expected one kpool_select_topk")
    source = source.replace(anchor, "\nROW_CHUNK = 8192\n\n" + anchor, 1)
    return f"# {MARKER}\n" + source


def main() -> None:
    from importlib.util import find_spec

    path = (
        Path(find_spec("megatron.bridge").origin).parent
        / "models/glm5_next/kpool_indexer.py"
    )
    path.write_text(patch_source(path.read_text()))
    print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
