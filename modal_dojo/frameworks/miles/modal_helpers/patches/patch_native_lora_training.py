"""Native LoRA runtime integration for Miles 65e785ff5e794a7081383e18e70c4555cb1fe0fb.

Remove the corresponding rewrites when the pinned Miles runtime implements them.
"""

from pathlib import Path


def patch_initial_lora_offload(root):
    """Reach the initial adapter-sync memory state without copying frozen weights.

    Miles releases every serving region and immediately restores weights while
    holding the controller lock. DeepSeek's disk backup makes this expensive and
    serializes engines that finish warmup at slightly different times. Releasing
    just KV/cache graphs leaves the same resident regions. The ordinary full
    offload before training still backs up the weights when actually needed.
    """
    path = Path(root) / "ray/rollout/server_cell.py"
    text = path.read_text()
    marker = (
        "# Modal Dojo: initial native-LoRA sync keeps frozen serving weights resident."
    )
    if marker in text:
        return
    anchor = (
        "            await api_client.release_memory_occupation()\n"
        "            await api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])\n"
    )
    if text.count(anchor) != 1:
        raise ValueError("Native Miles initial serving offload anchor changed")
    replacement = f"""            {marker}
            from miles.utils.lora.utils import lora_rollout_enabled

            if getattr(self.args, "model_name", None) == "deepseekv41" and lora_rollout_enabled(self.args):
                from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE

                await api_client.release_memory_occupation(
                    tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]
                )
            else:
                await api_client.release_memory_occupation()
                await api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])
"""
    text = text.replace(anchor, replacement)
    compile(text, str(path), "exec")
    path.write_text(text)


def patch_fragment_padding(root):
    """Pad each DeepSeek microbatch independently, preserving every real token.

    Native Miles pads every BSHD sample on a worker to that worker's longest
    sample. With one-sample microbatches this is unnecessary, especially for
    aborted/masked fragments. R3 sends padding to experts 0..topk-1, so excess
    padding also concentrates expert activations on the first EP rank.
    """
    path = Path(root) / "backends/training_utils/data/rollout.py"
    text = path.read_text()
    marker = "# Modal Dojo: one-sample DeepSeek microbatches need only their own aligned length."
    if marker in text:
        return
    anchor = '        rollout_data["max_seq_lens"] = [max_seq_len] * len(rollout_data["tokens"])\n'
    if text.count(anchor) != 1:
        raise ValueError("Native Miles BSHD padding anchor changed")
    replacement = (
        f"        {marker}\n"
        '        if args.model_name == "deepseekv41" and args.micro_batch_size == 1 and not args.use_dynamic_batch_size:\n'
        '            rollout_data["max_seq_lens"] = [\n'
        '                (length + pad_size - 1) // pad_size * pad_size for length in rollout_data["total_lengths"]\n'
        "            ]\n"
        "        else:\n" + "    " + anchor
    )
    text = text.replace(anchor, replacement)
    compile(text, str(path), "exec")
    path.write_text(text)


def patch_token_normalizer(root):
    """Masked scheduling fragments must not dilute the token-weighted gradient.

    Megatron accumulates this count across microbatches/ranks before dividing
    gradients; an individual all-masked microbatch may therefore report zero.
    The harness rejects an entirely untrainable conversation batch upstream.
    """
    path = Path(root) / "backends/training_utils/loss/objective.py"
    text = path.read_text()
    marker = "# Modal Dojo: masked fragments contribute neither loss nor normalization tokens."
    if marker in text:
        return
    anchor = (
        '    """The per-token loss normalizer: supervised tokens, with a fully masked sample counting as one."""\n'
        "    return sum(torch.clamp_min(loss_mask.sum(), 1) for loss_mask in loss_masks)\n"
    )
    if text.count(anchor) != 1:
        raise ValueError("Native Miles loss-normalization anchor changed")
    text = text.replace(
        anchor,
        f"    {marker}\n    return sum(loss_mask.sum() for loss_mask in loss_masks)\n",
    )
    compile(text, str(path), "exec")
    path.write_text(text)


def apply(root=Path("/root/miles/miles")):
    root = Path(root)
    patch_token_normalizer(root)
    patch_fragment_padding(root)
    patch_initial_lora_offload(root)


if __name__ == "__main__":
    apply()
