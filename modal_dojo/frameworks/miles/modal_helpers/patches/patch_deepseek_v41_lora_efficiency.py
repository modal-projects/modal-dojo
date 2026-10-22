"""Avoid redundant frozen-weight I/O and worker-wide padding for native V4.1 LoRA.

Source: radixark/miles@6c8e2378cfe3dc5f99633e9719a80586dac86683.
Remove each rewrite when upstream implements the corresponding optimization.
The startup rewrite preserves upstream's cancellation-safe release task.
"""

from pathlib import Path


def patch_initial_lora_offload(root):
    """Initial adapter sync needs KV/graphs released, with frozen weights resident.

    Releasing all regions and then restoring weights reaches the same state but
    unnecessarily copies the entire frozen base to and from local disk. Normal
    training transitions still offload and restore the weights as before.
    """
    path = Path(root) / "ray/rollout/server_cell.py"
    text = path.read_text()
    marker = (
        "# Modal Dojo: initial native-LoRA sync keeps frozen serving weights resident."
    )
    if marker in text:
        return
    anchor = (
        "        await api_client.release_memory_occupation()\n"
        "        await api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])\n"
    )
    if text.count(anchor) != 1:
        raise ValueError("Native Miles initial serving offload anchor changed")
    replacement = f"""        {marker}
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


def patch_microbatch_padding(root):
    """Align each one-sample BSHD microbatch to its own length, retaining R3 alignment.

    Padding every sample to the worker's longest sequence wastes computation
    and concentrates routing-replay padding on the first expert-parallel rank.
    Keep upstream behavior for pipeline parallelism and other batching modes.
    """
    path = Path(root) / "backends/training_utils/data/rollout.py"
    text = path.read_text()
    marker = "# Modal Dojo: one-sample DeepSeek microbatches need only their own aligned length."
    if marker in text:
        return
    anchor = '        rollout_data["max_seq_lens"] = [max_seq_len] * len(rollout_data["tokens"])\n'
    if text.count(anchor) != 1:
        raise ValueError("Native Miles BSHD padding anchor changed")
    replacement = f"""        {marker}
        if (
            args.model_name == "deepseekv41"
            and args.lora_rank > 0
            and args.micro_batch_size == 1
            and not args.use_dynamic_batch_size
            and args.pipeline_model_parallel_size == 1
        ):
            rollout_data["max_seq_lens"] = [
                (length + pad_size - 1) // pad_size * pad_size for length in rollout_data["total_lengths"]
            ]
        else:
    {anchor}"""
    text = text.replace(anchor, replacement)
    compile(text, str(path), "exec")
    path.write_text(text)


def apply(root=Path("/root/miles/miles")):
    patch_microbatch_padding(root)
    patch_initial_lora_offload(root)


if __name__ == "__main__":
    apply()
