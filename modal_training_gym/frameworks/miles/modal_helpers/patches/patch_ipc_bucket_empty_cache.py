"""Release exported LoRA buffers before SGLang applies the complete adapter.

Per-bucket empty_cache is insufficient: the exporter yields views into PP
slabs, the caller keeps its final bucket, and CUDA IPC can retain freed blocks.
Drop that final bucket after exhausting the exporter, then collect IPC and
allocator caches in finalize(), after every sender has received its engine's
acknowledgement and joined the existing Gloo barrier. This frees the trainer's
temporary storage before end_weight_update allocates normalized LoRA tensors.

Only Kimi-K3 installs this patch. No tensor values or transfer order change.
"""

from pathlib import Path

MARKER = "PATCHED_IPC_SYNC_FINALIZE_CACHE"
ROOT = Path("/root/miles/miles/backends/training_utils/weight_update")


def apply(root: Path = ROOT) -> None:
    replacements = {
        root / "updater.py": (
            "            protocol.after_base_weights()",
            "            # Release the final exported view before protocol finalization.\n"
            f"            bucket = None  # {MARKER}\n"
            "            protocol.after_base_weights()",
        ),
        root / "protocols" / "cuda_ipc.py": (
            "    def after_engines_resumed(self) -> None:",
            "    def finalize(self, weight_version: int) -> None:\n"
            f"        # {MARKER}: all bucket RPCs completed before the Gloo barrier.\n"
            "        torch.cuda.ipc_collect()\n"
            "        torch.cuda.empty_cache()\n"
            "        logger.info(\n"
            "            'IPC buffers released before engine finalize: allocated=%.2f GiB reserved=%.2f GiB',\n"
            "            torch.cuda.memory_allocated() / 2**30,\n"
            "            torch.cuda.memory_reserved() / 2**30,\n"
            "        )\n\n"
            "    def after_engines_resumed(self) -> None:",
        ),
    }
    updates = {}
    for path, (old, new) in replacements.items():
        source = path.read_text()
        if MARKER in source:
            continue
        if source.count(old) != 1:
            raise RuntimeError(
                f"{path}: expected one patch anchor; inspect Miles source"
            )
        updates[path] = source.replace(old, new)
    for path, source in updates.items():
        compile(source, str(path), "exec")
    for path, source in updates.items():
        path.write_text(source)
        print(f"Patched {path}: release IPC buffers before engine finalization")


if __name__ == "__main__":
    apply()
