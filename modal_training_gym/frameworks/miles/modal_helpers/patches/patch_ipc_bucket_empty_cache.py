"""
image: radixark/miles:dev-202609251434
commit: https://github.com/radixark/miles/commit/41c5e38b94ea23677de93b01a4a77d55677a8f09
file: miles/miles/backends/training_utils/weight_update/updater.py::WeightUpdater.update_weights
file: miles/miles/backends/training_utils/weight_update/protocols/cuda_ipc.py::UpdateWeightFromTensor
"""

from pathlib import Path

MARKER = "PATCHED_IPC_SYNC_FINALIZE_CACHE"
ROOT = Path("/root/miles/miles/backends/training_utils/weight_update")


def apply(root: Path = ROOT) -> None:
    replacements = {
        root / "updater.py": (
            "            protocol.after_base_weights()",
            f"            bucket = None  # {MARKER}\n"
            "            protocol.after_base_weights()",
        ),
        root / "protocols" / "cuda_ipc.py": (
            "    def after_engines_resumed(self) -> None:",
            "    def finalize(self, weight_version: int) -> None:\n"
            f"        # {MARKER}\n"
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
