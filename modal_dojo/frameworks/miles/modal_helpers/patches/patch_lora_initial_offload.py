"""
image: radixark/miles:dev-202609251434
commit: https://github.com/radixark/miles/commit/41c5e38b94ea23677de93b01a4a77d55677a8f09
file: miles/miles/ray/rollout/server_cell.py
"""

from pathlib import Path

TARGET = Path("/root/miles/miles/ray/rollout/server_cell.py")
MARKER = "PATCHED_LORA_INITIAL_OFFLOAD"
IMPORT = "from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS"
NEW_IMPORT = (
    "from sglang.srt.constants import (\n"
    "    GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS,\n"
    ")"
)
ANCHOR = """            await api_client.release_memory_occupation()
            await api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])
"""
REPLACEMENT = f"""            # {MARKER}
            if (
                (getattr(self.args, "lora_rank", 0) or 0) > 0
                and self.meta.update_weights
                and not self.args.debug_rollout_only
                and not self.args.check_weight_update_equal
            ):
                logger.info("LoRA startup: keeping base weights resident; releasing KV and CUDA graphs")
                await api_client.release_memory_occupation(
                    tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]
                )
            else:
                await api_client.release_memory_occupation()
                await api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])
"""


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        return
    if source.count(IMPORT) != 1 or source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: unexpected LoRA initialization source")
    source = source.replace(IMPORT, NEW_IMPORT).replace(ANCHOR, REPLACEMENT)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"Patched {target}: preserve frozen base during initial LoRA update")


if __name__ == "__main__":
    apply()
