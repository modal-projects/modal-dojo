"""Time each memory-handoff operation in pinned K3 SGLang.

Source: sglang@880e3d2453eb7ef1738350e8c35ba2b956cc93a9.
No extra CUDA synchronization or changes to operation order.
"""

from pathlib import Path

TARGET = Path(
    "/sgl-workspace/sglang/python/sglang/srt/managers/"
    "scheduler_components/weight_updater.py"
)
MARKER = "DOJO_MEMORY_HANDOFF"


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        return
    start = source.index("    def release_memory_occupation(")
    end = source.index("    def check_weights(", start)
    block = source[start:end]
    operations = (
        "self.flush_cache()",
        "torch.get_device_module().synchronize()",
        "self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)",
        "self.stashed_model_static_state = _export_static_state(\n"
        "                self.tp_worker.model_runner.model\n            )",
        "torch.distributed.barrier(self.tp_cpu_group)",
        "self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)",
        "self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)",
        "self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)",
        "self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)",
        "_import_static_state(\n                self.tp_worker.model_runner.model,\n"
        "                self.stashed_model_static_state,\n            )",
        "self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)",
    )
    # Validate every anchor before modifying the installed file.
    for operation in operations:
        if operation not in block:
            raise RuntimeError(f"{target}: missing handoff operation {operation}")
    lines = block.splitlines(keepends=True)
    result = []
    phase = "release"
    index = 0
    while index < len(lines):
        line = lines[index]
        if "def resume_memory_occupation(" in line:
            phase = "resume"
        operation = next(
            (
                op
                for op in operations
                if "".join(lines[index:]).startswith(
                    " " * (len(line) - len(line.lstrip())) + op + "\n"
                )
            ),
            None,
        )
        if operation is None:
            result.append(line)
            index += 1
            continue
        indent = line[: len(line) - len(line.lstrip())]
        label = operation.split("(", 1)[0].replace("self.", "")
        if "GPU_MEMORY_TYPE_" in operation:
            label += ":" + operation.split("GPU_MEMORY_TYPE_", 1)[1].rstrip(")")
        result.extend(
            [
                indent + "import time as _dojo_time\n",
                indent + "_dojo_started = _dojo_time.monotonic()\n",
            ]
        )
        count = len(operation.splitlines())
        result.extend(lines[index : index + count])
        result.append(
            indent + f'logger.info("{MARKER} phase={phase} op={label} elapsed_s=%.3f", '
            "_dojo_time.monotonic() - _dojo_started)\n"
        )
        index += count
    patched = source[:start] + "".join(result) + source[end:]
    compile(patched, str(target), "exec")
    target.write_text(patched)
    print(f"Patched {target}: per-operation offload timings")


if __name__ == "__main__":
    apply()
