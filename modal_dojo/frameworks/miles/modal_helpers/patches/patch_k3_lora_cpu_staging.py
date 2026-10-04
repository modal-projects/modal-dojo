"""
image: radixark/miles:dev-202609251434
commit: https://github.com/sgl-project/sglang/commit/880e3d2453eb7ef1738350e8c35ba2b956cc93a9
file: python/sglang/srt/managers/scheduler_components/weight_updater.py
"""

from pathlib import Path

TARGET = Path(
    "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/weight_updater.py"
)
MARKER = "PATCHED_K3_LORA_CPU_STAGING"
ANCHOR = """            if copy_tensors:
                # the stash outlives this RPC, but an IPC sender may reuse the bucket once it replies
                tensor = tensor.clone()
                if tensor.is_cuda:
                    copied_devices.add(tensor.device)
"""
REPLACEMENT = f"""            if copy_tensors:
                # {MARKER}
                tensor = tensor.detach().to(device="cpu", copy=True, non_blocking=False)
"""


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        if source.count(REPLACEMENT) != 1 or ANCHOR in source:
            raise RuntimeError(f"{target}: unexpected patched LoRA staging source")
        return
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: unexpected LoRA staging source")
    source = source.replace(ANCHOR, REPLACEMENT)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"Patched {target}: independent CPU staging for streamed K3 adapters")


if __name__ == "__main__":
    apply()
