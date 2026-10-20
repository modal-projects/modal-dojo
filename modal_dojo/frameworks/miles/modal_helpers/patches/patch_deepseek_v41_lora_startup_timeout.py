"""Allow the full V4.1 model's initial weight backup to finish before cancellation.

Miles' two-minute cell tick also covers release/resume of serving weights.
Backing up and restoring 112 GiB per GPU on local disk can exceed
ten minutes; allow the complete transition, including the slowest node. Cancellation leaves weights
offloaded, so the next generation health check cannot complete initialization.
This patch is confined to the pinned native-LoRA image.

commit: https://github.com/radixark/miles/commit/65e785ff5e794a7081383e18e70c4555cb1fe0fb
file: miles/ray/rollout/inference_controller.py
TODO: Remove when Miles exposes a configurable cell-transition timeout that
can accommodate full-model disk offload/restore.
"""

from pathlib import Path

TARGET = Path("/root/miles/miles/ray/rollout/inference_controller.py")
MARKER = "MODAL_DOJO_V41_LORA_STARTUP_TIMEOUT"
OLD = "CELL_TICK_TIMEOUT_SECONDS = 120.0"
NEW = f"CELL_TICK_TIMEOUT_SECONDS = 1800.0  # {MARKER}"


def apply(path: Path = TARGET) -> bool:
    source = path.read_text()
    if MARKER in source:
        return False
    if source.count(OLD) != 1:
        raise ValueError(f"Unexpected Miles cell tick timeout implementation: {path}")
    patched = source.replace(OLD, NEW, 1)
    compile(patched, str(path), "exec")
    path.write_text(patched)
    return True


if __name__ == "__main__":
    apply()
