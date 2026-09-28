"""Refresh K3 patch inputs from its recipe's pinned Miles image.

Run with ``uv run modal run scripts/fetch_kimi_k3_patch_snapshots.py``.
Review input drift before updating the corresponding golden outputs.
"""

from pathlib import Path

import modal

from modal_training_gym.train_recipes.miles_recipe.kimi_k3 import _DOCKER_IMAGE

app = modal.App("fetch-kimi-k3-patch-snapshots")
image = modal.Image.from_registry(_DOCKER_IMAGE).entrypoint([])


@app.function(image=image, serialized=True)
def read_sources() -> dict[str, str]:
    root = Path("/root/miles/miles/backends/training_utils/weight_update")
    return {
        "checkpoint_io.py": (root.parent / "checkpoint_io.py").read_text(),
        "cuda_ipc.py": (root / "protocols/cuda_ipc.py").read_text(),
        "updater.py": (root / "updater.py").read_text(),
        "hf_weight_iterator.py": Path(
            "/root/miles/miles/backends/megatron_utils/update_weight/hf_weight_iterator.py"
        ).read_text(),
    }


@app.local_entrypoint()
def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "tests/testdata/miles/kimi_k3"
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in read_sources.remote().items():
        (destination / f"{name}.input").write_text(source)
