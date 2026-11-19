"""Refresh source fixtures from the Inkling 2 recipe's pinned image (CPU only)."""

from pathlib import Path

import modal

from modal_dojo.train_recipes.miles_recipe import Inkling_2_Small_LoRA_256k_Recipe

ROOT = (
    Path(__file__).resolve().parents[1] / "tests/testdata/miles/inkling2_long_context"
)
SOURCE_PATHS = {
    "layers.py": "/root/miles/miles_plugins/models/inkling2/layers.py",
    "logit_processors.py": "/root/miles/miles/backends/training_utils/loss/hub/logit_processors.py",
    "transformer_block.py": "/root/Megatron-LM/megatron/core/transformer/transformer_block.py",
    "moe_layer.py": "/root/Megatron-LM/megatron/core/transformer/moe/moe_layer.py",
}
app = modal.App("fetch-inkling2-patch-snapshots")
image = modal.Image.from_registry(
    Inkling_2_Small_LoRA_256k_Recipe().docker_image
).entrypoint([])


@app.function(image=image, serialized=True)
def read_sources() -> dict[str, str]:
    return {name: Path(path).read_text() for name, path in SOURCE_PATHS.items()}


@app.local_entrypoint()
def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, source in read_sources.remote().items():
        (ROOT / f"{name}.input").write_text(source)
