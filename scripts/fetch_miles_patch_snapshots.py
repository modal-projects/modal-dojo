"""Fetch miles source snapshots used by the local patcher golden tests.

The source files exist only inside the pinned miles image, so this maintenance
utility uses Modal to read that image. Normal pytest collection never imports or
runs this module. It refreshes the committed ``*.input`` fixtures; CI checks the
resulting Git diff to detect image-source drift.
"""

from pathlib import Path

import modal

from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

MILES_IMAGE = MilesRecipe().docker_image

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTDATA_DIR = REPO_ROOT / "tests" / "testdata" / "miles"
MILES_SOURCE_PATHS = {
    "train.py": "/root/miles/train.py",
    "train_async.py": "/root/miles/train_async.py",
    "rollout_manager.py": "/root/miles/miles/ray/rollout/rollout_manager.py",
    "rm_hub_init.py": "/root/miles/miles/rollout/rm_hub/__init__.py",
    "sglang_rollout.py": "/root/miles/miles/rollout/sglang_rollout.py",
    "actor.py": "/root/miles/miles/backends/megatron_utils/actor.py",
    "model.py": "/root/miles/miles/backends/megatron_utils/model.py",
    "log_utils.py": "/root/miles/miles/backends/training_utils/log_utils.py",
}

app = modal.App("fetch-miles-snapshots")
image = modal.Image.from_registry(MILES_IMAGE).entrypoint([])


@app.function(image=image, serialized=True)
def read_sources() -> dict[str, str]:
    return {name: Path(path).read_text() for name, path in MILES_SOURCE_PATHS.items()}


@app.local_entrypoint()
def main() -> None:
    TESTDATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, source in read_sources.remote().items():
        (TESTDATA_DIR / f"{name}.input").write_text(source)
