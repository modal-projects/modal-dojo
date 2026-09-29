"""Fetch Slime source snapshots used by the local rollout-patcher golden test.

The source files exist only inside the pinned Slime image, so this maintenance
utility uses Modal to read that image. Normal pytest collection never imports or
runs this module. It refreshes the committed ``*.input`` fixtures; CI checks the
resulting Git diff to detect image-source drift.
"""

from pathlib import Path

import modal

from modal_dojo.frameworks.slime.launcher import SLIME_IMAGE

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTDATA_DIR = REPO_ROOT / "tests" / "testdata" / "slime"
SLIME_SOURCE_PATHS = {
    "train.py": "/root/slime/train.py",
    "train_async.py": "/root/slime/train_async.py",
    "rollout.py": "/root/slime/slime/ray/rollout.py",
    "rm_hub_init.py": "/root/slime/slime/rollout/rm_hub/__init__.py",
    "sglang_rollout.py": "/root/slime/slime/rollout/sglang_rollout.py",
    "actor.py": "/root/slime/slime/backends/megatron_utils/actor.py",
    "model.py": "/root/slime/slime/backends/megatron_utils/model.py",
}

app = modal.App("fetch-slime-snapshots")
image = modal.Image.from_registry(SLIME_IMAGE).entrypoint([])


@app.function(image=image, serialized=True)
def read_sources() -> dict[str, str]:
    return {name: Path(path).read_text() for name, path in SLIME_SOURCE_PATHS.items()}


@app.local_entrypoint()
def main() -> None:
    TESTDATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, source in read_sources.remote().items():
        (TESTDATA_DIR / f"{name}.input").write_text(source)
