"""CPU preflight in the native DeepSeek image; never allocates training GPUs.

MODAL_ENVIRONMENT=training-team uv run python scripts/check_native_lora_runtime.py
Add --reuse-frozen-base-backup to build and check the optional CUDA extension.
"""

import argparse
from pathlib import Path

import modal

from modal_dojo import DeepSeek_V4_1_Flash_LoRA_Recipe
from modal_dojo.frameworks.miles.launcher import _build_miles_base_image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-frozen-base-backup", action="store_true")
    args = parser.parse_args()
    recipe = DeepSeek_V4_1_Flash_LoRA_Recipe(
        reuse_frozen_base_backup=args.reuse_frozen_base_backup
    )
    image = (
        _build_miles_base_image(recipe)
        .pip_install("pytest==8.4.2")
        .add_local_dir(
            ROOT / "tests", "/repo/tests", copy=True, ignore=["**/__pycache__"]
        )
    )
    app = modal.App("dojo-native-lora-cpu-preflight")

    @app.function(image=image, cpu=4, memory=16384, timeout=1800, serialized=True)
    def check():
        import os
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-rs",
                "tests/runtime/test_native_lora_runtime.py",
                "tests/test_frozen_weight_backup.py",
            ],
            cwd="/repo",
            env={**os.environ, "PYTHONPATH": "/repo:/root:/root/Megatron-LM"},
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode, result.stdout + result.stderr

    with modal.enable_output(), app.run():
        code, output = check.remote()
    print(output)
    if code or "skipped" in output.splitlines()[-1]:
        raise SystemExit(code or 1)


if __name__ == "__main__":
    main()
