"""CPU validation in the recipe's actual Miles/SGLang image; no training GPUs.

MODAL_ENVIRONMENT=training-team uv run python scripts/check_deepseek_lora_coverage.py
"""

from pathlib import Path

import modal

from modal_dojo import DeepSeek_V4_1_Flash_LoRA_Recipe
from modal_dojo.frameworks.miles.launcher import _build_miles_base_image


def main():
    root = Path(__file__).resolve().parents[1]
    image = (
        _build_miles_base_image(DeepSeek_V4_1_Flash_LoRA_Recipe())
        .pip_install("pytest==8.4.2", "pytest-asyncio==1.2.0")
        .add_local_dir(
            root / "tests", "/repo/tests", copy=True, ignore=["**/__pycache__"]
        )
    )
    app = modal.App("dojo-deepseek-lora-coverage-cpu")

    @app.function(image=image, cpu=4, memory=16384, timeout=1800, serialized=True)
    def check():
        import os
        import subprocess
        import sys
        import xml.etree.ElementTree as ET

        results = []
        for cwd, tests in (
            ("/repo", ["tests/runtime/test_deepseek_lora_coverage.py"]),
            (
                "/root/miles",
                [
                    "tests/fast/backends/megatron_utils/lora/native/test_deepseek_v4_1.py",
                    "tests/fast/ray/rollout/test_server_cell_state_machine.py",
                    "tests/fast/backends/sglang_utils/test_compute_server_args.py",
                ],
            ),
        ):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-rs",
                    "--junitxml=/tmp/coverage-results.xml",
                    *tests,
                ],
                cwd=cwd,
                env={**os.environ, "PYTHONPATH": "/repo:/root:/root/Megatron-LM"},
                capture_output=True,
                text=True,
                check=False,
            )
            skipped = sum(
                int(suite.attrib.get("skipped", 0))
                for suite in ET.parse("/tmp/coverage-results.xml").iter("testsuite")
            )
            results.append((result.returncode, skipped, result.stdout + result.stderr))
        return results

    with modal.enable_output(), app.run():
        results = check.remote()
    failed = False
    for code, skipped, output in results:
        print(output)
        failed |= code != 0 or skipped != 0
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
