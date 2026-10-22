"""Explicit one-H200 CUDA gate for immutable snapshots; no model or shared volumes.

Run only when the experiment's GPU budget permits:
MODAL_ENVIRONMENT=training-team uv run --frozen python scripts/check_frozen_weight_backup.py

This test does not enable reuse in the training recipe or modify a running app.
Full DeepSeek serving still needs a later, separately identified canary.
"""

from modal_dojo.frameworks.miles.launcher import _build_miles_base_image
from pathlib import Path
from modal_dojo import DeepSeek_V4_1_Flash_LoRA_Recipe

import modal

ROOT = Path(__file__).resolve().parents[1]


def main():
    recipe = DeepSeek_V4_1_Flash_LoRA_Recipe(reuse_frozen_base_backup=True)
    image = _build_miles_base_image(recipe).add_local_file(
        ROOT / "tests/testdata/frozen_backup/gpu_probe.py",
        "/tmp/gpu_probe.py",
        copy=True,
    )
    app = modal.App("dojo-frozen-backup-correctness")

    @app.function(
        image=image, gpu="H200", cpu=4, memory=16384, timeout=600, serialized=True
    )
    def check():
        import subprocess

        # Each deliberate illegal write gets a fresh process/CUDA context.
        runner = r"""
import subprocess, sys
from torch_memory_saver import configure_subprocess
for mode in ("torch", "preload"):
    for mutation in ("none", "before", "after", "before_copy", "after_copy"):
        def run():
            return subprocess.run([sys.executable, "/tmp/gpu_probe.py", mode, mutation],
                                  capture_output=True, text=True, timeout=80)
        if mode == "preload":
            with configure_subprocess():
                result = run()
        else:
            result = run()
        output = result.stdout + result.stderr
        print(f"mode={mode} mutation={mutation} exit={result.returncode}", flush=True)
        print(output, flush=True)
        if mutation == "none":
            assert result.returncode == 0, output
        else:
            assert "ATTEMPTING_PROTECTED_WRITE" in output, output
            assert result.returncode != 0, "protected write unexpectedly succeeded"
            assert ("illegal memory access" in output.lower()
                    or "requested write on read-only memory" in output.lower()), output
print("FROZEN_BACKUP_CUDA_GATE_PASSED", flush=True)
"""
        result = subprocess.run(
            ["/opt/sglang/bin/python", "-c", runner],
            capture_output=True,
            text=True,
            timeout=560,
            check=False,
        )
        return result.returncode, result.stdout + result.stderr

    with modal.enable_output(), app.run():
        code, output = check.remote()
    print(output)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
