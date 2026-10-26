"""Validate the ``minimal`` trainer on the Apple-Silicon devbox (``torch.mps``).

The devbox must be started with ``--apple-silicon`` so it runs the Apple
Silicon bridge:

    flyte start devbox --apple-silicon

The train stage requests :func:`flyte.Resources(gpu="AppleSilicon")` (see
``modal_dojo.flyte.pipeline.apple_silicon_env`` / ``train_apple_silicon_stage``),
so Flyte runs it *natively on this Mac*, in a Python env built from the task
image's packages. The minimal trainer's plain-PyTorch loop then runs on
``torch.mps`` (this is auto-detected inside ``_train_minimal``).

Run (from the repo root):

    MODAL_DOJO_FLYTE_CONFIG=~/.flyte/config-devbox.yaml \\
      python flyte/examples/validate_apple_silicon.py

Set FLYTE_RUN_ONLY=1 to launch and print the run URL without blocking.
"""

from __future__ import annotations

import os

from modal_dojo import TrainConfig, flyte_launch
from modal_dojo.flyte.smoke import SmokeDataset, SmokeModel
from modal_dojo.train_recipes.slime_recipe import SlimeRecipe


def main():
    cfg = TrainConfig(
        dataset=SmokeDataset(),
        model=SmokeModel(),
        recipe=SlimeRecipe(actor_num_gpus_per_node=1, gpu_type="H100"),
        backend="flyte",
    )
    handle = flyte_launch(cfg, mode="minimal", accelerator="mps")
    print("URL:", handle.url)
    print("run_id:", handle.run_id)
    print("status:", handle.status)
    if os.environ.get("FLYTE_RUN_ONLY"):
        print("FLYTE_RUN_ONLY set — not blocking on completion.")
        return handle
    handle.wait()
    print("final status:", handle.status)
    # The run's train-stage log should show "minimal train on device: mps".
    return handle


if __name__ == "__main__":
    main()
