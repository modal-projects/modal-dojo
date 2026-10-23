"""Validate the Flyte 2 backend end-to-end (minimal mode, bundled smoke configs).

Uses ``modal_dojo.flyte.smoke`` (module-level, ships in the code bundle) so the
remote container can re-import the config classes, with no weights/datasets to
download — the whole path (image, object-store I/O via ``flyte.io.Dir``,
orchestration, checkpoint) is exercised cheaply.

Usage:
    # Union
    python flyte/validate_minimal.py

    # local devbox
    MODAL_DOJO_FLYTE_CONFIG=~/.flyte/config-devbox.yaml python flyte/validate_minimal.py

Set FLYTE_RUN_ONLY=1 to launch and print the run URL without blocking on
completion (first image builds can be slow).
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
    handle = flyte_launch(cfg, mode="minimal")
    print("URL:", handle.url)
    print("run_id:", handle.run_id)
    print("status:", handle.status)
    if os.environ.get("FLYTE_RUN_ONLY"):
        print("FLYTE_RUN_ONLY set — not blocking on completion.")
        return handle
    handle.wait()
    print("final status:", handle.status)
    return handle


if __name__ == "__main__":
    main()
