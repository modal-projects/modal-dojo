"""Validate the ``minimal`` trainer on a Union tenant GPU nodepool.

Uses the default config at ``~/.flyte/config.yaml`` (your Union tenant,
e.g. ``demo.hosted.unionai.cloud``) and Union's remote image builder, so the
first run builds the task image on Union's infra and later runs reuse it.

The train stage is routed onto the GPU environment, which requests the GPU
spec from ``MODAL_DOJO_FLYTE_GPU``. **Set it to a node you actually have** — the
Union console's nodepool view lists the accelerator types the tenant can
schedule (e.g. ``H100:1``, ``A100:1``, ``A10G:1``, ``L4:1``, …).

Run (from the repo root):

    MODAL_DOJO_FLYTE_GPU=H100:1 python flyte/examples/validate_union.py

After launch you can follow the run + train-stage logs ("minimal train on
device: cuda") in the Union console, or set FLYTE_RUN_ONLY=1 to launch and
print the run URL without blocking.
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
    # accelerator="gpu" routes the train stage onto the CUDA environment;
    # MODAL_DOJO_FLYTE_GPU (set above) must match a nudepool on this tenant.
    handle = flyte_launch(cfg, mode="minimal", accelerator="gpu")
    print("URL:", handle.url)
    print("run_id:", handle.run_id)
    print("status:", handle.status)
    if os.environ.get("FLYTE_RUN_ONLY"):
        print("FLYTE_RUN_ONLY set — not blocking on completion.")
        return handle
    handle.wait()
    print("final status:", handle.status)
    # The run's train-stage log should show "minimal train on device: cuda".
    return handle


if __name__ == "__main__":
    main()
