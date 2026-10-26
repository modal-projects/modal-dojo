"""Validate the ``minimal`` trainer on a CUDA devbox / NVIDIA host (``torch.cuda``).

Start the devbox with ``--gpu`` on an NVIDIA-enabled host (this needs a real
NVIDIA GPU, so run it on your DGX / spark box, not the Mac):

    flyte start devbox --gpu

The train stage is routed onto the GPU environment
(``modal_dojo.flyte.pipeline.gpu_env`` / ``train_gpu_stage``), which requests
the GPU spec from ``MODAL_DOJO_FLYTE_GPU`` (default ``H100:1``). The image's
``torch`` (from PyPI) bundles CUDA on Linux, so the minimal trainer's
plain-PyTorch loop runs on ``torch.cuda``.

Run (from the repo root, on the GPU host):

    MODAL_DOJO_FLYTE_CONFIG=~/.flyte/config-devbox.yaml \\
    MODAL_DOJO_FLYTE_GPU=H100:1 \\
      python flyte/examples/validate_gpu_devbox.py

``MODAL_DOJO_FLYTE_GPU`` controls the requested GPU (``H100:1``, ``A100:1``,
``A10G:1``, … — match your host); it also selects the ``gpu`` accelerator.

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
    # accelerator="gpu" routes the train stage onto the CUDA environment;
    # MODAL_DOJO_FLYTE_GPU (set above) determines the requested spec.
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
