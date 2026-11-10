"""vLLM serving helper — builds a Modal app that hosts a model via vLLM.

The model is served by ``modal_dojo.remote.serving.Server``, a Modal *server*
class registered with ``app.server`` (Modal's low-latency routing service for
inference workloads). The endpoint is the Modal class itself — its
``@modal.enter()`` starts the ``vllm serve`` subprocess.

``model_path`` accepts either:
  - a **HuggingFace repo id** (e.g. ``"Qwen/Qwen3-4B"``) — vLLM downloads
    the weights from HF into the cache volume on first boot, or
  - an **absolute container path** to an HF-format checkpoint directory
    (must contain ``config.json``) — typically a subdirectory of a
    training-checkpoints volume you pass via ``checkpoints_volume``.

Deploy separately from the training app:

    modal deploy path/to/tutorial.py::serve_app
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modal import App, Volume
    from modal_dojo.deploy_recipes.vllm_recipe import VllmRecipe


def build_vllm_serve_app(
    *,
    recipe: "VllmRecipe",
    app_name: str,
    model_path: str,
    served_model_name: str,
    revision: str | None = None,
    checkpoints_volume: "Volume | str | None" = None,
    checkpoints_mount_path: str | None = None,
    unauthenticated: bool = False,
) -> "App":
    from modal import App, Image, Volume

    from modal_dojo.common import hf_secrets
    from modal_dojo.remote import serving

    gpu = recipe.gpu or "H100"
    n_gpu = recipe.n_gpu or 1
    vllm_port = 8000
    vllm_version = "0.13.0"
    startup_timeout = recipe.startup_timeout

    image = (
        Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        .uv_pip_install(
            f"vllm=={vllm_version}",
            "huggingface-hub==0.36.0",
        )
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
        .add_local_python_source("modal_dojo", copy=True)
    )

    hf_cache_vol = Volume.from_name("huggingface-cache", create_if_missing=True)
    vllm_cache_vol = Volume.from_name("vllm-cache", create_if_missing=True)
    volumes: dict[str | PurePosixPath, Any] = {
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    }

    if checkpoints_volume is not None:
        mount = checkpoints_mount_path or "/checkpoints"
        if isinstance(checkpoints_volume, str):
            checkpoints_volume = Volume.from_name(
                checkpoints_volume, create_if_missing=True
            )
        volumes[mount] = checkpoints_volume

    tags = {
        "_modal_source": "modal-dojo",
        "_modal_job_type": "serving",
        "_modal_framework": "vllm-serve",
    }
    app = App(app_name, tags=tags)

    _extra = list(recipe.extra_vllm_args or [])
    if revision:
        _extra = ["--revision", revision, *_extra]

    app.Server = app.server(
        image=image,
        gpu=f"{gpu}:{n_gpu}",
        scaledown_window=10 * 60,
        startup_timeout=startup_timeout,
        volumes=volumes,
        secrets=hf_secrets(),
        env={
            serving.SERVE_SPEC_ENV: json.dumps(
                {
                    "model_path": model_path,
                    "served_model_name": served_model_name,
                    "port": vllm_port,
                    "n_gpu": n_gpu,
                    "extra_args": _extra,
                }
            )
        },
        include_source=False,
        port=vllm_port,
        exit_grace_period=25,
        routing_region="us-east",
        target_concurrency=32,
        unauthenticated=unauthenticated,
    )(serving.Server)
    return app
