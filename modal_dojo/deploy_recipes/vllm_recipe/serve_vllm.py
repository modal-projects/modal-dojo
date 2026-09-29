"""vLLM serving helper — builds a Modal app that hosts a model via vLLM.

The model is served by ``Server``, a Modal *server* class registered with
``@app.server`` (Modal's low-latency routing service for inference workloads).
The endpoint is the Modal class itself — its ``@modal.enter()`` starts the
``vllm serve`` subprocess.

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
    checkpoints_volume: "Volume | str | None" = None,
    checkpoints_mount_path: str | None = None,
    unauthenticated: bool = True,
) -> "App":
    import modal
    from modal import App, Image, Volume

    from modal_dojo.common import hf_secrets

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

    @app.server(
        image=image,
        gpu=f"{gpu}:{n_gpu}",
        scaledown_window=10 * 60,
        startup_timeout=startup_timeout,
        volumes=volumes,
        secrets=hf_secrets(),
        serialized=True,
        include_source=False,
        port=vllm_port,
        exit_grace_period=25,
        routing_region="us-east",
        target_concurrency=32,
        unauthenticated=unauthenticated,
    )
    class Server:
        @modal.enter()
        def startup(self):
            import os
            import subprocess

            if model_path.startswith("/") and os.path.isdir(model_path):
                hf_config_path = os.path.join(model_path, "config.json")
                if not os.path.exists(hf_config_path):
                    print(
                        "[modal-dojo] Converting checkpoint from mt to hf for serving."
                    )

            cmd = [
                "vllm",
                "serve",
                "--uvicorn-log-level=info",
                model_path,
                "--served-model-name",
                served_model_name,
                "--host",
                "0.0.0.0",
                "--port",
                str(vllm_port),
                "--tensor-parallel-size",
                str(n_gpu),
                *_extra,
            ]
            print(*cmd)
            subprocess.Popen(" ".join(cmd), shell=True)

    setattr(app, "Server", Server)
    return app
