"""Inference servers deployed by ``modal_dojo.deploy_recipes``.

Each class is a Modal server: ``@modal.enter()`` starts the inference engine
as a subprocess and Modal routes HTTP traffic straight to its port. What to
serve is described by a JSON document in ``SERVE_SPEC_ENV``, set on the
function when the app is built.
"""

from __future__ import annotations

import json
import os
from typing import Any

import modal

SERVE_SPEC_ENV = "MODAL_DOJO_SERVE_SPEC"


def _serve_spec() -> dict[str, Any]:
    return json.loads(os.environ[SERVE_SPEC_ENV])


class Server:
    """vLLM server. Spec keys: ``model_path``, ``served_model_name``,
    ``port``, ``n_gpu``, ``extra_args``."""

    @modal.enter()
    def startup(self):
        import subprocess

        spec = _serve_spec()
        model_path = spec["model_path"]

        if model_path.startswith("/") and os.path.isdir(model_path):
            hf_config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(hf_config_path):
                print("[modal-dojo] Converting checkpoint from mt to hf for serving.")

        cmd = [
            "vllm",
            "serve",
            "--uvicorn-log-level=info",
            model_path,
            "--served-model-name",
            spec["served_model_name"],
            "--host",
            "0.0.0.0",
            "--port",
            str(spec["port"]),
            "--tensor-parallel-size",
            str(spec["n_gpu"]),
            *spec["extra_args"],
        ]
        print(*cmd)
        subprocess.Popen(" ".join(cmd), shell=True)


class SGLangEndpoint:
    """SGLang server. Spec keys: ``model_path``, ``served_model_name``,
    ``port``, ``tp``, ``dp``, ``server_args``, ``startup_timeout``."""

    @modal.enter()
    def start(self):
        from modal_dojo.deploy_recipes.sglang_recipe._sglang_endpoint import (
            build_server_cmd,
            rewrite_chat_template_kwargs,
            start_server,
            wait_for_server_ready,
            warmup_chat_completions,
        )

        spec = _serve_spec()
        model_path = spec["model_path"]
        port = spec["port"]
        served_model_name = spec["served_model_name"]

        args = rewrite_chat_template_kwargs(spec["server_args"], model_path=model_path)
        cmd = build_server_cmd(
            model_path=model_path,
            port=port,
            tp=spec["tp"],
            dp=spec["dp"],
            extra_server_args=args,
        )
        self.proc = start_server(cmd)
        wait_for_server_ready(
            self.proc,
            port=port,
            timeout=float(spec["startup_timeout"]),
        )
        warmup_chat_completions(
            port=port,
            payload={
                "model": served_model_name,
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "max_tokens": 8,
                "temperature": 0,
            },
            successful_requests=2,
            request_timeout=60.0,
        )
        print(f"[modal-dojo] SGLang serving {served_model_name} ready.")

    @modal.exit()
    def stop(self):
        from modal_dojo.deploy_recipes.sglang_recipe._sglang_endpoint import (
            stop_server,
        )

        stop_server(getattr(self, "proc", None))
