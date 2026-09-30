"""SGLang server subprocess helpers.

Stateless functions for launching, health-checking, warming up, and
tearing down an SGLang ``launch_server`` subprocess. These are composed
by the ``SGLangEndpoint`` Modal server class in ``serve_sglang.py`` — the
endpoint *is* the Modal class, and these helpers are the plumbing it
drives from ``@modal.enter()`` / ``@modal.exit()``.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

# Operational flags that always apply, independent of the recipe.
DEFAULT_OPERATIONAL_ARGS: dict[str, str] = {
    "--enable-metrics": "",
    "--decode-log-interval": "1",
    "--enable-cache-report": "",
    "--model-loader-extra-config": '{"enable_multithread_load":true,"num_threads":64}',
}


def build_server_cmd(
    *,
    model_path: str,
    port: int = 8000,
    tp: int | None = None,
    dp: int | None = None,
    extra_server_args: dict[str, str] | None = None,
) -> list[str]:
    """Build the ``sglang.launch_server`` argv."""
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--model-path",
        model_path,
    ]
    if tp is not None:
        cmd.extend(["--tp", str(tp)])
    if dp is not None:
        # SGLang's CLI flag is ``--dp-size`` (not ``--dp``); ``--enable-dp-attention``
        # is required for DeepSeek-style MLA DP attention and is a no-op when
        # ``dp_size == 1``.
        cmd.extend(["--dp-size", str(dp), "--enable-dp-attention"])

    merged = {**DEFAULT_OPERATIONAL_ARGS, **(extra_server_args or {})}
    for key, value in merged.items():
        if value == "":
            cmd.append(key)
        else:
            cmd.extend([key, value])
    return cmd


def rewrite_chat_template_kwargs(
    extra_server_args: dict[str, str] | None,
    *,
    model_path: str,
) -> dict[str, str]:
    """Convert ``--chat-template-kwargs`` to ``--chat-template``.

    SGLang doesn't expose ``--chat-template-kwargs`` as a CLI flag
    (it's a per-request API parameter only).  Work around this by
    downloading the model's chat template from its tokenizer config,
    prepending Jinja ``{% set %}`` statements for the requested
    kwargs, and pointing sglang at the modified template file.

    Returns a copy of ``extra_server_args`` with the rewrite applied.
    """
    args = dict(extra_server_args or {})
    raw = args.pop("--chat-template-kwargs", None)
    if raw is None:
        return args

    kwargs = json.loads(raw) if isinstance(raw, str) else raw

    template = _load_chat_template(model_path)
    if not template:
        print(
            "[sglang] warning: could not load chat template from model; "
            "ignoring --chat-template-kwargs"
        )
        return args

    prefix = "\n".join(f"{{% set {k} = {json.dumps(v)} %}}" for k, v in kwargs.items())
    path = "/tmp/_sglang_chat_template.jinja"
    with open(path, "w") as f:
        f.write(prefix + "\n" + template)

    args["--chat-template"] = path
    print(f"[sglang] rewrote --chat-template-kwargs as --chat-template {path}")
    return args


def _load_chat_template(model_path: str) -> str:
    """Return the model's Jinja chat template string, or ``""``."""
    import os

    config = None
    # Local checkpoint path
    local = os.path.join(model_path, "tokenizer_config.json")
    if os.path.isfile(local):
        with open(local) as f:
            config = json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_path, "tokenizer_config.json")
            with open(path) as f:
                config = json.load(f)
        except Exception as exc:
            print(f"[sglang] failed to download tokenizer_config.json: {exc}")
            return ""

    template = config.get("chat_template", "")
    if isinstance(template, list):
        template = next(
            (t["template"] for t in template if t.get("name") == "default"),
            template[0]["template"] if template else "",
        )
    return template


def start_server(cmd: list[str]) -> subprocess.Popen:
    """Launch the SGLang server subprocess."""
    print(f"[sglang] starting: {shlex.join(cmd)}")
    return subprocess.Popen(cmd)


def wait_for_server_ready(
    proc: subprocess.Popen,
    *,
    port: int,
    timeout: float,
    poll_interval: float = 5.0,
) -> None:
    """Poll the SGLang ``/health`` endpoint until ready or the process dies."""
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"

    while time.time() < deadline:
        if (rc := proc.poll()) is not None:
            raise subprocess.CalledProcessError(rc, cmd=proc.args)
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if 200 <= resp.getcode() < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(poll_interval)

    raise TimeoutError(f"SGLang health check timed out after {timeout}s on port {port}")


def warmup_chat_completions(
    *,
    port: int,
    payload: Mapping[str, Any],
    successful_requests: int = 3,
    request_timeout: float = 30.0,
) -> None:
    """Send a few chat-completion requests to warm caches and JIT."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(dict(payload)).encode()

    for i in range(successful_requests):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=request_timeout):
                pass
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            print(f"[sglang] warmup {i + 1}/{successful_requests} failed: {exc}")


def stop_server(proc: subprocess.Popen | None) -> None:
    """Terminate the SGLang server subprocess, escalating to kill if needed."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
