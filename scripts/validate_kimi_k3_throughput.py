"""Pruned-K3 proof of concurrent 64k inference and adapter refresh.

Defaults to one B300; K3_THROUGHPUT_GPUS=16 selects two B300:8 nodes and TP16.
Timings compare the same two requests sequentially and concurrently. A pruned
model cannot establish full-model throughput or fit.
"""

import json
import os
from pathlib import Path

import modal
from modal.experimental import clustered

from configs.kimi_k3_b300 import build_recipe
from modal_dojo.frameworks.miles.launcher import _build_miles_base_image

recipe = build_recipe(concurrency_per_engine=2, metrics=None)
GPU_COUNT = int(os.environ.get("K3_THROUGHPUT_GPUS", "1"))
if GPU_COUNT not in (1, 16):
    raise ValueError("K3_THROUGHPUT_GPUS must be 1 or 16")
NODE_COUNT = 2 if GPU_COUNT == 16 else 1
app = modal.App("k3-b300-throughput-proof")
image = _build_miles_base_image(recipe).add_local_file(
    str(Path(__file__).with_name("kimi_k3_probe_adapter.py")),
    "/root/k3_adapter_checks.py",
    copy=True,
)


@app.function(
    image=image,
    gpu="B300:8" if GPU_COUNT == 16 else "B300",
    cloud="aws" if GPU_COUNT == 16 else None,
    cpu=16,
    memory=(128 * 1024, 256 * 1024),
    timeout=2400,
    retries=0,
    volumes={"/hf": modal.Volume.from_name("huggingface-cache")},
    experimental_options={"efa_enabled": True} if GPU_COUNT == 16 else {},
    serialized=True,
)
@(
    clustered(size=NODE_COUNT, rdma=True)
    if GPU_COUNT == 16
    else lambda function: function
)
def probe(state):
    import math
    import os
    import signal
    import subprocess
    import sys
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import requests
    from huggingface_hub import snapshot_download
    from modal.experimental import get_cluster_info
    from transformers import AutoTokenizer

    from modal_dojo.common.launcher_utils import prewarm_remote_code

    if GPU_COUNT == 16:
        info = get_cluster_info()
    else:
        from types import SimpleNamespace

        info = SimpleNamespace(rank=0, container_ipv4_ips=["127.0.0.1"])
    rank = info.rank
    ip = info.container_ipv4_ips[rank]
    env = {
        **os.environ,
        **recipe.environment,
        "SGLANG_HOST_IP": ip,
        "HOST_IP": ip,
        "MILES_HOST_IP": ip,
        "TRITON_CACHE_DIR": "/tmp/k3-probe/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/k3-probe/inductor",
        "TILELANG_CACHE_DIR": "/tmp/k3-probe/tilelang",
        "SGLANG_CACHE_DIR": "/tmp/k3-probe/sglang",
    }
    if GPU_COUNT == 1:
        env.pop("GLOO_SOCKET_IFNAME", None)
        env.pop("TP_SOCKET_IFNAME", None)
    os.environ.update(env)
    model = snapshot_download(
        "Pinaster/Kimi-K3-4layer-64experts",
        revision="9eac75a1ff2214467edd8f98fc20e656f7b474ca",
        cache_dir="/hf/hub",
        local_files_only=True,
    )
    prewarm_remote_code(model, env, required=True)
    settings = {
        key.removeprefix("sglang_"): value
        for key, value in vars(recipe).items()
        if key.startswith("sglang_")
        and value is not None
        and key not in ("sglang_server_concurrency", "sglang_config")
    }
    if GPU_COUNT == 1:
        # TRTLLM MLA excludes 64 < query heads < 128. TP1 has 96 heads;
        # production TP16 has six. This is a scheduler/LoRA concurrency proof,
        # not an attention-kernel performance prediction for the full model.
        settings["decode_attention_backend"] = "cutedsl_mla"
    settings.update(
        model_path=model,
        trust_remote_code=True,
        tp_size=GPU_COUNT,
        nnodes=NODE_COUNT,
        node_rank=rank,
        dist_init_addr=f"{info.container_ipv4_ips[0]}:20300",
        host="0.0.0.0",
        port=20350,
        log_level="info",
        context_length=65536,
        chunked_prefill_size=4096,
        lora_use_virtual_experts=False,
        enable_lora=True,
        lora_target_modules=["all"],
        experts_shared_outer_loras=True,
        enable_weights_cpu_backup=True,
        enable_memory_saver=True,
        skip_server_warmup=True,
    )
    Path("/tmp/k3-args.json").write_text(json.dumps(settings))
    program = (
        "import json; from sglang.srt.entrypoints.http_server import launch_server; "
        "from sglang.srt.server_args import ServerArgs; "
        "launch_server(ServerArgs(**json.load(open('/tmp/k3-args.json'))))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", program],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    def read_logs():
        for line in process.stdout:
            print(line, end="", flush=True)

    reader = threading.Thread(target=read_logs, daemon=True)
    reader.start()
    started = time.monotonic()
    client = requests.Session()
    client.trust_env = False
    result = {
        "rank": rank,
        "gpu_count": GPU_COUNT,
        "model": model,
        "decode_attention_backend": settings["decode_attention_backend"],
    }
    try:
        if rank:
            while not state.get("done", False):
                if process.poll() is not None:
                    raise RuntimeError(f"Worker engine exited {process.returncode}")
                if time.monotonic() - started > 2200:
                    raise TimeoutError("Head did not finish")
                time.sleep(2)
            return result
        url = f"http://{ip}:20350"
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Engine exited {process.returncode}")
            if time.monotonic() - started > 1200:
                raise TimeoutError("Engine never became healthy")
            try:
                if client.get(url + "/health_generate", timeout=25).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        result["startup_seconds"] = time.monotonic() - started
        sys.path.insert(0, "/root")
        from k3_adapter_checks import check_adapters

        result["adapter_checks"] = check_adapters(client, url, model, tp_size=GPU_COUNT)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        prompts = []
        for record in (
            "First diagnostic record: all checks passed.\n",
            "Second independent record: checks complete.\n",
        ):
            ids = tokenizer.encode(record, add_special_tokens=False)
            prompts.append((ids * (64512 // len(ids) + 1))[:64512])

        def generate(ids):
            with requests.Session() as session:
                session.trust_env = False
                tick = time.monotonic()
                response = session.post(
                    url + "/generate",
                    json={
                        "input_ids": ids,
                        "lora_path": "staging_probe",
                        "return_logprob": True,
                        "sampling_params": {
                            "max_new_tokens": 1022,
                            "min_new_tokens": 1022,
                            "ignore_eos": True,
                            "temperature": 0,
                        },
                    },
                    timeout=600,
                )
                response.raise_for_status()
                output = response.json()
                entries = output["meta_info"]["output_token_logprobs"]
                assert len(entries) == 1022 and all(
                    math.isfinite(x[0]) for x in entries
                )
                return {
                    "seconds": time.monotonic() - tick,
                    "tokens": len(entries),
                    "total_tokens": len(ids) + len(entries),
                    "logprob_sum": sum(x[0] for x in entries),
                }

        # Warm the actual long-sequence shapes before measuring either candidate.
        with ThreadPoolExecutor(max_workers=2) as pool:
            result["warmup"] = list(pool.map(generate, prompts))
        result["benchmarks"] = []
        for concurrency in (1, 2, 2, 1):
            client.get(url + "/flush_cache", timeout=60).raise_for_status()
            tick = time.monotonic()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                outputs = list(pool.map(generate, prompts))
            duration = time.monotonic() - tick
            result["benchmarks"].append(
                {
                    "concurrency": concurrency,
                    "seconds": duration,
                    "generated_tokens_per_second": 2044 / duration,
                    "outputs": outputs,
                }
            )
            print("K3_THROUGHPUT " + json.dumps(result["benchmarks"][-1]), flush=True)
        client.get(url + "/health_generate", timeout=30).raise_for_status()
        response = client.post(
            url + "/unload_lora_adapter",
            json={"lora_name": "staging_probe"},
            timeout=60,
        )
        response.raise_for_status()
        assert response.json()["success"], response.text
        result["health_and_adapter_unload"] = True
        result["gpu_memory"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return result
    finally:
        if rank == 0:
            state["done"] = True
        client.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        reader.join(timeout=2)


@app.local_entrypoint()
def main():
    with modal.Dict.ephemeral() as state:
        result = probe.remote(state)
    destination = Path(".modal-dojo/new_models/Kimi_K3_B300/throughput-proof.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
