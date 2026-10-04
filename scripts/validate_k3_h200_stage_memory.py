"""Native 12-layer K3 stage memory preflight on one H200:8 node.

Run explicitly on eight H200 GPUs with the existing Hugging Face cache:
    uv run modal run --env <env> --detach scripts/validate_k3_h200_stage_memory.py

Uses real Megatron/Miles config, KDA/MLA, routed experts, TP2/CP4/EP8,
LoRA, FP8 and recomputation, with synthetic weights/tokens. A failed native
operation is useful evidence. Passing is not 64-GPU validation: PP transport,
real-weight routing skew, other stages and rollout colocation are absent.
"""

import json
from pathlib import Path
import modal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.kimi_k3_h200 import build_recipe
from modal_dojo import Kimi_K3

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".modal-dojo/new_models/Kimi_K3_H200"
app = modal.App("k3-h200-native-stage-memory")
recipe = build_recipe(
    metrics=None,
    actor_num_nodes=1,
    pipeline_model_parallel_size=1,
    decoder_first_pipeline_num_layers=None,
    decoder_last_pipeline_num_layers=None,
)
image = (
    modal.Image.from_registry(recipe.docker_image)
    .entrypoint([])
    .env(
        {
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_MNNVL_ENABLE": "0",
            "NCCL_RAS_ENABLE": "0",
            "PYTHONPATH": "/root/Megatron-LM:/root/miles",
            "HF_MODULES_CACHE": "/tmp/hf_modules",
        }
    )
)
for name in (
    "patch_k3_fused_lora",
    "patch_k3_fused_activation",
    "patch_k3_fp8_checkpoint",
):
    image = image.add_local_file(
        ROOT / f"modal_dojo/frameworks/miles/modal_helpers/patches/{name}.py",
        f"/root/{name}.py",
    )
image = image.add_local_file(Path(__file__), "/root/native_stage_source.py")


def worker():
    import contextlib
    import gc
    import os
    import shlex
    import subprocess
    import sys
    import time
    import traceback
    import torch
    import torch.distributed as dist
    import yaml
    from huggingface_hub import snapshot_download
    from torch_memory_saver import torch_memory_saver as tms

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    options = json.loads(Path("/tmp/native-stage-args.json").read_text())
    model_path = snapshot_download(
        "moonshotai/Kimi-K3", cache_dir="/hf-cache/hub", local_files_only=True
    )
    arch = shlex.split(
        subprocess.check_output(
            [
                sys.executable,
                "/root/miles/miles/utils/external_utils/model_args_utils.py",
                "kimi-k3",
            ],
            text=True,
        )
    )
    cli = options["cli"]
    drop = {
        "--custom-generate-function-path",
        "--custom-rm-path",
        "--custom-model-provider-path",
        "--load",
        "--ref-load",
        "--save",
        "--num-layers",
        "--moe-layer-freq",
        "--decoder-first-pipeline-num-layers",
        "--decoder-last-pipeline-num-layers",
    }
    clean = []
    i = 0
    while i < len(cli):
        if cli[i] in drop:
            i += 2
        else:
            clean.append(cli[i])
            i += 1
    extra = options["extra"]
    Path(f"/tmp/native-extra-{rank}.yaml").write_text(yaml.safe_dump(extra))
    sys.argv = [
        "train.py",
        *arch,
        *clean,
        "--hf-checkpoint",
        model_path,
        "--load",
        "/tmp/no-checkpoint-native-probe",
        "--save",
        "/tmp/native-probe-save",
        "--custom-config-path",
        f"/tmp/native-extra-{rank}.yaml",
        "--prompt-data",
        "/tmp/probe.jsonl",
        "--input-key",
        "prompt",
        "--label-key",
        "label",
        "--rm-type",
        "deepscaler",
    ]
    from miles.utils.arguments import parse_args

    with (
        open(f"/tmp/native-args-{rank}.log", "w") as argument_log,
        contextlib.redirect_stdout(argument_log),
    ):
        args = parse_args()
    # Parse/validate the real 93-layer architecture first, then select a
    # synthetic all-MoE 12-layer stage; this is not a checkpoint-compatible model.
    args.num_layers = 12
    args.moe_layer_freq = [1] * 12
    args.rank = rank
    args.world_size = 8
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.num_rollout = 2
    from miles.backends.megatron_utils.initialize import init

    init(args)
    from megatron.core import mpu
    from miles.backends.megatron_utils.model_provider import get_model_provider_func
    from miles_plugins.models.kimi_k3.lora import wrap_model_provider_with_kimi_k3_lora
    from megatron.training.training import get_model
    from megatron.core.enums import ModelType
    from megatron.core.packed_seq_params import PackedSeqParams

    tms.memory_margin_bytes = 4 * 2**30
    result = {
        "rank": rank,
        "status": "initializing",
        "layers": 12,
        "sequence": 65536,
        "tp": 2,
        "cp": 4,
        "ep": 8,
        "fp8": args.fp8,
        "recompute_num_layers": args.recompute_num_layers,
        "limitations": "synthetic weights, all-MoE first-stage layout; no PP transport, optimizer, rollout or real routing skew",
    }

    def mem(stage):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        v = {
            "stage": stage,
            "rank": rank,
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "free_gib": free / 2**30,
            "total_gib": total / 2**30,
        }
        result.setdefault("memory", []).append(v)
        print("NATIVE_MEMORY " + json.dumps(v), flush=True)
        Path(f"/tmp/native-stage-rank{rank}.json").write_text(json.dumps(result))

    started = time.monotonic()
    try:
        provider = wrap_model_provider_with_kimi_k3_lora(
            get_model_provider_func(args, "actor"), args
        )

        def stage_provider(pre_process=True, post_process=True, **kwargs):
            return provider(pre_process=True, post_process=False, **kwargs)

        with tms.region(tag="default", enable_cpu_backup=True):
            model = get_model(
                stage_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False
            )[0]
        model.train()
        from miles_plugins.models.hf_attention import detect_and_setup_hybrid_cp

        detect_and_setup_hybrid_cp(
            model, mpu.get_context_parallel_group(), mpu.get_context_parallel_rank(), 4
        )
        mem("model_initialized")
        # Verify the native LoRA freeze released initialization-only CPU copies.
        result["frozen_cpu_init_gib"] = (
            sum(
                getattr(p, "_high_precision_init_val").numel()
                * getattr(p, "_high_precision_init_val").element_size()
                for p in model.parameters()
                if isinstance(
                    getattr(p, "_high_precision_init_val", None), torch.Tensor
                )
                and not p.requires_grad
            )
            / 2**30
        )
        routed = []

        def hook(module, inputs):
            counts = inputs[1]
            entry = {
                "tokens": inputs[0].shape[0],
                "max_expert_tokens": max(counts),
                "min_expert_tokens": min(counts),
            }
            routed.append(entry)

        for name, module in model.named_modules():
            if name.endswith(".experts.linear_fc1"):
                module.register_forward_pre_hook(hook)
        seq = 65536
        tokens = torch.randint(
            100, args.padded_vocab_size, (1, seq // 4), device="cuda"
        )
        cu = torch.tensor([0, seq], dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=seq,
            max_seqlen_kv=seq,
            qkv_format="thd",
        )
        dist.barrier()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result["status"] = "forward"
        mem("before_forward")
        output = model(
            input_ids=tokens,
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=packed,
        )
        result["status"] = "backward"
        mem("after_forward")
        output.float().square().mean().backward()
        mem("after_backward")
        grads = [
            p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        result["finite_output"] = bool(torch.isfinite(output).all())
        result["gradient_count"] = len(grads)
        result["finite_grads"] = bool(grads) and all(
            bool(torch.isfinite(g).all()) for g in grads
        )
        result["status"] = (
            "passed"
            if result["finite_output"] and result["finite_grads"]
            else "nonfinite"
        )
        result["routing"] = routed
    except Exception as exc:
        result["status"] = "oom" if isinstance(exc, torch.OutOfMemoryError) else "error"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
        result["elapsed_seconds"] = time.monotonic() - started
        Path(f"/tmp/native-stage-rank{rank}.json").write_text(json.dumps(result))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)  # Never synchronize a CUDA stream after a peer OOM.
    result["elapsed_seconds"] = time.monotonic() - started
    Path(f"/tmp/native-stage-rank{rank}.json").write_text(json.dumps(result))
    print(
        "NATIVE_RESULT "
        + json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("traceback", "routing", "memory")
            }
        ),
        flush=True,
    )
    # Fail fast so torchrun terminates peers blocked in a collective.
    if result["status"] != "passed":
        raise SystemExit(1)
    dist.destroy_process_group()


@app.function(
    image=image,
    gpu="H200:8",
    cloud="gcp",
    memory=(1792 * 1024, 1920 * 1024),
    cpu=32,
    volumes={"/hf-cache": modal.Volume.from_name("huggingface-cache")},
    serialized=True,
    timeout=1800,
)
def check(cli, extra):
    import ast
    import os
    import signal
    import importlib.util
    import subprocess
    from torch_memory_saver.hooks.mode_preload import configure_subprocess

    for name in (
        "patch_k3_fused_lora",
        "patch_k3_fused_activation",
        "patch_k3_fp8_checkpoint",
    ):
        spec = importlib.util.spec_from_file_location(name, f"/root/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.apply()
    Path("/tmp/native-stage-args.json").write_text(
        json.dumps({"cli": cli, "extra": extra})
    )
    tree = ast.parse(Path("/root/native_stage_source.py").read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker"
    )
    Path("/tmp/native_stage_worker.py").write_text(
        "from pathlib import Path\nimport json\n" + ast.unparse(fn) + "\nworker()\n"
    )
    with (
        configure_subprocess(),
        open("/tmp/native-stdout.log", "w") as stdout,
        open("/tmp/native-stderr.log", "w") as stderr,
    ):
        proc = subprocess.Popen(
            [
                "/opt/sglang/bin/python3",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=8",
                "/tmp/native_stage_worker.py",
            ],
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    stdout = Path("/tmp/native-stdout.log").read_text()
    stderr = Path("/tmp/native-stderr.log").read_text()
    print(
        "\n".join(
            line
            for line in stdout.splitlines()
            if line.startswith(("NATIVE_MEMORY", "NATIVE_RESULT"))
        ),
        flush=True,
    )
    return {
        "exit_code": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "ranks": [
            json.loads(p.read_text())
            for p in sorted(Path("/tmp").glob("native-stage-rank*.json"))
        ],
    }


@app.local_entrypoint()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = check.remote(recipe.cli_args(model=Kimi_K3()), recipe.extra_config)
    (OUT / "native-stage-result.json").write_text(json.dumps(result, indent=2))
    (OUT / "native-stage-output.log").write_text(
        result.pop("stdout", "") + "\n" + result.pop("stderr", "")
    )
    print(
        json.dumps(
            {
                "exit_code": result["exit_code"],
                "ranks": [
                    {k: v for k, v in r.items() if k not in ("traceback", "routing")}
                    for r in result["ranks"]
                ],
            },
            indent=2,
        )
    )

    if (
        result["exit_code"] != 0
        or len(result["ranks"]) != 8
        or any(rank["status"] != "passed" for rank in result["ranks"])
    ):
        raise RuntimeError(f"Native H200 memory test failed; inspect {OUT}")
