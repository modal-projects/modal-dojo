# ---
# order: 2
# ---
#
# # Training multi-turn coding agents
#
# This tutorial trains [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) on
# [SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2).
# During rollouts, the agent inspects repositories, edits code, and runs commands
# in a [Modal Sandbox](https://modal.com/docs/guide/sandboxes) using
# [Harbor](https://docs.harborframework.com/).

import hashlib
import io
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Literal

import modal
from datasets import load_dataset

from modal_training_gym import (
    DatasetConfig,
    Qwen3_6_27B,
    Qwen3_6_27B_Recipe,
    TrainConfig,
)

# ## Get the dataset
#
# We'll first convert the HF dataset into Harbor tasks and create repository-disjoint
# train/eval sets. We filter out tasks that would otherwise degrade learning or require 
# oversampling if removed, which prolongs rollout generation.

HF_DATASET = "nebius/SWE-rebench-V2"
HF_REVISION = "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0"
SLIME_GIT_REVISION = "3585d4a7eb1a5c108810238b47c37d3107d0a2ba"
TASK_ROOT = Path("/data") / f"swe-rebench-{HF_REVISION[:8]}-{SLIME_GIT_REVISION[:8]}"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def is_eval(row: dict) -> bool:
    return int(_digest(row["metadata"]["source"]["repo"]), 16) % 5 == 0


def convert(root: Path) -> None:
    from agentic_rl.envs.harbor import convert as harbor
    from agentic_rl.envs.swe_rebench import convert as swerebench

    shutil.rmtree(root / "tasks", ignore_errors=True)
    rows = []
    for row in load_dataset(
        HF_DATASET, split="train", revision=HF_REVISION, streaming=True
    ):
        if not swerebench._passes_quality(row, "A"):
            continue
        task_dir = root / "tasks" / swerebench._safe_id(row["instance_id"])
        try:
            swerebench.build_task_dir(row, task_dir)
            converted = harbor.translate_task(task_dir, dataset=root.name)
        except (swerebench.SkipRow, harbor.SkipTask):
            shutil.rmtree(task_dir, ignore_errors=True)
            continue
        converted["metadata"]["task_path"] = f"{root.name}/tasks/{task_dir.name}"
        converted["metadata"]["source"] = {"repo": row["repo"]}
        rows.append(converted)
    staging = root / ".all.converted.jsonl"
    staging.write_text("".join(json.dumps(row) + "\n" for row in rows))
    os.replace(staging, root / "all.converted.jsonl")


class SWERebench(DatasetConfig):
    def __init__(
        self,
        split: Literal["train", "eval"],
        limit: int | None = None,
        keep: frozenset[str] | None = None,
    ):
        self.split = split
        self.limit = limit
        self.keep = keep

    def cache_key(self) -> str:
        keep = "all" if self.keep is None else _digest(",".join(sorted(self.keep)))[:12]
        return f"{TASK_ROOT.name}-{self.split}-{self.limit}-{keep}"

    def input_key(self) -> str:
        return "prompt"

    def label_key(self) -> str:
        return "label"

    def apply_chat_template(self) -> bool:
        return False

    def rows(self):
        if not (TASK_ROOT / "all.converted.jsonl").exists():
            convert(TASK_ROOT)
        with (TASK_ROOT / "all.converted.jsonl").open() as source:
            rows = [json.loads(line) for line in source]
        rows = [
            row
            for row in rows
            if is_eval(row) == (self.split == "eval")
            and (self.keep is None or row["metadata"]["instance_id"] in self.keep)
        ]
        rows.sort(key=lambda row: _digest(row["metadata"]["instance_id"]))
        return rows[: self.limit]


TRAIN_TASKS = 300
PROBE_SAMPLES = 8


def probed_tasks(config: TrainConfig) -> frozenset[str]:
    volume = modal.Volume.from_name(config.recipe.data_volume_name, create_if_missing=True)
    model = config.model.model_name.replace("/", "--")
    path = f"{TASK_ROOT.name}/probe-{model}-n{PROBE_SAMPLES}-t{TRAIN_TASKS}.json"
    try:
        return frozenset(json.loads(b"".join(volume.read_file(path))))
    except FileNotFoundError:
        pass
    rewards = defaultdict(float)
    unusable = set()
    for sample in config.evaluate(config.dataset, n_samples=PROBE_SAMPLES):
        task = sample.metadata["instance_id"]
        rewards[task, sample.rollout_index] += sample.score
        if sample.metadata.get("remove_sample"):
            unusable.add(task)
    solved = defaultdict(set)
    for (task, _), reward in rewards.items():
        solved[task].add(reward > 0)
    keep = frozenset(
        task for task, values in solved.items() if values == {True, False} and task not in unusable
    )
    if not keep:
        raise RuntimeError("No probed training task had mixed rewards.")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(json.dumps(sorted(keep)).encode()), path)
    return keep


# ## Start training
#
# With the [Qwen3_6_27B_Recipe](https://gym.modal.dev/reference/qwen3_6_27b_recipe)
# recipe class, it's just that simple.

def build_config(dataset, eval_temperature=None):
    return TrainConfig(
        model=Qwen3_6_27B(),
        dataset=dataset,
        eval_dataset=SWERebench("eval"),
        recipe=Qwen3_6_27B_Recipe(
            slime_git_repository="https://github.com/modal-projects/slime.git",
            slime_git_revision=SLIME_GIT_REVISION,
            data_volume_name="slime-data",
            memory=(1024, 2 * 1024 * 1024),
            train_function_kwargs={"ephemeral_disk": 2 * 1024 * 1024},
            environment={
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": "1",
                "NCCL_RAS_ENABLE": "0",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "ASYNC_RL_TASK_ROOT": "/data",
                "ASYNC_RL_REWARD_SHAPE": "binary",
            },
            image_run_commands=[
                "apt-get update && apt-get install -y --no-install-recommends "
                "rdma-core libibverbs1 ibverbs-providers",
                "uv pip install --system modal==1.5.5 mini-swe-agent datasets",
            ],
            image_env={
                "PYTHONPATH": "/root/Megatron-LM/:/root/slime",
                "MSWEA_SILENT_STARTUP": "1",
            },
            app_tags={"agentic_rollout": "harbor"},
            gpu_type="H200",
            colocate=False,
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            rollout_num_gpus=32,
            rollout_num_gpus_per_engine=2,
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=2,
            context_parallel_size=2,
            sequence_parallel=True,
            conversion_tensor_model_parallel_size=4,
            conversion_pipeline_model_parallel_size=2,
            decoder_last_pipeline_num_layers=30,
            attention_backend="flash",
            ref_load="",
            sglang_speculative_algorithm="EAGLE",
            sglang_speculative_num_steps=3,
            sglang_speculative_eagle_topk=1,
            sglang_speculative_num_draft_tokens=4,
            num_rollout=500,
            rollout_batch_size=32,
            n_samples_per_prompt=8,
            global_batch_size=256,
            advantage_estimator="grpo",
            lr=4e-6,
            lr_decay_style="constant",
            weight_decay=0.1,
            eps_clip=0.2,
            eps_clip_high=None,
            use_kl_loss=False,
            kl_coef=0.0,
            entropy_coef=0.0,
            rollout_temperature=1.0,
            eval_temperature=eval_temperature,
            rollout_max_response_len=8192,
            eval_max_response_len=8192,
            eval_interval=5,
            n_samples_per_eval_prompt=1,
            save="/checkpoints",
            save_interval=5,
            sglang_server_concurrency=32,
            max_tokens_per_gpu=16384,
            log_probs_chunk_size=128,
            capture_trace=True,
            custom_rollout_log_function="agentic_rl.metrics.log_rollout_data",
            extra_config={
                "custom_generate_function_path": "agentic_rl.generate.generate",
                "agentic_max_steps": 75,
                "agentic_episode_timeout": 1800,
                "agentic_eval_timeout": None,
                "agentic_exec_timeout": 120,
                "router_policy": "consistent_hashing",
                "skip_eval_before_train": False,
            },
        ),
    )


if __name__ == "__main__":
    keep = probed_tasks(build_config(SWERebench("train", limit=TRAIN_TASKS)))
    config = build_config(SWERebench("train", keep=keep), eval_temperature=0.6)
    run = config.launch()
    print(f"run id: {run.training_run_id}")
