"""SWE-smith full-context validation for DeepSeek V4.1 Flash LoRA.

Run with --launch for five steps on 32 H200s; --steps 1 is the initial proof.
Training responses fill the 32,768-token prompt-plus-response window, less
the two tokens reserved by the pinned SGLang scheduler.
Evaluation retains ordinary stopping. Reward is patch token F1, not code
execution or SWE-bench resolved rate. No autoconfig group is published here.
"""

import argparse
import hashlib
import importlib.util
import json
import tempfile
import uuid
from dataclasses import replace
from functools import partial
from pathlib import Path

from modal_dojo import (
    TrainConfig,
    DeepSeek_V4_1_Flash_LoRA,
    DeepSeek_V4_1_Flash_LoRA_Recipe,
)
from modal_dojo.common.dataset import DatasetConfig

MODEL_REPO = DeepSeek_V4_1_Flash_LoRA.model_name
MODEL_REVISION = DeepSeek_V4_1_Flash_LoRA.revision

DATASET_REPO = "SWE-bench/SWE-smith"
DATASET_REVISION = "ea6d7173829c7ec8fa16c22055699ff2e9188091"
CONTEXT_LENGTH = 32768
PROMPT_LIMIT = 4096
# tp_worker.get_worker_info reserves one token, then
# Scheduler.init_req_max_new_tokens reserves one more for admission.
SGLANG_RESERVED_TOKENS = 2


class ValidationPrompts(DatasetConfig):
    hf_repo = DATASET_REPO

    def __init__(self, records, revision):
        self.revision = revision
        payload = "".join(json.dumps(row) + "\n" for row in records)
        self.digest = hashlib.sha256(payload.encode()).hexdigest()
        self.local_path = str(
            Path(tempfile.gettempdir()) / f"swe-validation-{self.digest}.jsonl"
        )
        self.remote_path = f"/opt/validation/{self.digest}.jsonl"
        Path(self.local_path).write_text(payload)

    def input_key(self):
        return "prompt"

    def apply_chat_template(self):
        return False

    def label_key(self):
        return "label"

    def cache_key(self):
        return f"free-swimmer-swebench:{self.revision}:{self.digest}"

    def rows(self):
        path = Path(self.remote_path)
        if not path.exists():
            path = Path(self.local_path)
        with path.open() as handle:
            for line in handle:
                yield json.loads(line)


def prepare_data():
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    revision = DATASET_REVISION
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION, trust_remote_code=True
    )
    encoder_path = hf_hub_download(
        MODEL_REPO, "encoding/encoding.py", revision=MODEL_REVISION
    )
    spec = importlib.util.spec_from_file_location("deepseek_v41_encoding", encoder_path)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    source = load_dataset(
        DATASET_REPO, split="train", revision=revision, streaming=True
    )
    records, skipped, scanned = [], 0, 0
    for row in source:
        scanned += 1
        prompt, label = row.get("problem_statement"), row.get("patch")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(label, str)
            or not label.strip()
        ):
            skipped += 1
            continue
        messages = [{"role": "user", "content": prompt}]
        encoded = encoder.encode_messages(messages, thinking_mode="thinking")
        token_count = len(tokenizer.encode(encoded, add_special_tokens=False))
        if token_count > PROMPT_LIMIT:
            skipped += 1
            continue
        records.append(
            {
                "prompt": encoded,
                "label": label,
                "metadata": {
                    "instance_id": row.get("instance_id"),
                    "prompt_tokens": token_count,
                },
            }
        )
        if len(records) == 72:
            break
        if scanned >= 10000:
            break
    if len(records) != 72:
        raise ValueError(
            f"Expected 72 nonempty, fitting SWE-smith examples; found {len(records)}"
        )
    print(
        json.dumps(
            {
                "dataset_revision": revision,
                "scanned": scanned,
                "skipped": skipped,
                "train_rows": 64,
                "eval_rows": 8,
                "max_prompt_tokens": max(
                    r["metadata"]["prompt_tokens"] for r in records
                ),
            }
        ),
        flush=True,
    )
    return ValidationPrompts(records[:64], revision), ValidationPrompts(
        records[64:], revision
    )


def validation_image(image, *, data_files):
    for local, remote in data_files:
        image = image.add_local_file(local, remote_path=remote, copy=True)
    return image


async def fill_context(args, sample, sampling_params, *, evaluation=False):
    """Force training responses to the context limit; keep evaluation natural."""
    from miles.rollout.sglang_rollout import GenerateState, generate

    if not evaluation:
        tokenizer = GenerateState(args).tokenizer
        prompt_tokens = len(tokenizer.encode(sample.prompt, add_special_tokens=False))
        remaining = (
            args.rollout_max_context_len - SGLANG_RESERVED_TOKENS - prompt_tokens
        )
        if remaining <= 0:
            raise ValueError("Prompt leaves no response budget")
        sampling_params = dict(
            sampling_params,
            max_new_tokens=remaining,
            ignore_eos=True,
            stop=[],
            stop_token_ids=[],
        )
    result = await generate(args, sample, sampling_params, evaluation=evaluation)
    if not evaluation and result.status != result.Status.ABORTED:
        expected = args.rollout_max_context_len - SGLANG_RESERVED_TOKENS
        if len(result.tokens) != expected:
            raise ValueError(
                f"Context stress sample has {len(result.tokens)} tokens; "
                f"expected {expected} (two scheduler-reserved tokens)"
            )
    return result


def build_config(dataset, eval_dataset, *, steps=5, overrides=None):
    base = DeepSeek_V4_1_Flash_LoRA_Recipe()
    recipe = replace(
        base,
        name="deepseek-v41-lora-swebench-32k",
        num_rollout=steps,
        rm_type="f1",
        custom_generate_function=fill_context,
        eval_interval=4,
        save_interval=steps,
        n_samples_per_eval_prompt=2,
        eval_max_response_len=4096,
        skip_eval_before_train=False,
        # Keep real, full-context batches for trainer-only diagnosis if a
        # validation fails after expensive generation. Each launch is isolated.
        extra_config={
            **base.extra_config,
            "save_debug_rollout_data": (
                f"/checkpoints/validation-{uuid.uuid4().hex}/rollouts/{{rollout_id}}.pt"
            ),
        },
        environment={
            **base.environment,
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
        },
        image_overlay=partial(
            validation_image,
            data_files=(
                (dataset.local_path, dataset.remote_path),
                (eval_dataset.local_path, eval_dataset.remote_path),
            ),
        ),
    )
    if overrides:
        recipe = replace(recipe, **overrides)
    return TrainConfig(
        model=DeepSeek_V4_1_Flash_LoRA(),
        dataset=dataset,
        eval_dataset=eval_dataset,
        recipe=recipe,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--steps", type=int, choices=(1, 3, 5), default=5)
    parser.add_argument("--overrides", type=json.loads, default={})
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()
    dataset, eval_dataset = prepare_data()
    config = build_config(
        dataset, eval_dataset, steps=args.steps, overrides=args.overrides
    )
    config.recipe.cli_args(
        model=config.model, dataset=dataset, eval_dataset=eval_dataset
    )
    print(
        json.dumps(
            {
                "model": MODEL_REPO,
                "revision": MODEL_REVISION,
                "steps": args.steps,
                "gpu_type": "H200",
                "gpus": config.recipe.gpu_allocation.total_gpus,
                "context_length": CONTEXT_LENGTH,
                "response_policy": "fill usable context, ignore EOS for training only",
                "scheduler_reserved_tokens": SGLANG_RESERVED_TOKENS,
                "eval_response_cap": 4096,
                "reward": "patch token F1",
                "group_id": config.group_id,
            }
        ),
        flush=True,
    )
    if args.launch:
        run = config.launch(show_output=True)
        result = {
            "run_id": run.training_run_id,
            "modal_app_id": run.modal_app_id,
            "dataset_revision": dataset.revision,
            "steps": args.steps,
        }
        if args.result_file:
            args.result_file.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
