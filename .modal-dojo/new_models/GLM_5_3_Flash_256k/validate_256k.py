"""256k capacity probe for GLM_5_3_Flash_LoRA_256k_Recipe.

Each DAPO-math prompt gets filler prepended in the user turn so the prompt is
~TARGET tokens; the model then answers the real question (dapo "Answer:" reward).
Filler is prompt (no loss) but still runs through 256k prefill and 256k
Megatron fwd/bwd/logprob.

  uv run python validate_256k.py --steps 1            # dry: print config
  uv run python validate_256k.py --steps 1 --launch
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from modal_dojo.common.dataset import DatasetConfig

CONTEXT = 262144
RESPONSE_BUDGET = 8192


def report_memory(stage):
    import torch

    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    print(
        "GLM256K_MEMORY "
        + json.dumps(
            {
                "stage": stage,
                "rank": torch.distributed.get_rank(),
                "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                "device_free_gib": free / 2**30,
                "device_total_gib": total / 2**30,
            }
        ),
        flush=True,
    )


async def dapo_rm(args, sample, **kwargs) -> float:
    from miles.rollout.rm_hub.math_dapo_utils import compute_score

    response = sample.response.split("<|user|>")[0]
    return compute_score(response, sample.label)["score"]


def before_log_prob(args, model, store_prefix):
    report_memory("before_log_prob:" + store_prefix)


def _oom_summary(device, alloc, device_alloc, device_free):
    import collections

    import torch

    groups = collections.defaultdict(lambda: [0, 0])
    for seg in torch.cuda.memory._snapshot()["segments"]:
        for block in seg["blocks"]:
            if block["state"] != "active_allocated":
                continue
            frames = [
                f"{f['filename'].split('site-packages/')[-1]}:{f['line']}:{f['name']}"
                for f in block.get("frames", [])
                if "/torch/" not in f["filename"] and not f["filename"].startswith("<")
            ][:3]
            key = " <- ".join(frames) or "<before history>"
            groups[key][0] += block["size"]
            groups[key][1] += 1
    rank = torch.distributed.get_rank()
    print(f"GLM256K_OOM rank={rank} alloc_request_gib={alloc / 2**30:.2f}", flush=True)
    for key, (size, count) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:30]:
        print(
            f"GLM256K_OOM rank={rank} gib={size / 2**30:.2f} n={count} {key}",
            flush=True,
        )


def _install_backward_probes(model):
    import torch

    rank = torch.distributed.get_rank()

    def gib():
        return torch.cuda.memory_allocated() / 2**30

    def layer_pre(module, inputs):
        if torch.is_grad_enabled():
            print(
                f"GLM256K_BWD_MOE rank={rank} layer={module.layer_number} "
                f"allocated_gib={gib():.2f} peak_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}",
                flush=True,
            )

    def experts_pre(module, inputs):
        if torch.is_grad_enabled() and gib() > 60:
            print(
                f"GLM256K_BWD_CHUNK rank={rank} layer={module._glm256k_layer} "
                f"rows={inputs[0].shape[0]} allocated_gib={gib():.2f}",
                flush=True,
            )

    for chunk in model if isinstance(model, (list, tuple)) else [model]:
        for module in chunk.modules():
            if type(module).__name__ == "MoELayer":
                module.register_forward_pre_hook(layer_pre)
                module.experts._glm256k_layer = module.layer_number
                module.experts.register_forward_pre_hook(experts_pre)


_PROBES_INSTALLED = False


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    global _PROBES_INSTALLED
    report_memory(f"before_train:{rollout_id}:{step_id}")
    if not _PROBES_INSTALLED:
        import torch

        _PROBES_INSTALLED = True
        _install_backward_probes(model)
        torch.cuda.memory._record_memory_history(max_entries=200000, stacks="python")
        torch._C._cuda_attach_out_of_memory_observer(_oom_summary)


class LongContextMath(DatasetConfig):
    """Pads each question with other DAPO problems so filler routes like real text."""

    HEAD = "Unrelated reference problems follow; ignore them.\n\n"
    POOL = 4000

    def __init__(self, n_rows: int, target: int = CONTEXT - RESPONSE_BUDGET):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.3-Flash")
        ds = load_dataset(
            "zhuzilin/dapo-math-17k", split=f"train[:{n_rows + self.POOL}]"
        )
        pool = [self._problem(r) for r in ds.select(range(n_rows, len(ds)))]
        lens = [len(tok.encode(t + "\n\n", add_special_tokens=False)) for t in pool]
        self.n_rows = n_rows
        self.units = []
        for row in ds.select(range(n_rows)):
            text = tok.apply_chat_template(
                self._messages(row, []),
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="low",
            )
            budget = target - len(tok.encode(text, add_special_tokens=False))
            units = 0
            while units < len(lens) and budget > lens[units]:
                budget -= lens[units]
                units += 1
            self.units.append(units)

    @staticmethod
    def _problem(row) -> str:
        return row["prompt"][-1]["content"]

    @classmethod
    def _messages(cls, row, filler):
        messages = [dict(m) for m in row["prompt"]]
        user = messages[-1]
        user["content"] = (
            cls.HEAD
            + "".join(f + "\n\n" for f in filler)
            + "End of reference problems.\n\n"
            + user["content"]
        )
        return messages

    def input_key(self) -> str:
        return "prompt"

    def label_key(self) -> str:
        return "label"

    def rows(self):
        from datasets import load_dataset

        ds = load_dataset(
            "zhuzilin/dapo-math-17k", split=f"train[:{self.n_rows + self.POOL}]"
        )
        pool = [self._problem(r) for r in ds.select(range(self.n_rows, len(ds)))]
        for row, units in zip(ds.select(range(self.n_rows)), self.units, strict=True):
            yield {
                "prompt": self._messages(row, pool[:units]),
                "label": row["label"],
            }


def build_config(steps: int):
    from modal_dojo import (
        GLM_5_3_Flash_LoRA,
        GLM_5_3_Flash_LoRA_256k_Recipe,
        TrainConfig,
    )

    recipe = GLM_5_3_Flash_LoRA_256k_Recipe(
        num_rollout=steps,
        custom_rm_function=dapo_rm,
        apply_chat_template_kwargs={"reasoning_effort": "low"},
        custom_megatron_before_log_prob_hook=before_log_prob,
        custom_megatron_before_train_step_hook=before_train_step,
        capture_trace=True,
        trace_sample_limit=8,
        max_retries=0,
    )
    dataset = LongContextMath(recipe.rollout_batch_size * steps)
    return TrainConfig(model=GLM_5_3_Flash_LoRA(), recipe=recipe, dataset=dataset)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--launch", action="store_true")
    a = p.parse_args()
    config = build_config(a.steps)
    print(config.recipe.gpu_allocation.summary())
    if a.launch:
        run = config.launch()
        rec = {
            "run_id": run.training_run_id,
            "app_id": run.modal_app_id,
            "steps": a.steps,
            "launched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        out = Path(__file__).parent / "runs.jsonl"
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
