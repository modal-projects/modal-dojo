"""Launch a bounded Inkling 2 Small SWE proof with an existing verifier/harness.

Prepare JSONL with `prompt`, `label`, and optional `metadata` from your SWE
training split. Keep gold patches and hidden tests out of the prompt. The reward
callable must grade the generated patch in a fresh sandbox using the task's
tests. For multi-turn agents, also pass a generation callable that retains the
engine's exact tokens/logprobs and masks tool observations from policy loss.

Preview (no GPUs):
  uv run python -m configs.inkling_2_small_swe_256k --data tasks.jsonl \
    --reward my_swe_hooks:reward
Prove one step by adding --launch; after inspecting reward, traces and dashboard
timings, run --steps 10 --launch for the smoke. --generate my_swe_hooks:generate
selects a multi-turn rollout. The existing 8k math preset stays the default for
Inkling_2_Small; select this recipe explicitly.
This example reuses the base recipe's checkpoint Volume in the selected Modal
environment, avoiding another conversion when that checkpoint is already cached.

256k is the total token budget: 16k initial prompt plus up to 240k response.
In multi-turn rollouts, tool observations also occupy the total context budget.
This is a training configuration, not a SWE-bench leaderboard evaluation.
The backend is experimental until the full-length proof and smoke pass.
"""

import argparse
import importlib
import json
from pathlib import Path

from modal_dojo import (
    DatasetConfig,
    Inkling_2_Small,
    Inkling_2_Small_LoRA_256k_Recipe,
    TrainConfig,
)


class SWETasks(DatasetConfig):
    def __init__(self, rows):
        if not rows or any(not row.get("prompt") or "label" not in row for row in rows):
            raise ValueError(
                "Every SWE task needs a nonempty prompt and a verifier label"
            )
        self._rows = rows

    def input_key(self):
        return "prompt"

    def label_key(self):
        return "label"

    def rows(self):
        yield from self._rows


def _load_callable(path):
    module, name = path.split(":", 1)
    value = getattr(importlib.import_module(module), name)
    if not callable(value):
        raise ValueError(f"{path} is not callable")
    return value


def build_config(rows, reward, *, generate=None, steps=1):
    if steps not in (1, 10):
        raise ValueError("Select one proof step or ten smoke steps")
    return TrainConfig(
        model=Inkling_2_Small(),
        dataset=SWETasks(rows),
        recipe=Inkling_2_Small_LoRA_256k_Recipe(
            name="miles-inkling_2_small_lora_recipe",
            num_rollout=steps,
            custom_rm_function=reward,
            custom_generate_function=generate,
            # Measure the step before adding a potentially large checkpoint save.
            save=None,
            save_interval=None,
            skip_eval_before_train=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--reward", required=True, help="Verifier callable as module:function"
    )
    parser.add_argument(
        "--generate", help="Optional agent rollout callable as module:function"
    )
    parser.add_argument("--steps", type=int, choices=(1, 10), default=1)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.data.read_text().splitlines() if line.strip()
    ]
    config = build_config(
        rows,
        _load_callable(args.reward),
        generate=_load_callable(args.generate) if args.generate else None,
        steps=args.steps,
    )
    print(config.recipe.gpu_allocation.summary())
    print(
        f"{len(rows)} tasks; {args.steps} step(s); 16k prompt + 240k response; verifier={args.reward}"
    )
    if args.launch:
        run = config.launch()
        print(f"Run: {run.training_run_id}")


if __name__ == "__main__":
    main()
