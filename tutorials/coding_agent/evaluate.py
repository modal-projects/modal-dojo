"""Evaluate pretrained or saved weights without any optimizer updates.

Run a four-task proof first, then use the same eval-800 protocol for both arms.
"""

import argparse
import json
from dataclasses import replace
from uuid import uuid4

from tutorials.coding_agent.main import DATA_ROOT, config


def build_config(*, checkpoint="", checkpoint_step=None, proof=False, concurrency=128):
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if checkpoint_step is not None and not checkpoint:
        raise ValueError("checkpoint_step requires a checkpoint directory")
    subset = "eval-4" if proof else "eval-800"
    extra = {**config.recipe.extra_config, "lr_decay_iters": 1,
             "agentic_eval_concurrency": min(concurrency, 4) if proof else concurrency}
    if checkpoint_step is not None:
        extra["ckpt_step"] = checkpoint_step
    recipe = replace(
        config.recipe,
        num_rollout=0,
        load=checkpoint,
        no_load_optim=True,
        save=None,
        save_interval=None,
        max_retries=0,
        # The proof uses one serving node. Full eval retains the
        # original 16 engines, with eight concurrent episodes per engine on average.
        rollout_num_gpus=8 if proof else 32,
        extra_config=extra,
        eval_config={
            "defaults": {"n_samples_per_eval_prompt": 1, "temperature": 0.6, "top_p": 1.0},
            "datasets": [{"name": subset, "path": str(DATA_ROOT / f"{subset}.jsonl")}],
        },
        save_debug_rollout_data=(
            f"/checkpoints/agentic_rollout_dumps/coding-agent-eval-{uuid4().hex}/rollout_{{rollout_id}}.pt"
        ),
    )
    return replace(config, recipe=recipe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="", help="Megatron checkpoint parent directory; omit for pretrained weights")
    parser.add_argument("--checkpoint-step", type=int, help="Saved Megatron iteration (29 is after 30 updates)")
    parser.add_argument("--proof", action="store_true")
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = build_config(checkpoint=args.checkpoint, checkpoint_step=args.checkpoint_step,
                       proof=args.proof, concurrency=args.concurrency)
    print(json.dumps({
        "checkpoint": args.checkpoint or "pretrained", "checkpoint_step": args.checkpoint_step,
        "optimizer_updates": cfg.recipe.num_rollout,
        "eval_config": cfg.recipe.eval_config,
        "eval_concurrency": cfg.recipe.extra_config["agentic_eval_concurrency"],
        "actor_gpus": 16, "rollout_gpus": cfg.recipe.rollout_num_gpus,
        "dump": cfg.recipe.save_debug_rollout_data,
    }, indent=2), flush=True)
    if not args.dry_run:
        run = cfg.launch()
        print(f"run id: {run.training_run_id}", flush=True)
        print(f"app id: {run.modal_app_id}", flush=True)


if __name__ == "__main__":
    main()
