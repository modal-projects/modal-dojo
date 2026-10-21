# ---
# order: 11
# ---
#
# # Cookbook RL on Spindle
#
# Run a raw Tinker Cookbook RL config against a Spindle backend that Dojo
# deploys for the duration of training. The cookbook driver runs locally;
# Spindle runs training and sampling on Modal. This example uses Qwen3.5-2B,
# one H100 trainer and one H100 sampler, and defaults to a one-step proof.
#
# Install with Python 3.12: `uv sync --extra spindle`.
# Configure a Modal environment with `spindle-proxy`
# (`MODAL_PROXY_TOKEN_ID`, `MODAL_PROXY_TOKEN_SECRET`). For gated models or
# authenticated Hub downloads, add `huggingface-secret` (`HF_TOKEN`). The proxy
# token must be allowed in that environment. See Spindle's server setup:
# https://github.com/modal-projects/spindle#set-up-the-server
# Dojo generates a separate API key for each run; no Tinker account is needed.
#
# Run `uv run --extra spindle tutorials/spindle_rl.py --environment YOUR_ENV`.
# After the proof, use `--steps 10` for a short smoke test. Use a fresh log path
# for each run; v0 cannot resume checkpoints across owned Spindle deployments.
#
# Each raw trajectory group is printed and saved to `dojo-trajectories.jsonl`
# under the log path, including token IDs, log probabilities, rewards, and
# decoded prompts/responses. Cookbook retains its normal logs/checkpoints.
# V0 does not create a Dojo TrainingRun or post trajectories to the dashboard.
# Spindle's LoRA API uses deployed apps: Dojo gives each run unique names and
# stops them on exit, failure, or cancellation. A hard process kill can leave
# these apps deployed; use the printed Modal app IDs to stop them. Saved
# checkpoints and shared model caches remain on their configured volumes.

import argparse
import asyncio
import tempfile

from spindle.configs.qwen35_2b_lora_32k import Config as Qwen2BConfig
from tinker_cookbook.recipes.math_rl.arithmetic_env import ArithmeticDatasetBuilder
from tinker_cookbook.rl import train as rl_train

from modal_dojo import SpindleRecipe

MODEL_NAME = "Qwen/Qwen3.5-2B"


def build_recipe(*, steps=1, log_path=None, environment=None, proxy_secret="spindle-proxy"):
    deploy_config = Qwen2BConfig(
        trainer_gpus_per_node=1,
        trainer_max_clients_per_instance=1,
        inference_max_replicas=1,
        overrides={"miles_cfg.max_lora_slots": 1},
    )
    deploy_config.platform["modal"]["environment"] = environment
    deploy_config.platform["secrets"]["sampler_proxy"] = proxy_secret
    rl_config = rl_train.Config(
        log_path=log_path or tempfile.mkdtemp(prefix="dojo-spindle-"),
        model_name=MODEL_NAME,
        renderer_name="qwen3_5_disable_thinking",
        recipe_name="tutorial_spindle_rl",
        dataset_builder=ArithmeticDatasetBuilder(
            model_name_for_tokenizer=MODEL_NAME,
            renderer_name="qwen3_5_disable_thinking",
            batch_size=4,
            n_batches=20,
            group_size=4,
        ),
        learning_rate=1e-5,
        max_tokens=64,
        lora_rank=32,
        loss_fn="importance_sampling",
        eval_every=5,
        save_every=5,
        max_steps=steps,
    )
    return SpindleRecipe(rl_config, deploy_config=deploy_config)


# Existing Dojo datasets can opt in with `to_dataset_builder`. Conversion is
# explicit because rows alone cannot specify an RL environment and reward:
#
# ```python
# from modal_dojo import DojoDatasetBuilder, HuggingFaceDataset
#
# dataset = HuggingFaceDataset(
#     hf_repo="your-org/qa", input_column="question", output_column="answer",
# )
# builder = DojoDatasetBuilder.from_dataset(
#     dataset,
#     model_name=MODEL_NAME,
#     renderer_name="qwen3_5_disable_thinking",
#     batch_size=4,
#     group_size=4,
#     reward_fn=lambda response, label: float(response.strip() == str(label).strip()),
# )
# ```
#
# Pass `builder` as the cookbook config's `dataset_builder`. Hugging Face
# supports text/messages and requires an explicit reward function. Harbor
# converts the selected task split into cookbook tool environments and uses
# Harbor grading. Custom datasets must implement `to_dataset_builder`; the
# base implementation raises at configuration time, before GPU allocation.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--log-path")
    parser.add_argument("--environment")
    parser.add_argument("--proxy-secret", default="spindle-proxy")
    args = parser.parse_args()
    recipe = build_recipe(
        steps=args.steps, log_path=args.log_path,
        environment=args.environment, proxy_secret=args.proxy_secret,
    )
    print(f"Logs and trajectories: {recipe.rl_config.log_path}")
    asyncio.run(recipe.train())
