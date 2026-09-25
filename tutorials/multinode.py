# ---
# order: 10
# ---
#
# # Frontier-scale training
#
# When you decide you need a frontier-scale model for your workload, you want
# the biggest hardware you can get, and runs that stay up. This tutorial trains
# [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) on 4 trainer nodes with 8
# B300s each plus 8 disaggregated rollout GPUs, using full-weight
# [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) (GSPO)
# on [Tongyi-Zhiwen/DocQA-RL-1.6K](https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K).

import math
import re
import string
import time

from datasets import load_dataset

from modal_dojo import (
    DatasetConfig,
    Endpoint,
    GLM_4_7,
    TrainConfig,
)
from modal_dojo.train_recipes.slime_recipe import GLM_4_7_Recipe

# ## Set up
#
# The model and recipe classes already contain most of the presets that you'd
# care about, but we print a few parameters you may find interesting.

_ANSWER_TOKENS = 2048


def extract_answer(response: str) -> str | None:
    text = response.replace("*", "")
    match = re.search(r"The correct answer is \(?([A-D])\)?", text)
    if match:
        return match.group(1)
    match = re.search(r"Therefore, the answer is\s*([^\n]+)", text)
    if not match:
        return None
    return re.split(r"\.(?:\s|$)", match.group(1).strip(), maxsplit=1)[0].strip()


def scored_answer(text: str) -> str:
    return extract_answer(text) or str(text).strip()


class DocQADataset(DatasetConfig):
    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> str:
        return "label"

    def rows(self):
        prompt_limit = GLM_4_7_Recipe.max_tokens_per_gpu - _ANSWER_TOKENS
        for row in load_dataset("Tongyi-Zhiwen/DocQA-RL-1.6K", split="train"):
            if int(row["extra_info"]["input_length"]) > prompt_limit:
                continue
            prompt = next(m["content"] for m in row["prompt"] if m["role"] == "user")
            gold = str(row["reward_model"]["ground_truth"])
            yield {
                "messages": [{"role": "user", "content": prompt}],
                "label": scored_answer(gold),
            }


model = GLM_4_7()
dataset = DocQADataset()


def _normalize(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation)).casefold().strip()


def answers_equal(pred: str, label: str) -> bool:
    try:
        return math.isclose(float(pred), float(label), rel_tol=0.0, abs_tol=1e-9)
    except ValueError:
        return _normalize(pred) == _normalize(label)


async def docqa_rm(args, sample, **kwargs) -> float:
    pred = scored_answer(model.parse_response(sample.response or "").content)
    return float(bool(pred) and answers_equal(pred, sample.label))


recipe = GLM_4_7_Recipe(
    num_rollout=3000,
    save_interval=10,
    rollout_batch_size=64,
    n_samples_per_prompt=8,
    global_batch_size=128,
    rollout_max_response_len=_ANSWER_TOKENS,
    custom_rm_function=docqa_rm,
)

print(f"training and rollout gpus colocated: {recipe.colocate}")
print(
    f"training nodes: {recipe.actor_num_nodes}, gpus/node: {recipe.actor_num_gpus_per_node}"
)
print(
    f"rollout gpus: {recipe.rollout_num_gpus}, rollout gpus/engine: {recipe.rollout_num_gpus_per_engine}"
)
print(
    f"parallelism: tp={recipe.tensor_model_parallel_size}, pp={recipe.pipeline_model_parallel_size}, "
    f"cp={recipe.context_parallel_size}, ep={recipe.expert_model_parallel_size}"
)
print(f"optimizer cpu offload: {recipe.optimizer_cpu_offload}")

# ## Kick off training

config = TrainConfig(
    model=model,
    dataset=dataset,
    recipe=recipe,
)


def train(config):
    with config.launch() as run:
        print(f"run id: {run.training_run_id}")
        checkpoint = None
        while True:
            done = run.done()
            latest = run.latest_checkpoint()
            if latest is not None and latest != checkpoint:
                checkpoint = latest
                print(f"new checkpoint: {checkpoint.path}")
            if done:
                break
            time.sleep(30)
        if checkpoint is None:
            raise RuntimeError("run produced no checkpoint")
        print(f"checkpoint: {checkpoint.path}")
    return checkpoint


# ## Test out the trained model


def deploy_trained_model(checkpoint):
    print("deploying trained model endpoint...")
    trained_deployment = Endpoint.launch(
        model, checkpoint, unauthenticated=True, recreate_if_existing=True
    )
    trained_deployment.wait_until_ready(timeout=45 * 60)
    print(f"checkpoint deployed to {trained_deployment.url}")
    return trained_deployment


def run_trained_sample(trained_deployment):
    example = next(iter(dataset.rows()))
    msg = trained_deployment.chat(example[dataset.input_key()])
    print(msg.get("content") or msg.get("reasoning_content") or "")


if __name__ == "__main__":
    checkpoint = train(config)
    trained_deployment = deploy_trained_model(checkpoint)
    run_trained_sample(trained_deployment)
