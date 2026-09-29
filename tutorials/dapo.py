# ---
# order: 5
# ---
#
# # Long chain-of-thought reasoning for large-scale RL
#
# Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) is a modified version
# of GRPO that substantially improves long chain-of-thought (COT) reasoning,
# enabling large-scale RL for all. This tutorial trains
# [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) using DAPO on olympiad-style
# math problems sourced from the
# [zhuzilin/dapo-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)
# Huggingface dataset.
#
# You can read more about the algorithm in [the paper](https://arxiv.org/abs/2503.14476).

import re
import time

from modal_dojo import (
    Endpoint,
    HuggingFaceDataset,
    Qwen3_5_4B,
    Qwen3_5_4B_Recipe,
    TrainConfig,
)

# ## Deploy the base model
#
# We first deploy the base model with an
# [Endpoint](https://modal.com/docs/guide/endpoints)
# to get a baseline for performance.

model = Qwen3_5_4B()


def deploy_base_model():
    print("deploying base model endpoint...")
    base_deployment = Endpoint.launch(
        model, unauthenticated=True, recreate_if_existing=True
    )
    base_deployment.wait_until_ready()
    print(f"base model deployed to {base_deployment.url}")
    return base_deployment


# ## Define a scoring function
#
# Following the paper, we'll normalize as they do and return 1 for correct answers
# and -1 for incorrect answers. Although we'd like to give a more granular score
# for predictions, we can't simply use the numerical difference between a prediction
# and a ground-truth answer, since a numerically-close answer can be more wrong
# than one further away.

def _extract_answer(response: str) -> str:
    match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", response)
    return match[-1].strip() if match else "[INVALID]"

def _normalize_answer(answer: str) -> str:
    answer = str(answer).strip()
    answer = answer.split("=")[-1]
    for old, new in [("$", ""), ("\\$", ""), (",", ""), (" ", ""),
                      ("\\text{", ""), ("}", ""), ("\\boxed{", "")]:
        answer = answer.replace(old, new)
    return answer.strip()

def score_answer(response: str, label: str) -> int:
    pred = _normalize_answer(_extract_answer(response))
    gt = _normalize_answer(label)
    try:
        gt = str(int(float(gt)))
    except (ValueError, OverflowError):
        pass
    return 1 if pred == gt else -1

# ## Get the dataset
#
# Let's train on 2000 samples and hold out 100 for evaluation.

train_dataset = HuggingFaceDataset(
    "zhuzilin/dapo-math-17k",
    hf_split="train[:2000]",
    input_column="prompt",
    output_column="label",
    input_format="messages",
    always_download=True,
)

eval_dataset = HuggingFaceDataset(
    "zhuzilin/dapo-math-17k",
    hf_split="train[2000:2100]",
    input_column="prompt",
    output_column="label",
    input_format="messages",
    always_download=True,
)

# ## Evaluate the base model
#
# Let's get our baseline measure of performance.

def run_eval(deployment, max_concurrency: int = 2) -> float:
    from concurrent.futures import ThreadPoolExecutor

    deployment.wait_until_ready()

    def _score_one(example):
        msg = deployment.chat(
            example[eval_dataset.input_key()],
            chat_template_kwargs={"enable_thinking": True},
            max_tokens=8192,
        )
        response = msg.get("content") or msg.get("reasoning_content") or ""
        return score_answer(response, example[eval_dataset.label_key()])

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        scores = list(executor.map(_score_one, eval_dataset.rows()))
    percent_correct = (
        len([s for s in scores if s == 1]) / len(scores) if scores else float("nan")
    )
    return percent_correct


def run_baseline_evals(deployment):
    print("running base model evaluation...")
    base_mean = run_eval(deployment)
    print(f"percent correct: {base_mean:.1%}")

# ## Creating a reward function
#
# Our scoring function is good for evaluation, but we can improve the signal granularity
# during training. How? For example, DAPO adds a soft penalty for overlong responses,
# ensuring it learns to answer questions correctly and with fewer tokens.
#
# $$
# R_{\text{length}} =
# \begin{cases}
# 0 & \text{if } |y| \le L_{\text{max}} - L_{\text{cache}} \\
# \dfrac{(L_{\text{max}} - L_{\text{cache}}) - |y|}{L_{\text{cache}}} & \text{if } L_{\text{max}} - L_{\text{cache}} < |y| \le L_{\text{max}} \\
# -1 & \text{if } |y| > L_{\text{max}}
# \end{cases}
# $$
#
# For demonstration purposes, we set $L_\text{max}$ to the generation cap, 8192,
# and $L_\text{cache}$ to 2048, lower than the original values, 16384 and 4096,
# respectively. These values, however, do maintain the paper's 4:1 ratio.

async def dapo_overlong_rm(args, sample, **kwargs) -> float:
    response = model.parse_response(sample.response)
    base = score_answer(response.content, sample.label)

    L_max = args.rollout_max_response_len
    L_cache = 2048
    n = sample.response_length

    if n <= L_max - L_cache:
        length_penalty = 0.0
    elif n <= L_max:
        length_penalty = ((L_max - L_cache) - n) / L_cache
    else:
        length_penalty = -1.0

    return base + length_penalty

# ## Commence training
#
# In addition to model-specific parameters, the recipe below also includes
# [DAPO-specific](https://arxiv.org/abs/2503.14476)
# modifications. Those include:
#
# - Clip-Higher: `eps_clip=0.2`, `eps_clip_high=0.28`
# - No KL penalty: `use_kl_loss=False`
# - Token-level policy-gradient loss: `calculate_per_token_loss=True`
# - Dynamic sampling: `over_sampling_batch_size=48` and `dynamic_sampling_filter_path`
#
# Again, for demonstration purposes, we set `n_samples_per_prompt=8`.

config = TrainConfig(
    model=model,
    dataset=train_dataset,
    recipe=Qwen3_5_4B_Recipe(
        actor_num_gpus_per_node=8,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        num_rollout=15,
        save_interval=5,
        n_samples_per_prompt=8,
        rollout_batch_size=16,
        global_batch_size=32,
        rollout_max_response_len=8192,
        use_kl_loss=False,
        eps_clip=0.2,
        eps_clip_high=0.28,
        custom_rm_function=dapo_overlong_rm,
        apply_chat_template_kwargs='{"enable_thinking": true}',
        environment={
            "PYTHONPATH": "/root/Megatron-LM/:/root",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        },
        over_sampling_batch_size=48,
        dynamic_sampling_filter_path=(
            "slime.rollout.filter_hub.dynamic_sampling_filters."
            "check_reward_nonzero_std"
        ),
        balance_data=True,
        calculate_per_token_loss=True,
    ),
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

# ## Evaluate the trained model
#
# Let's run the same eval on the trained checkpoint.


def deploy_trained_model(checkpoint):
    print("deploying trained model endpoint...")
    trained_deployment = Endpoint.launch(
        model, checkpoint, unauthenticated=True, recreate_if_existing=True
    )
    trained_deployment.wait_until_ready()
    print(f"checkpoint deployed to {trained_deployment.url}")
    return trained_deployment


def run_trained_evals(trained_deployment):
    print("running checkpoint evaluation...")
    trained_correct = run_eval(trained_deployment)
    print(f"percent correct: {trained_correct:.1%}")


if __name__ == "__main__":
    base_deployment = deploy_base_model()
    run_baseline_evals(base_deployment)
    checkpoint = train(config)
    trained_deployment = deploy_trained_model(checkpoint)
    run_trained_evals(trained_deployment)
