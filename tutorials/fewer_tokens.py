# ---
# order: 5
# ---
#
# # Use fewer tokens without lobotomizing your model
#
# When billed per token, the price of using any LLM depends on the cost per token
# and the total number of tokens. When using an open-source model, you can take
# control over the cost per token by
# [owning your inference](https://modal.com/blog/introducing-auto-endpoints).
# However, open-source models are notorious for using significantly more tokens
# per request.
#
# Here, we train [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
# to answer problems from [GSM8k](https://huggingface.co/datasets/skrishna/gsm8k_only_answer),
# which it can already do pretty well, but using fewer output tokens.

import re
import time
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

from modal_training_gym import (
    Endpoint,
    HuggingFaceDataset,
    Qwen3_5_4B,
    Qwen3_5_4B_Recipe,
    TrainConfig,
)

# ## Get a baseline
#
# We'll see how the model performs on the task, and how many tokens it reqiures to do so.

model = Qwen3_5_4B()


def deploy_base_model():
    base_deployment = Endpoint.launch(model, unauthenticated=True)
    try:
        base_deployment.wait_until_ready()
    except BaseException:
        base_deployment.stop()
        raise
    print(f"base model deployed to {base_deployment.url}")
    return base_deployment


def _extract_gsm8k_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1]
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def score_gsm8k(response: str, label: str) -> float:
    pred = _extract_gsm8k_answer(response)
    try:
        return float(float(pred.replace(",", "")) == float(label))
    except ValueError:
        return 0.0


train_dataset = HuggingFaceDataset(
    "skrishna/gsm8k_only_answer",
    hf_split="train[:200]",
    input_column="text",
    output_column="label",
    input_format="text",
)

eval_dataset = HuggingFaceDataset(
    "skrishna/gsm8k_only_answer",
    hf_split="train[200:400]",
    input_column="text",
    output_column="label",
    input_format="text",
)


def run_eval(deployment, max_concurrency: int = 16) -> tuple[float, float]:
    deployment.wait_until_ready()
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)

    def _score_one(example):
        msg = deployment.chat(
            example[eval_dataset.input_key()],
            chat_template_kwargs={"enable_thinking": True},
            max_tokens=8192,
        )
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        correct = score_gsm8k(content, example[eval_dataset.label_key()])
        tokens = tokenizer.encode(reasoning + content, add_special_tokens=False)
        return correct, len(tokens)

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        rows = list(executor.map(_score_one, eval_dataset.rows()))
    if not rows:
        return float("nan"), float("nan")
    accuracy = sum(c for c, _ in rows) / len(rows)
    mean_tokens = sum(n for _, n in rows) / len(rows)
    return accuracy, mean_tokens


def run_baseline_evals(deployment):
    print("running base model evaluation...")
    accuracy, mean_tokens = run_eval(deployment)
    print(f"percent correct: {accuracy:.1%}")
    print(f"mean output tokens: {mean_tokens:.0f}")
    return accuracy, mean_tokens


# ## Defining the reward function
#
# We score based on both correctness and an
# [ALP](https://arxiv.org/abs/2506.05256)-style hybrid length penalty. For each
# group, let $p$ be the group solve rate, $\ell_i$ the sample's ``response_length``,
# $L=$ ``rollout_max_response_len``, and $c_i\in\{0,1\}$ correctness. Then
#
# $$
# r_i = c_i \left(1 - p \dfrac{\ell_i}{L}\right)
# $$


async def gsm8k_correctness_rm(args, sample, **kwargs) -> float:
    if sample.status == sample.Status.TRUNCATED:
        return 0.0
    response = model.parse_response(sample.response or "")
    return score_gsm8k(response.content, sample.label)


def density_post_process(args, samples, **kwargs):
    n = args.n_samples_per_prompt
    max_len = args.rollout_max_response_len
    raw, processed = [], []
    for start in range(0, len(samples), n):
        group = samples[start : start + n]
        solve_rate = sum(bool(s.reward) for s in group) / len(group)
        rewards = [
            1.0 - solve_rate * s.response_length / max_len if s.reward else 0.0
            for s in group
        ]
        mean = sum(rewards) / len(rewards)
        std = (sum((r - mean) ** 2 for r in rewards) / max(len(rewards) - 1, 1)) ** 0.5
        raw += rewards
        processed += [(r - mean) / (std + 1e-6) for r in rewards]
    return raw, processed


# ## Train
#
# Sit back and watch it rip.


def deploy_trained_model(checkpoint):
    trained_deployment = Endpoint.launch(model, checkpoint, unauthenticated=True)
    try:
        trained_deployment.wait_until_ready()
    except BaseException:
        trained_deployment.stop()
        raise
    print(f"checkpoint deployed to {trained_deployment.url}")
    return trained_deployment


def run_trained_evals(trained_deployment):
    print("running checkpoint evaluation...")
    accuracy, mean_tokens = run_eval(trained_deployment)
    print(f"percent correct: {accuracy:.1%}")
    print(f"mean output tokens: {mean_tokens:.0f}")
    return accuracy, mean_tokens


def eval_checkpoint(checkpoint):
    deployment = deploy_trained_model(checkpoint)
    try:
        step = int(checkpoint.name.removeprefix("iter_")) + 1
        return (f"Step {step}", *run_trained_evals(deployment))
    finally:
        deployment.stop()


config = TrainConfig(
    model=model,
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    recipe=Qwen3_5_4B_Recipe(
        num_rollout=10,
        rollout_batch_size=16,
        n_samples_per_prompt=8,
        global_batch_size=16,
        rollout_max_response_len=8192,
        save_interval=1,
        apply_chat_template_kwargs='{"enable_thinking": true}',
        custom_rm_function=gsm8k_correctness_rm,
        custom_reward_post_process_function=density_post_process,
    ),
)


def train(config):
    with config.launch() as run, ThreadPoolExecutor() as evals:
        print(f"run id: {run.training_run_id}")
        checkpoint, pending = None, []
        while True:
            done = run.done()
            latest = run.latest_checkpoint()
            if latest is not None and latest != checkpoint:
                checkpoint = latest
                print(f"new checkpoint: {checkpoint.path}")
                pending.append(evals.submit(eval_checkpoint, checkpoint))
            if done:
                break
            time.sleep(30)
        results = [f.result() for f in pending]
        if run.status.value != "completed":
            raise RuntimeError(
                run.error or f"training ended with status {run.status.value}"
            )
        return results


def print_results(rows):
    print("| | Accuracy | Mean output tokens |")
    print("| --- | ---: | ---: |")
    for label, accuracy, mean_tokens in rows:
        print(f"| {label} | {accuracy:.1%} | {mean_tokens:.0f} |")


if __name__ == "__main__":
    base_deployment = deploy_base_model()
    try:
        rows = [("Baseline", *run_baseline_evals(base_deployment))]
    finally:
        base_deployment.stop()
    rows.extend(train(config))
    print_results(rows)

# ## Results
#
# | | Accuracy | Mean output tokens |
# | --- | ---: | ---: |
# | Baseline | 67.0% | 4272 |
# | Step 1 | 71.5% | 3878 |
# | Step 2 | 72.0% | 3598 |
# | Step 3 | 80.0% | 2866 |
# | Step 4 | 83.5% | 2442 |
# | Step 5 | 82.0% | 2244 |
# | Step 6 | 83.5% | 1961 |
# | Step 7 | 85.5% | 1662 |
# | Step 8 | 85.0% | 1455 |
# | Step 9 | 86.5% | 1307 |
# | Step 10 | 85.5% | 1136 |
