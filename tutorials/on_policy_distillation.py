# ---
# order: 3
# deps: ifbench @ git+https://github.com/allenai/IFBench.git@fcd289db21d43aaa96c6d9291d32561cd6e19305
# ---
#
# # Efficient inference using on-policy distillation
#
# For workloads where you need the throughput and/or cost-savings of a smaller model,
# but the capabilities afforded by a larger model, on-policy distillation (OPD) is
# an effective and practical way to hit your targets. In this tutorial, we'll use
# [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) to teach the smaller
# [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) how to satisfy
# writing constraints from
# [allenai/IF_multi_constraints_upto5](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5).
# When we do offline evals, we'll use the held-out
# [allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test).
#
# <details>
# <summary>How does OPD work?</summary>
#
# In addition to the standard RL algorithm, during each rollout step,
# the student's response token IDs are sent to the teacher which returns
# per-token log-probabilities. This changes the advantage, the extra signal each token
# gets based on how much better or worse the response was than expected, as follows:
#
# $$
# A_t = A_t^{\text{GRPO}} - \lambda_{\text{opd}} \cdot (\log \pi_{\text{student}} - \log \pi_{\text{teacher}})
# $$
#
# The first term represents sparse rewards for correct answers,
# while the second term represents the dense signals that push towards the
# teacher's token-level distribution. Together, they teach the student
# what to say (i.e., getting the correct answer) and how to say it (i.e.,
# how the teacher would respond).
#
# </details>
#
# To do cross-family OPD (i.e., use a teacher from a different model family such as Deepseek), see
# [this tutorial](https://gym.modal.dev/tutorials/cross_tok_distill).

import ast
import json
import time

from datasets import load_dataset

from modal_training_gym import (
    CustomDeployment,
    DatasetConfig,
    Endpoint,
    Qwen3_5_4B,
    Qwen3_5_4B_Recipe,
    Qwen3_5_9B,
    TrainConfig,
)

# ## Deploy the base models
#
# First, we'll deploy the teacher and base models to derive a baseline.
# We can use an [Endpoint](https://modal.com/docs/guide/endpoints)
# to serve the student. However, for the teacher model, we need per-token logprobs,
# which are not currently supported by Endpoints when speculative decoding is
# enabled. So we instead use a
# [CustomDeployment](https://gym.modal.dev/reference/customdeployment)
# to serve the teacher.

student_model = Qwen3_5_4B()
teacher_model = Qwen3_5_9B()


def deploy_models():
    print("deploying base student endpoint...")
    base_student_deployment = Endpoint.launch(
        student_model, unauthenticated=True, recreate_if_existing=True
    )

    print("deploying teacher deployment...")
    teacher_deployment = CustomDeployment.launch(
        teacher_model,
        app_name="qwen3.5-9b-teacher",
        unauthenticated=True,
    )

    base_student_deployment.wait_until_ready()
    print(f"student base model deployed to {base_student_deployment.url}")

    teacher_deployment.wait_until_ready()
    print(f"teacher model deployed to {teacher_deployment.url}")

    return base_student_deployment, teacher_deployment


# ## Define a scoring function


def score_constraints(response: str, label: str) -> int:
    from ifbench import instructions_registry

    spec = json.loads(label)
    text = response or ""
    for instruction_id, raw_kwargs in zip(spec["instruction_id_list"], spec["kwargs"]):
        instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](
            instruction_id
        )
        kwargs = {k: v for k, v in (raw_kwargs or {}).items() if v is not None}
        instruction.build_description(**kwargs)
        if "prompt" in (instruction.get_instruction_args() or ()):
            instruction.build_description(prompt=spec["prompt"])
        if not (text.strip() and instruction.check_following(text)):
            return -1
    return 1


# ## Get the dataset
#
# As [this Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
# describes, using a small number of samples with many rollouts can be sufficient for OPD.
# Following suit, we'll only use 300 training samples.


def _known_if_instructions(row) -> bool:
    from ifbench import instructions_registry

    spec = ast.literal_eval(row["ground_truth"])[0]
    return all(
        instruction_id in instructions_registry.INSTRUCTION_DICT
        for instruction_id in spec["instruction_id"]
    )


class IFMultiConstraintsDataset(DatasetConfig):
    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> str:
        return "label"

    def rows(self):
        ds = load_dataset(
            "allenai/IF_multi_constraints_upto5", split="train", streaming=True
        )
        kept = 0
        for row in ds:
            if not _known_if_instructions(row):
                continue
            spec = ast.literal_eval(row["ground_truth"])[0]
            prompt = row["messages"][0]["content"]
            yield {
                "messages": [{"role": "user", "content": prompt}],
                "label": json.dumps(
                    {
                        "prompt": prompt,
                        "instruction_id_list": spec["instruction_id"],
                        "kwargs": spec["kwargs"],
                    }
                ),
            }
            kept += 1
            if kept >= 300:
                break


class IFBenchTestDataset(DatasetConfig):
    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> str:
        return "label"

    def rows(self):
        for row in load_dataset("allenai/IFBench_test", split="train"):
            yield {
                "messages": [{"role": "user", "content": row["prompt"]}],
                "label": json.dumps(
                    {k: row[k] for k in ("prompt", "instruction_id_list", "kwargs")}
                ),
            }


train_dataset = IFMultiConstraintsDataset()

eval_dataset = IFBenchTestDataset()

# ## Evaluate the base models
#
# First, we should check if our teacher is good enough to, well, be a teacher.
# Then, we'll see how the student fares in comparison to establish the gap we
# must close.


def run_eval(deployment, *, max_concurrency: int = 2) -> float:
    from concurrent.futures import ThreadPoolExecutor

    deployment.wait_until_ready()

    def _score_one(example):
        msg = deployment.chat(
            example[eval_dataset.input_key()],
            chat_template_kwargs={"enable_thinking": True},
        )
        response = msg.get("content") or msg.get("reasoning_content") or ""
        return score_constraints(response, example[eval_dataset.label_key()])

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        scores = list(executor.map(_score_one, eval_dataset.rows()))
    percent_correct = (
        len([s for s in scores if s == 1]) / len(scores) if scores else float("nan")
    )
    return percent_correct


def run_baseline_evals(base_student_deployment, teacher_deployment):
    print("running teacher base model evaluation...")
    teacher_correct = run_eval(teacher_deployment)
    print(f"percent correct: {teacher_correct:.1%}")

    print("running student base model evaluation...")
    base_student_correct = run_eval(base_student_deployment)
    print(f"percent correct: {base_student_correct:.1%}")


# ## Creating a reward function
#
# To use our scoring function for training, we have to modify some framework-specific
# functions. Luckily, this is relatively painless.
#
# <details>
# <summary>How OPD defines KL</summary>
#
# KL-divergence is defined as:
#
# $$
# D_{\mathrm{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
# $$
#
# where $P$ is the behavior distribution and $Q$ is the target distribution.
#
# Forward KL treats the teacher model as $P$ and the student model as $Q$. However, the
# $\log \frac{P(x)}{Q(x)}$ term would then be weighted by the teacher model's probability distribution $P$,
# resulting in high surprisal on modes unfamiliar to the student model.
#
# To counter this, OPD uses "reverse" KL divergence to grade the student model's output:
#
# $$
# D_{\mathrm{KL}}(\pi_{\mathrm{student}} \| \pi_{\mathrm{teacher}})
# $$
#
# where our student model is treated as the behavior distribution and
# the teacher model is our target distribution.
#
# When the teacher has high surprisal on a student mode, the term $\log(P(x)) - log(Q(x))$
# will yield a high positive KL divergence to penalize the student model.
# Now, the student model only gets penalized on modes relevant to itself.
#
# </details>
#
# <details>
# <summary>A fun exercise</summary>
#
# Try tweaking the composite reward signal by applying an integer coefficient to the
# binary integer reward signal used in the custom reward function to value correct answers
# over student-teacher alignment.
#
# </details>


async def ifbench_opd_rm(args, sample, **kwargs):
    from slime.rollout.on_policy_distillation import reward_func as _opd_reward

    teacher_response = await _opd_reward(args, sample, **kwargs)

    response = student_model.parse_response(sample.response)
    score = score_constraints(response.content, sample.label)
    sample.score = score
    if not isinstance(getattr(sample, "metadata", None), dict):
        sample.metadata = {}
    sample.metadata["shaped_reward"] = float(score)

    return teacher_response


def ifbench_opd_post_process(args, samples, **kwargs):
    from slime.rollout.on_policy_distillation import post_process_rewards as _opd_post

    _, _ = _opd_post(args, samples, **kwargs)

    rewards = [getattr(sample, "score", -1) for sample in samples]
    return rewards, rewards


# ## Start training
#
# In addition to the standard parameters we use for most tutorials,
# we set parameters such as `environment` and `extra_config` to supply
# framework-necessary environment variables and flags.

def build_config(teacher_deployment):
    return TrainConfig(
        model=student_model,
        dataset=train_dataset,
        recipe=Qwen3_5_4B_Recipe(
            num_rollout=10,
            save_interval=10,
            rollout_batch_size=16,
            n_samples_per_prompt=4,
            global_batch_size=16,
            rollout_max_response_len=2048,
            custom_rm_function=ifbench_opd_rm,
            custom_reward_post_process_function=ifbench_opd_post_process,
            apply_chat_template_kwargs={"enable_thinking": True},
            image_overlay=lambda img: img.pip_install(
                "ifbench @ git+https://github.com/allenai/IFBench.git@fcd289db21d43aaa96c6d9291d32561cd6e19305"
            ),
            environment={
                "PYTHONPATH": "/root/Megatron-LM/:/root",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": "1",
            },
            extra_config={
                "use_opd": True,
                "opd_type": "sglang",
                "opd_kl_coef": 1.0,
                "rm_url": f"{teacher_deployment.url}/generate",
            },
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


# ## Evaluate the trained student
#
# We'll deploy our trained student and compare it
# to our baseline evaluation from earlier.


def deploy_trained_model(checkpoint):
    print("deploying trained student endpoint...")
    trained_student_deployment = Endpoint.launch(
        student_model, checkpoint, unauthenticated=True, recreate_if_existing=True
    )
    trained_student_deployment.wait_until_ready()
    print(f"checkpoint deployed to {trained_student_deployment.url}")
    return trained_student_deployment


def run_trained_evals(trained_student_deployment):
    print("running student checkpoint evaluation...")
    trained_student_correct = run_eval(trained_student_deployment)
    print(f"percent correct: {trained_student_correct:.1%}")


if __name__ == "__main__":
    base_student_deployment, teacher_deployment = deploy_models()
    run_baseline_evals(base_student_deployment, teacher_deployment)
    config = build_config(teacher_deployment)
    checkpoint = train(config)
    trained_student_deployment = deploy_trained_model(checkpoint)
    run_trained_evals(trained_student_deployment)
