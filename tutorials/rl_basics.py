# ---
# order: 0
# deps: nltk
# ---
#
# # Getting started with RL
#
# This tutorial introduces some core features of the Modal Dojo by walking through
# a simple example of Reinforcement Learning with Verifiable Rewards (RLVR), a
# foundational method of RL post-training. Here, we teach
# [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
# how to write correct haikus. Step by step, we'll show the foundations of running
# training jobs on the Modal Dojo.

import nltk
from nltk.corpus import cmudict

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
# As with all training tasks, we need a baseline to decide how much training we need.
# To do that, we need a way to run inference on the base model so that we can try it out.
#
# Luckily, [Endpoints](https://modal.com/docs/guide/endpoints) allows us to easily deploy a
# production-ready LLM inference endpoint on Modal's managed infrastructure. It supports both
# open model weights in addition to custom fine tunes, sourced from either a Hugging Face repo or a
# [Modal Volume](https://modal.com/docs/guide/volumes). To use it, we provide a
# [class](https://gym.modal.dev/reference/endpoint) to instantiate one programatically.
#
# It will take a moment to download the model weights onto a Modal Volume and boot containers past the
# [cold-start](https://modal.com/docs/guide/cold-start#what-is-a-cold-start).
# Once you see the URL has been printed, you're ready to move on!

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
# To evaluate the base model, we need a function that takes as input a haiku and outputs a score
# (a.k.a. reward when we're training) to represent whether it follows the 5-7-5 syllable format.
# We can do that using NLTK's
# [CMU Pronouncing Dictionary](https://github.com/prosegrinder/python-cmudict),
#
# How should we define our scoring function? We could give it a score of 0 if it doesn't follow
# the format and 1 if it does, but that's not very informative for both the models being trained, and,
# more importantly, the human training the models! Instead, we want the score to provide sufficient
# granularity such that it's immediately obvious what the failure mode is (if any). Below, we implement
# the following function:
#
# - Return -10 if the model was so incompetent that it failed to return three lines.
# - Otherwise, return the negative sum of absolute differences between the predicted and target
# syllable count for each line.
#
# What does this mean? That the model will receive increasingly negative scores the further off
# its haiku is, with a maximum score of 0. Let's now see how it does.

_cmudict_cache = {}


def _get_cmudict() -> dict:
    if not _cmudict_cache:
        nltk.download("cmudict", quiet=True)
        _cmudict_cache.update(cmudict.dict())
    return _cmudict_cache


def _count_syllables(text: str) -> int:
    cmu = _get_cmudict()
    total = 0
    for word in re.findall(r"[a-zA-Z]+", text):
        phones = cmu.get(word.lower())
        if phones:
            total += sum(p[-1].isdigit() for p in phones[0])
        else:
            count = len(re.findall(r"[aeiouy]+", word.lower()))
            if word.lower().endswith("e") and count > 1:
                count -= 1
            total += max(count, 1)
    return total


def score_haiku(response: str) -> float:
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    if len(lines) != 3:
        return -10
    total_diff = sum(
        abs(_count_syllables(line) - target) for line, target in zip(lines, [5, 7, 5])
    )
    return -float(total_diff)


# ## Get the dataset
#
# Note that we've only qualitatively assessed its performance. Now, we should get concrete
# numbers. How do we do that? First, we'll have to curate a dataset. Luckily,
# [statworx/haiku](https://huggingface.co/datasets/statworx/haiku) from Huggingface
# already exists, so we don't have to create one ourselves.
#
# Note that for more complex tasks, it is almost certainly the case that you will be
# creating your own dataset. Why? Because the task you're trying to get your model
# to do is either too expensive or simply too hard for a bigger model. In either case,
# this is because the task is sufficiently out-of-distribution, and no existing dataset
# will serve your needs.
#
# See the [DatasetConfig](https://gym.modal.dev/reference/datasetconfig)
# documentation for a deeper dive.

train_dataset = HuggingFaceDataset(
    "statworx/haiku",
    hf_split="train[:10]",
    input_column="keywords",
    output_column="text",
    input_format="text",
    prompt_template="Write a haiku about {input}.",
    always_download=True,
)

eval_dataset = HuggingFaceDataset(
    "statworx/haiku",
    hf_split="train[10:15]",
    input_column="keywords",
    output_column="text",
    input_format="text",
    prompt_template="Write a haiku about {input}.",
    always_download=True,
)

# ## Evaluate the base model
#
# All we need to do now is, for each sample in our eval dataset,
# call the Endpoint, score each response, and calculate the mean.
# By default, Endpoints can process multiple inputs
# [concurrently](https://modal.com/docs/guide/servers#concurrency-and-autoscaling),
# so we loop over samples in parallel to speed up eval.


def run_eval(deployment, max_concurrency: int = 2) -> float:
    from concurrent.futures import ThreadPoolExecutor

    deployment.wait_until_ready()

    def _score_one(example):
        msg = deployment.chat(
            example[eval_dataset.input_key()],
            chat_template_kwargs={"enable_thinking": False},
        )
        return score_haiku(msg.get("content") or msg.get("reasoning_content") or "")

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        scores = list(executor.map(_score_one, eval_dataset.rows()))
    return sum(scores) / len(scores) if scores else float("nan")


def run_baseline_evals(deployment):
    print("running base model evaluation...")
    base_mean = run_eval(deployment)
    print(f"average score: {base_mean:.1f}")


# ## Creating a reward function
#
# To make our scoring function a reward function, we just need to extract the text from the
# model's response and pass it to our existing score_haiku. Simple enough.


async def haiku_rm(args, sample, **kwargs) -> float:
    response = model.parse_response(sample.response)
    return score_haiku(response.content)


# ## Train the model
#
# Finally, onto the training. The Modal Dojo supports both the
# [Slime](https://github.com/THUDM/slime) and
# [Miles](https://github.com/radixark/miles) frameworks.
# Here, we use Slime for demonstration purposes.
#
# Training is simple: pass in the model you intend to train,
# the dataset you wish to train on, and a recipe for how you want
# training to occur. The recipe wraps all
# [framework-native flags](https://thudm.github.io/slime/get_started/usage.html)
# in addition to providing Modal-specific ones.
#
# Once we run the code below, training kicks off and we'll immediately get a run ID, which we may
# use to watch the run's progress in the dashboard.

config = TrainConfig(
    model=model,
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    recipe=Qwen3_5_4B_Recipe(
        num_rollout=10,
        save_interval=10,
        rollout_batch_size=16,
        n_samples_per_prompt=8,
        global_batch_size=16,
        apply_chat_template_kwargs='{"enable_thinking": false}',
        custom_rm_function=haiku_rm,
        image_overlay=lambda image: image.run_commands(
            "uv pip install --system aiohttp 'nltk>=3.8.0'",
            "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
        ),
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


# ## Serve and evaluate the trained checkpoint
#
# We'll get the latest checkpoint and create a new Endpoint so we may evaluate it.
# Now, let's run the same eval as before.


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
    trained_mean = run_eval(trained_deployment)
    print(f"average score: {trained_mean:.1f}")


# ## Continue training off the checkpoint
# Hmm, it looks like the trained model is still not doing very well.
# A likely cause is that it only trained for 10 iterations.
# Let's continue training, starting from the last checkpoint.


def continue_training(checkpoint):
    new_config = TrainConfig(
        model=model,
        dataset=train_dataset,
        resume_from_checkpoint=checkpoint,
        recipe=Qwen3_5_4B_Recipe(
            custom_rm_function=haiku_rm,
            num_rollout=20,
            save_interval=20,
            rollout_batch_size=16,
            n_samples_per_prompt=8,
            global_batch_size=16,
            apply_chat_template_kwargs='{"enable_thinking": false}',
            image_overlay=lambda image: image.run_commands(
                "uv pip install --system aiohttp 'nltk>=3.8.0'",
                "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
            ),
        ),
    )

    with new_config.launch() as new_run:
        print(f"run id: {new_run.training_run_id}")
        new_checkpoint = None
        while True:
            done = new_run.done()
            latest = new_run.latest_checkpoint()
            if latest is not None and latest != new_checkpoint:
                new_checkpoint = latest
                print(f"new checkpoint: {new_checkpoint.path}")
            if done:
                break
            time.sleep(30)
        if new_checkpoint is None:
            raise RuntimeError("run produced no checkpoint")
        print(new_checkpoint.path)
    return new_checkpoint


# ## Evals Evals Evals
#
# Once again, we'll create a new Endpoint for the new checkpoint and run evals on it.


def deploy_continued_model(new_checkpoint):
    print("deploying continued model endpoint...")
    new_deployment = Endpoint.launch(
        model, new_checkpoint, unauthenticated=True, recreate_if_existing=True
    )
    new_deployment.wait_until_ready()
    print(f"new checkpoint deployed to {new_deployment.url}")
    return new_deployment


def run_continued_evals(new_deployment):
    print("running new checkpoint evaluation...")
    new_mean = run_eval(new_deployment)
    print(f"average score: {new_mean:.1f}")


if __name__ == "__main__":
    base_deployment = deploy_base_model()
    run_baseline_evals(base_deployment)
    checkpoint = train(config)
    trained_deployment = deploy_trained_model(checkpoint)
    run_trained_evals(trained_deployment)
    new_checkpoint = continue_training(checkpoint)
    new_deployment = deploy_continued_model(new_checkpoint)
    run_continued_evals(new_deployment)
