# ---
# order: 11
# deps: mujoco, numpy, pillow
# ---
#
# # Pick and place with a vision-language-action model
#
# Vision-language-action (VLA) models take in what a robot sees and what it's
# told to do, and output what it should do next. In this tutorial, we teach
# [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) to act as one.
# Given an overhead image of a table and an instruction like "Put the red block
# in the blue tray", it answers with where to pick and where to place:
#
# ```text
# PICK(412, 268) PLACE(731, 590)
# ```
#
# We then carry out that command in a [MuJoCo](https://mujoco.org) physics
# simulation, and the model only gets full reward if the block actually lands in
# the right tray.
#
# <video controls autoplay muted loop style="display: block; margin: 0 auto;">
# <source src="/pick-and-place-before-after.mp4" type="video/mp4">
# <source src="https://dojo.modal.dev/pick-and-place-before-after.mp4" type="video/mp4">
# </video>

import asyncio
import base64
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from modal_dojo import (
    CustomDeployment,
    MultimodalDataset,
    Qwen3_VL_8B,
    Qwen3_VL_8B_Recipe,
    TrainConfig,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim

model = Qwen3_VL_8B()

# ## Set up the simulator
#
# Each scene is a table with four colored blocks and three trays, all placed
# randomly from a seed. Why a seed? Our dataset only has to store a number, and
# the reward function can rebuild exactly the same scene later to check the
# model's answer. The extra blocks and trays are distractors, so the model has to
# actually read the instruction to know what to do.
#
# To carry out a command, the simulator looks straight down at the pick point
# and, if there's a block there, releases it 10 cm above the place point. Physics
# takes over from there, so a block that lands on a tray wall can still bounce
# out. All of this lives in
# [`sim.py`](https://github.com/modal-projects/modal-dojo/blob/main/tutorials/pick_and_place/sim.py)
# so we can focus on training.
#
# ## Define a scoring function
#
# How should we score a command? We could return 1 if the right block ends up in
# the right tray and 0 otherwise, but then grabbing the wrong block looks no
# different from grabbing the right one and dropping it just short. Instead, we
# give 0.25 for picking up the right block, so the model gets credit for reading
# the instruction correctly even before its placement is accurate.


def score_response(response: str, seed: int) -> tuple[float, bytes | None]:
    command = sim.parse_command(response)
    if command is None:
        return 0.0, None
    result = sim.execute(seed, command)
    if result["success"]:
        return 1.0, result["image"]
    if result["picked"] == sim.sample_scene(seed)["target_block"]:
        return 0.25, result["image"]
    return 0.0, result["image"]


# ## Get the dataset
#
# Rather than downloading a dataset, we generate one. Each row holds a rendered
# overhead image, the instruction, and the scene's seed as the label. Note that
# there are no answers: the simulator is the grader! We train on seeds 0 to 2047
# and hold out 100 scenes from a separate range for evaluation.

PROMPT = (
    "<image>\n"
    "You control a robot with a suction gripper. The image is an overhead view "
    "of the table.\n"
    "Instruction: {instruction}\n"
    "Respond with only a command of the form PICK(x, y) PLACE(x, y), where (x, y) "
    "are image coordinates from 0 to 1000, x from left to right and y from top to "
    "bottom. Pick at the center of the block and place over the center of the tray."
)


class PickAndPlaceDataset(MultimodalDataset):
    def __init__(self, seeds):
        self.seeds = list(seeds)
        super().__init__(modality="image")

    def source_rows(self):
        for seed in self.seeds:
            instruction = sim.sample_scene(seed)["instruction"]
            image = base64.b64encode(sim.overhead_image(seed)).decode("ascii")
            yield {
                "prompt": PROMPT.format(instruction=instruction),
                "media": f"data:image/png;base64,{image}",
                "label": str(seed),
            }


train_dataset = PickAndPlaceDataset(range(2048))
eval_dataset = PickAndPlaceDataset(range(100_000, 100_100))

# ## Deploy the base model
#
# We serve the base model with a
# [CustomDeployment](https://dojo.modal.dev/reference/customdeployment) to get a
# baseline for performance.


def deploy_base_model():
    print("deploying base model deployment...")
    base_deployment = CustomDeployment.launch(
        model,
        unauthenticated=True,
        app_name="pick-and-place-baseline",
    )
    base_deployment.wait_until_ready(timeout=3000)
    print(f"base model deployed to {base_deployment.url}")
    return base_deployment


# ## Evaluate the base model
#
# For RL to work, the base model needs to succeed some of the time, but not
# always. Why? GRPO learns by comparing several answers to the same prompt, and
# if they all get the same reward, there's nothing to compare. If the base model
# is near 0% or 100%, lower or raise `sim.N_BLOCKS` and `sim.N_TRAYS` to make the
# task easier or harder.


def run_eval(deployment, max_concurrency: int = 8) -> float:
    def _score_one(row):
        content = [
            {"type": "text", "text": row["prompt"].replace("<image>", "").strip()},
            {"type": "image_url", "image_url": {"url": row["images"][0]}},
        ]
        message = deployment.chat([{"role": "user", "content": content}])
        response = model.parse_response(message.get("content") or "").content
        reward, _ = score_response(response, int(row["label"]))
        return reward == 1.0

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        successes = list(executor.map(_score_one, eval_dataset.rows()))
    return sum(successes) / len(successes)


def run_baseline_evals(deployment):
    print("running base model evaluation...")
    print(f"success rate: {run_eval(deployment):.1%}")


# ## Creating a reward function
#
# To make our scoring function a reward function, we parse the model's response
# and score it against the scene in the label. The simulation is CPU-bound, so we
# run it in a thread to keep the event loop free. We also attach the final
# overhead frame to each sample, so you can see where every block ended up in
# the dashboard's rollout view.


async def pick_and_place_rm(args, sample, **kwargs) -> float:
    response = model.parse_response(sample.response or "").content
    reward, image = await asyncio.to_thread(
        score_response, response, int(sample.label)
    )
    if image is not None:
        data_uri = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
        sample.metadata = {**(sample.metadata or {}), "image": data_uri}
    return reward


# ## Begin training
#
# Since the reward function runs MuJoCo inside the training containers, we extend
# the training image with an `image_overlay` that installs MuJoCo and OSMesa (for
# rendering without a display) and copies in `sim.py`. Each step samples 8 scenes
# with 8 commands each, and we train for 200 steps.
#
# ![Reward curve](/pick-and-place-reward.webp)
#
# The training reward climbs quickly and then saturates. Since it includes the
# partial credit for picking the right block, the held-out eval below is a better
# measure of how often the model actually succeeds.


def pick_and_place_image(image):
    return (
        image.apt_install("libosmesa6")
        .uv_pip_install("mujoco", "numpy", "pillow")
        .add_local_file(sim.__file__, remote_path="/root/sim.py", copy=True)
    )


config = TrainConfig(
    model=model,
    dataset=train_dataset,
    recipe=Qwen3_VL_8B_Recipe(
        actor_num_gpus_per_node=8,
        tensor_model_parallel_size=4,
        num_rollout=200,
        save_interval=200,
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        global_batch_size=32,
        rollout_max_response_len=64,
        sglang_mem_fraction_static=0.75,
        custom_rm_function=pick_and_place_rm,
        no_save_optim=True,
        image_overlay=pick_and_place_image,
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


# ## Evaluate the trained checkpoint
#
# Let's run the same eval on the trained checkpoint.


def deploy_trained_model(checkpoint):
    print("deploying trained model deployment...")
    trained_deployment = CustomDeployment.launch(
        model,
        checkpoint,
        unauthenticated=True,
        app_name="pick-and-place-trained",
    )
    trained_deployment.wait_until_ready(timeout=3000)
    print(f"checkpoint deployed to {trained_deployment.url}")
    return trained_deployment


def run_trained_evals(trained_deployment):
    print("running checkpoint evaluation...")
    print(f"success rate: {run_eval(trained_deployment):.1%}")


if __name__ == "__main__":
    base_deployment = deploy_base_model()
    run_baseline_evals(base_deployment)
    checkpoint = train(config)
    trained_deployment = deploy_trained_model(checkpoint)
    run_trained_evals(trained_deployment)

# ## Results
#
# | Model | Success rate |
# | --- | ---: |
# | Base | 12% |
# | Trained | 88% |
#
# After 200 steps, the model succeeds on 88% of scenes it never saw during
# training, up from 12%.
