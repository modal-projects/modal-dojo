# ---
# order: 8
# deps: pillow
# ---
#
# # GUI Grounding with Qwen3-VL-8B
#
# This tutorial trains **Qwen3-VL-8B-Instruct** via GRPO to predict click
# coordinates given a screenshot and a natural-language instruction like
# "click the Submit button".
#
# The task is simple: given an image of a GUI and an instruction identifying
# a UI element, output the normalized `(x, y)` center coordinate of that
# element. This is a foundational capability for computer-use agents.
#
# We use the [ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot)
# benchmark — a standard GUI grounding evaluation set covering iOS, Android,
# macOS, Windows, and Web screenshots with annotated bounding boxes.
#
# The reward is bbox-aware: a click that lands anywhere inside the target
# element scores +1 (a real click succeeds), and predictions that miss decay
# toward −1 over a margin scaled to the element's own size.

import re
import time

from modal_training_gym import (
    CustomDeployment,
    MultimodalDataset,
    Qwen3_VL_8B,
    Qwen3_VL_8B_Recipe,
    TrainConfig,
    WandbConfig,
)

# ## Dataset
#
# We use [rootsautomation/ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot)
# — ~1,200 GUI screenshots annotated with natural-language instructions and
# bounding boxes. Each row has:
#
# - `image` — a screenshot from iOS/Android/macOS/Windows/Web
# - `instruction` — e.g. "click the Submit button"
# - `bbox` — `[left, top, right, bottom]` in normalized [0, 1] coordinates
#
# We keep the full bounding box as the training target (a click anywhere
# inside it counts as a hit) and ask the model to output a single `(x, y)`
# click point.
#
# For this tutorial we train on 800 samples and hold out 200 for evaluation.

GROUNDING_PROMPT = (
    "<image>\n"
    "You are a GUI agent. Given the screenshot, click on the element "
    "described below.\n\n"
    "Instruction: {instruction}\n\n"
    "Respond with ONLY the normalized (x, y) coordinates of the click "
    "target, formatted as: (x, y)\n"
    "where x and y are decimals between 0 and 1 representing the "
    "horizontal and vertical position on the screen."
)


class ScreenSpotDataset(MultimodalDataset):
    """GUI grounding dataset from ScreenSpot."""

    hf_repo = "rootsautomation/ScreenSpot"
    hf_split = "test"

    def __init__(self, *, n_rows: int, row_offset: int = 0):
        self.n_rows = n_rows
        self.row_offset = row_offset
        super().__init__(modality="image")

    def source_rows(self):
        import base64
        import io

        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, split=self.hf_split)
        start = min(self.row_offset, len(ds))
        stop = min(start + self.n_rows, len(ds))
        # Demo-scale: inline base64 rows in memory; stream large corpora.
        for row in ds.select(range(start, stop)):
            left, top, right, bottom = row["bbox"]
            instruction = row["instruction"]

            buf = io.BytesIO()
            row["image"].save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            data_uri = f"data:image/png;base64,{img_b64}"

            yield {
                "prompt": GROUNDING_PROMPT.format(instruction=instruction),
                "media": data_uri,
                "label": f"{left:.4f},{top:.4f},{right:.4f},{bottom:.4f}",
            }


train_dataset = ScreenSpotDataset(n_rows=800)

eval_dataset = ScreenSpotDataset(n_rows=200, row_offset=800)

# ## Reward function
#
# Rather than measuring distance to the box's *center*, we reward whether the
# click would actually land on the element. Let `outside` be the Euclidean
# distance from the predicted point to the bounding box (`0` when the point is
# inside it), and `margin = max(diagonal_of_box, 0.05)`:
#
# ```text
# R = +1.0                          if outside == 0   (click inside element)
#   = 1.0 - 2.0 * outside / margin  if 0 < outside < margin
#   = -1.0                          if outside >= margin
# ```
#
# This fixes two problems with a center-distance reward:
#
# - **Click success is rewarded directly.** Any point inside the element gets
#   the full +1, even if it's far from the geometric center — exactly like a
#   real click.
# - **Tolerance scales with element size.** A fixed 5%-of-screen threshold is
#   too lenient on tiny icons (you can miss and still score) and too harsh on
#   big buttons (a valid click gets penalized). Scaling the falloff to the
#   element's own diagonal (with a 5% floor for tiny targets) avoids both.
#
# The model also gets −1 if it fails to output parseable coordinates.


def _parse_coordinates(text: str) -> tuple[float, float] | None:
    """Extract (x, y) from model output like '(0.45, 0.32)' or '0.45, 0.32'."""
    nums = re.findall(r"([\d.]+)", text)
    if len(nums) < 2:
        return None
    try:
        x, y = float(nums[0]), float(nums[1])
        if 0 <= x <= 1 and 0 <= y <= 1:
            return (x, y)
    except (ValueError, IndexError):
        pass
    return None


def _parse_bbox(label: str) -> tuple[float, float, float, float]:
    """Parse a 'left,top,right,bottom' label into floats."""
    left, top, right, bottom = (float(v) for v in label.split(","))
    return left, top, right, bottom


def _distance_outside_box(
    x: float, y: float, box: tuple[float, float, float, float]
) -> float:
    """Euclidean distance from (x, y) to the bbox; 0.0 when inside it."""
    left, top, right, bottom = box
    dx = max(left - x, 0.0, x - right)
    dy = max(top - y, 0.0, y - bottom)
    return (dx * dx + dy * dy) ** 0.5


async def grounding_reward(args, sample, **kwargs) -> float:
    response = getattr(sample, "response", "") or ""
    label = getattr(sample, "label", "") or ""

    pred = _parse_coordinates(response)
    if pred is None:
        return -1.0

    box = _parse_bbox(label)
    left, top, right, bottom = box

    # Any click inside the element succeeds → full reward.
    outside = _distance_outside_box(pred[0], pred[1], box)
    if outside == 0.0:
        return 1.0

    # Outside: decay from +1 at the edge to −1 a full diagonal away, with a
    # floor so tiny targets keep a usable gradient.
    diag = ((right - left) ** 2 + (bottom - top) ** 2) ** 0.5
    margin = max(diag, 0.05)
    if outside >= margin:
        return -1.0
    return 1.0 - 2.0 * outside / margin


# ## Baseline Eval
#
# Let's evaluate the base Qwen3-VL-8B model on our held-out set before
# training to see how well it grounds UI elements out of the box.


def grounding_eval_fn(deployment: CustomDeployment, example: dict) -> dict:
    # Eval sends the screenshot as a separate image_url, so drop the marker.
    prompt = example["prompt"].replace("<image>", "").strip()
    label = example["label"]
    images = example["images"]

    # OpenAI multimodal user message: text + one image_url part per screenshot.
    content = [
        {"type": "text", "text": prompt},
        *({"type": "image_url", "image_url": {"url": img}} for img in images),
    ]
    msg = deployment.chat(
        [{"role": "user", "content": content}],
    )
    response = msg.get("content") or msg.get("reasoning_content") or ""

    pred = _parse_coordinates(response)
    box = _parse_bbox(label)
    if pred is None:
        inside = False
        outside = 1.0
    else:
        outside = _distance_outside_box(pred[0], pred[1], box)
        inside = outside == 0.0

    return {
        "score": 1.0 if inside else 0.0,
        "response": response,
        "inside_box": inside,
        "dist_outside": round(outside, 4),
        "pred": f"{pred[0]:.4f},{pred[1]:.4f}" if pred else "PARSE_FAIL",
        "label": label,
    }


def run_eval(deployment, *, max_concurrency: int = 2) -> tuple[float, list[dict]]:
    from concurrent.futures import ThreadPoolExecutor

    deployment.wait_until_ready(timeout=3000)

    def _score_one(example):
        return grounding_eval_fn(deployment, example)

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        rows = list(executor.map(_score_one, eval_dataset.rows()))
    mean = sum(r["score"] for r in rows) / len(rows) if rows else float("nan")
    return mean, rows


model = Qwen3_VL_8B()


def deploy_base_model():
    print("deploying base model deployment...")
    base_deployment = CustomDeployment.launch(
        model,
        unauthenticated=True,
    )
    print(f"Base model URL: {base_deployment.url}")
    return base_deployment


def run_baseline_evals(deployment):
    print("--- Evaluating base model... ---")
    base_mean, base_rows = run_eval(deployment)
    n_hits = sum(1 for r in base_rows if r.get("inside_box"))
    print(
        f"Base accuracy (clicks inside element): "
        f"{n_hits}/{len(base_rows)} ({base_mean:.1%})"
    )


# ## Training
#
# We use `Qwen3_VL_8B_Recipe` which carries VL-specific defaults:
# - **Frozen vision tower** (`freeze_params_name_list=["vision_model"]`) — RL
#   only updates the language backbone. This is the standard recipe for VLM RL:
#   a single sparse reward is too noisy to safely fine-tune a pretrained visual
#   encoder (you'd risk collapsing its features), and grounding is really about
#   teaching the decoder to *read out* coordinates from features the ViT already
#   provides. It's also cheaper — no optimizer state or backward pass for the ViT.
# - Padded (bshd) batches for the vision encoder
# - TP=4 for the 8B model across 8 H100s
# - Short response cap (64 tokens — coordinates are brief)
# - A high SGLang KV-cache fraction (0.75) for fast colocated rollouts
#
# The recipe overrides below are tuned to speed up training (~30m → ~19m on
# one 8×H100 node) while preserving the reward curve, uniquely fitted to this
# tutorial's short coordinate outputs. Short outputs let us increase rollout
# concurrency, which sets the memory budget, which sets the shard count.
#
# This tutorial runs 15 rollouts as a quick demo. For a more meaningful
# accuracy gain, increase `num_rollout`.
#
# We pass `metrics=WandbConfig(project="…")` so reward/KL/length curves stream to
# Weights & Biases — the key comes from the `wandb-secret` Modal secret. The
# Training Gym dashboard picks up the run's project/entity/id and wires up the
# **Open in W&B** button on the run. Drop `metrics=` to disable logging.

config = TrainConfig(
    model=model,
    dataset=train_dataset,
    recipe=Qwen3_VL_8B_Recipe(
        actor_num_gpus_per_node=8,
        tensor_model_parallel_size=4,
        num_rollout=15,
        save_interval=15,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        global_batch_size=16,
        rollout_max_response_len=64,
        sglang_mem_fraction_static=0.75,
        custom_rm_function=grounding_reward,
        no_save_optim=True,
        metrics=WandbConfig(project="computer-use-grounding"),
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
        print(f"Checkpoint: {checkpoint.path}")
    return checkpoint


# ## Evaluate the trained model
#
# Let's run the same eval on the trained checkpoint.


def deploy_trained_model(checkpoint):
    print("deploying trained model deployment...")
    trained_deployment = CustomDeployment.launch(
        model,
        checkpoint=checkpoint,
        app_name="qwen3-vl-8b-grounding-serve",
        served_model_name="qwen3-vl-8b-grounding",
        unauthenticated=True,
    )
    print(f"Trained model URL: {trained_deployment.url}")
    return trained_deployment


def run_trained_evals(trained_deployment):
    print("--- Evaluating trained model... ---")
    trained_mean, trained_rows = run_eval(trained_deployment)
    n_hits = sum(1 for r in trained_rows if r.get("inside_box"))
    print(
        f"Trained accuracy (clicks inside element): "
        f"{n_hits}/{len(trained_rows)} ({trained_mean:.1%})"
    )


if __name__ == "__main__":
    base_deployment = deploy_base_model()
    run_baseline_evals(base_deployment)
    checkpoint = train(config)
    trained_deployment = deploy_trained_model(checkpoint)
    run_trained_evals(trained_deployment)
