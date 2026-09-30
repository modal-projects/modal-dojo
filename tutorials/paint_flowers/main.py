# ---
# order: 1
# deps: pillow
# ---
#
# # Painting flowers with code
#
# As this [blog post](https://surya.website/rling-qwen-to-paint-with-code) shows,
# you can train a model to create watercolour sketches of flowers using
# [p5.brush](https://p5brush.org),
# and use a judge to do pairwise comparisons against a reference pool of images
# for the reward function.
#
# In this tutorial, we train [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
# and use
# [HuggingEnvs/watercolour-reference-pool](https://huggingface.co/datasets/HuggingEnvs/watercolour-reference-pool)
# as the reference pool. During each rollout, sketches are rendered to PNGs in a
# [Modal Sandbox](https://modal.com/docs/guide/sandboxes) and
# [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) compares each against the
# reference image pool.
#
# ![Reward curve](https://modal-cdn.com/cdnbot/flower-reward1maaxmni_1df0871b.webp)
#
# <video controls autoplay muted loop style="display: block; margin: 0 auto;">
# <source src="https://modal-cdn.com/example-paint_flowers.mp4" type="video/mp4">
# </video>

import asyncio
import base64
import itertools
import random
import sys
from pathlib import Path

from modal_dojo import (
    DatasetConfig,
    Endpoint,
    Qwen3_5_4B,
    Qwen3_5_4B_Recipe,
    Qwen3_6_27B,
    Sandbox,
    TrainConfig,
)
from modal_dojo.common.sample_extraction import IMAGE_SAMPLE_LIMIT_ENV

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers

base_model = Qwen3_5_4B()

# ## Get the dataset
#
# Using the present species and palette colors, we create a set of prompts to train our model.

SPECIES = ["hibiscus"]
PALETTES = [
    "peach",
    "crimson",
    "butter",
    "lilac",
    "coral",
    "indigo",
    "blush",
    "amber",
]

SYSTEM_PROMPT = """\
You paint watercolours by writing a p5.js 2.x sketch that uses the p5.brush library, available as the global `brush`.

Reply with one ```javascript fence containing a complete sketch, and nothing else.

Use exactly this skeleton:

  async function setup() {
    createCanvas(600, 600, WEBGL);
    brush.scaleBrushes(3);
    angleMode(DEGREES);
    noLoop();
  }

  function draw() {
    translate(-width / 2, -height / 2);
    background(hex);
    // painting goes here
  }

So: angles are in degrees, and after the translate the canvas runs from 0 to 600 on both axes with the origin at the top LEFT, not the centre. The centre of the canvas is (300, 300). Every coordinate you paint at must be between 0 and 600: a negative coordinate is off the canvas and paints nothing at all. Colours are hex strings like "#e08a72".

Follow the user brief for species and palette.

Paint one flower, centred, filling most of the frame, with a stem and leaves. Do not fill the canvas edge to edge.

Three numbers matter more than any adjective:
- Five petals. Paint each petal two or three times over, not once: a first pass at full size, then a smaller and more opaque pass inside it, and a small dark one near the centre. That layering is where a watercolour gets its depth, and it puts the whole painting at fifteen to thirty filled shapes. Keep the petals as five broad lobes, not fifteen separate little marks.
- Petals reaching 200 to 240 units from the centre, so the flower occupies the frame.
- Opacity never below 150, and 180 to 230 on the petals, with brush.fillBleed between 0.2 and 0.3. A dilute wash with a wide bleed disperses until no pigment reaches density and the flower comes out invisible.

These ten brush methods exist. Nothing else on `brush` exists, and there is no way to
name or select a brush: everything is painted as a filled shape. Do not call any other brush
method, and never use the bare p5 drawing functions such as ellipse, rect, vertex or beginShape:
  brush.scaleBrushes(factor)
  brush.noStroke()
  brush.fill(colorHex, opacity)
  brush.noFill()
  brush.fillBleed(amount)
  brush.fillTexture(amount, borderIntensity)
  brush.beginShape(curvature)
  brush.vertex(x, y)
  brush.endShape(true)
  brush.circle(x, y, radius, scribble)

colorHex is a string like "#e08a72". opacity runs 0 to 255. amount, curvature,
borderIntensity and scribble run 0 to 1.

Every mark is a filled shape. Build petals and leaves with brush.beginShape, a run of at least
three brush.vertex calls, then brush.endShape(true), and call brush.fill before each one. A stem
is a long narrow filled shape, not a line. brush.circle fills a disc.
"""

USER_TEMPLATE = (
    "Paint a {palette} {species} in watercolour: one bloom, seen from the "
    "front, with a stem and leaves, on coloured paper."
)


def build_prompts(combos: list[tuple[str, str]], n: int) -> list[dict[str, str]]:
    rows = []
    for species, palette in itertools.islice(itertools.cycle(combos), n):
        rows.append(
            {"prompt": USER_TEMPLATE.format(species=species, palette=palette)}
        )
    return rows


class FlowerPromptDataset(DatasetConfig):
    def __init__(self, prompts: list[dict[str, str]]):
        self.prompts = prompts

    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> str:
        return "label"

    def rows(self):
        return [
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["prompt"]},
                ],
                "label": r["prompt"],
            }
            for r in self.prompts
        ]


N_TRAIN = 224
N_EVAL = 8

combos = list(itertools.product(SPECIES, PALETTES))
random.Random(7).shuffle(combos)
train_dataset = FlowerPromptDataset(build_prompts(combos, N_TRAIN))
eval_dataset = FlowerPromptDataset(build_prompts(combos, N_EVAL))


# ## Creating a reward function
#
# The reward function renders each sketch in a
# [Modal Sandbox](https://modal.com/docs/guide/sandboxes) and uses an LLM judge
# to do pairwise comparisons. We serve the judge as an
# [Endpoint](https://modal.com/docs/guide/endpoints).

def deploy_judge():
    print("deploying judges...")
    judge = Endpoint.launch(
        Qwen3_6_27B(),
        unauthenticated=True,
        recreate_if_existing=True,
    )
    helpers.launch_hpsv3()
    judge.wait_until_ready(timeout=30 * 60)
    return judge


def render_in_sandbox(code: str) -> tuple[bytes | None, dict]:
    try:
        with Sandbox(
            image=helpers.renderer_image(),
            workdir="/render",
            timeout=300,
            cpu=1.0,
            memory=2048,
            block_network=True,
            app_name="training-gym-flower-render",
        ) as sandbox:
            sandbox.write("/render/render.js", helpers.RENDER_JS)
            sandbox.write("/render/sketch.js", code)
            result = sandbox.run(
                "node", "/render/render.js", "/render/sketch.js", timeout=180
            )
        out, err = result.stdout, result.stderr
        if "PNGB64:" in out:
            png = base64.b64decode(out.split("PNGB64:", 1)[1].strip())
            return png, {"render": "ok"}
        kind = "fail" if "SKETCH_ERROR:" in err else "unavailable"
        return None, {"render": kind, "stderr": err[-400:]}
    except Exception as e:
        return None, {
            "render": "unavailable",
            "stderr": f"{type(e).__name__}: {e}"[-400:],
        }


def make_flower_rm(judge):
    async def flower_rm(args, sample, **kwargs) -> float | None:
        code = helpers.extract_sketch(sample.response, base_model.parse_response)
        if code is None:
            reward, meta, png = 0.0, {"gate": "no valid sketch"}, None
        else:
            png, render_meta = await asyncio.to_thread(render_in_sandbox, code)
            reward, meta, png = await asyncio.to_thread(
                helpers.score_png, png, code, judge, render_meta
            )
        metadata = {**(getattr(sample, "metadata", None) or {}), **meta}
        if png is not None:
            metadata["image"] = png
        sample.metadata = metadata
        if reward is None:
            sample.remove_sample = True
        return reward

    return flower_rm


# ## Training
#
# After that, it's simple to start training!

ROLLOUT_BATCH_SIZE = 8
N_SAMPLES_PER_PROMPT = 8


def build_config(judge):
    return TrainConfig(
        model=base_model,
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        recipe=Qwen3_5_4B_Recipe(
            custom_rm_function=make_flower_rm(judge),
            custom_reward_post_process_function=helpers.skip_infra_rewards,
            num_rollout=100,
            rollout_batch_size=ROLLOUT_BATCH_SIZE,
            global_batch_size=ROLLOUT_BATCH_SIZE,
            n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
            save_interval=50,
            apply_chat_template_kwargs='{"enable_thinking": false}',
            image_overlay=lambda image: helpers.overlay_flower_image(image).env(
                {
                    IMAGE_SAMPLE_LIMIT_ENV: str(
                        ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT
                    )
                }
            ),
        ),
    )


if __name__ == "__main__":
    judge = deploy_judge()
    config = build_config(judge)
    run = config.launch()
    print(f"run id: {run.training_run_id}")
