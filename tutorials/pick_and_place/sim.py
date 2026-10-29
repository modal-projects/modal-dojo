import io
import math
import os
import random
import re
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")

N_BLOCKS = 4
N_TRAYS = 3
IMAGE_SIZE = 384

COLORS = {
    "red": (0.85, 0.15, 0.15),
    "green": (0.2, 0.7, 0.25),
    "blue": (0.2, 0.35, 0.9),
    "yellow": (0.95, 0.8, 0.15),
    "purple": (0.55, 0.25, 0.75),
    "orange": (0.95, 0.5, 0.1),
}

WORKSPACE = 0.25  # objects are placed within ±25 cm of the table center
BLOCK_HALF = 0.02  # 4 cm cubes
TRAY_HALF = 0.06  # 12 cm square trays
TRAY_WALL = 0.006
TRAY_HEIGHT = 0.03
RELEASE_HEIGHT = 0.1  # the gripper lets go with the block's center this high

CAMERA_HEIGHT = 1.0
CAMERA_FOV = 32.0  # degrees; covers the workspace plus a small margin


def sample_scene(seed: int) -> dict:
    rng = random.Random(seed)
    tray_colors = rng.sample(sorted(COLORS), N_TRAYS)
    block_colors = rng.sample(sorted(COLORS), N_BLOCKS)
    placed: list[tuple[float, float, float]] = []  # (x, y, clearance radius)

    def free_spot(radius: float) -> tuple[float, float]:
        limit = WORKSPACE - radius
        for _ in range(1000):
            x, y = rng.uniform(-limit, limit), rng.uniform(-limit, limit)
            if all(
                (x - px) ** 2 + (y - py) ** 2 > (radius + pr) ** 2
                for px, py, pr in placed
            ):
                placed.append((x, y, radius))
                return x, y
        raise RuntimeError(f"could not place object for seed {seed}")

    trays = [(color, *free_spot(TRAY_HALF * 1.45)) for color in tray_colors]
    blocks = [(color, *free_spot(BLOCK_HALF * 2.0)) for color in block_colors]
    target_block = rng.choice(block_colors)
    target_tray = rng.choice(tray_colors)
    return {
        "blocks": blocks,
        "trays": trays,
        "target_block": target_block,
        "target_tray": target_tray,
        "instruction": f"Put the {target_block} block in the {target_tray} tray.",
    }


def _box(half, pos, color, extra="") -> str:
    size = " ".join(f"{h:g}" for h in half)
    pos = " ".join(f"{p:g}" for p in pos)
    rgba = " ".join(f"{c:g}" for c in (*color, 1))
    return f'<geom type="box" size="{size}" pos="{pos}" rgba="{rgba}" {extra}/>'


def scene_xml(scene: dict) -> str:
    h, w, t = TRAY_HALF, TRAY_WALL, TRAY_HEIGHT
    bodies = []
    for color, x, y in scene["trays"]:
        parts = [((h, h, 0.003), (0, 0, 0.003))]  # floor
        parts += [((h, w, t / 2), (0, s * (h - w), t / 2)) for s in (-1, 1)]
        parts += [((w, h, t / 2), (s * (h - w), 0, t / 2)) for s in (-1, 1)]
        geoms = "".join(_box(half, pos, COLORS[color]) for half, pos in parts)
        bodies.append(f'<body pos="{x} {y} 0">{geoms}</body>')
    for color, x, y in scene["blocks"]:
        geom = _box((BLOCK_HALF,) * 3, (0, 0, 0), COLORS[color], 'mass="0.1"')
        bodies.append(
            f'<body name="{color}" pos="{x} {y} {BLOCK_HALF}">'
            f'<freejoint name="{color}"/>{geom}</body>'
        )
    return f"""
<mujoco>
  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
  </visual>
  <default><geom material="matte"/></default>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="0.93 0.93 0.95" width="8" height="8"/>
    <material name="matte" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <light pos="0.3 0.3 2" dir="-0.15 -0.15 -1" directional="true"
           diffuse="0.45 0.45 0.45" specular="0 0 0"/>
    {_box((0.4, 0.6, 0.01), (0, -0.2, -0.01), (0.82, 0.74, 0.62))}
    <camera name="top" pos="0 0 {CAMERA_HEIGHT}" fovy="{CAMERA_FOV}"/>
    {"".join(bodies)}
  </worldbody>
</mujoco>"""


def build_world(scene: dict):
    import mujoco

    model = mujoco.MjModel.from_xml_string(scene_xml(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _snapshot(renderer, data, camera: str):
    from PIL import Image

    renderer.update_scene(data, camera)
    return Image.fromarray(renderer.render())


def _png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def overhead_image(seed: int) -> bytes:
    import mujoco

    model, data = build_world(sample_scene(seed))
    with mujoco.Renderer(model, IMAGE_SIZE, IMAGE_SIZE) as renderer:
        return _png(_snapshot(renderer, data, "top"))


def image_to_world(u: float, v: float, z: float) -> tuple[float, float]:
    half_width = math.tan(math.radians(CAMERA_FOV / 2)) * (CAMERA_HEIGHT - z)
    return (u / 500 - 1) * half_width, (1 - v / 500) * half_width


COMMAND = re.compile(
    r"PICK\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*"
    r"PLACE\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)"
)


def parse_command(text: str) -> tuple[float, float, float, float] | None:
    match = COMMAND.search(text or "")
    if match is None:
        return None
    values = tuple(float(g) for g in match.groups())
    if not all(0 <= v <= 1000 for v in values):
        return None
    return values


def execute(seed: int, command) -> dict:
    import mujoco
    import numpy as np

    scene = sample_scene(seed)
    model, data = build_world(scene)
    pick_u, pick_v, place_u, place_v = command
    pick_xy = image_to_world(pick_u, pick_v, 2 * BLOCK_HALF)
    place_xy = image_to_world(place_u, place_v, RELEASE_HEIGHT)

    # Suction picks whatever block is straight below the pick point.
    geom = np.zeros(1, np.int32)
    mujoco.mj_ray(
        model, data, np.array([*pick_xy, 0.5]), np.array([0.0, 0.0, -1.0]),
        None, True, -1, geom,
    )
    hit = model.body(model.geom_bodyid[geom[0]]).name if geom[0] >= 0 else ""
    picked = hit if hit in COLORS else None

    if picked is not None:
        block = data.joint(picked)
        bx, by, _ = block.qpos[:3]
        block.qpos[:3] = (
            place_xy[0] + bx - pick_xy[0],
            place_xy[1] + by - pick_xy[1],
            RELEASE_HEIGHT,
        )
        block.qvel[:] = 0
        mujoco.mj_forward(model, data)
        for _ in range(round(1.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)

    success = False
    if picked == scene["target_block"]:
        x, y, z = data.body(picked).xpos
        tx, ty = next(
            (tx, ty) for c, tx, ty in scene["trays"] if c == scene["target_tray"]
        )
        inner = TRAY_HALF - TRAY_WALL
        success = abs(x - tx) < inner and abs(y - ty) < inner and z < TRAY_HEIGHT

    with mujoco.Renderer(model, IMAGE_SIZE, IMAGE_SIZE) as renderer:
        image = _png(_snapshot(renderer, data, "top"))
    return {"picked": picked, "success": bool(success), "image": image}
