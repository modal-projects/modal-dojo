---
order: 7
github: https://github.com/modal-projects/sf3
---

# Fighting LLMs in Street Fighter III

Instead of collecting prompts to train your model against,
you can synthetically generate data using an
[environment](https://gym.modal.dev/guides/recipe#environment).

Here, we show how we train [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
on Street Fighter III using self-play. To enable this, we use the
[OnlineRollout](https://gym.modal.dev/reference/onlinerollout) class and a
custom generate function.

<video controls autoplay muted loop playsinline width="100%">
  <source src="https://modal-cdn.com/cdnbot/iter49_vs_cpu1_Gouki_vs_Chun-Li_win_7c06e16b.mp4" type="video/mp4">
  <a href="https://modal-cdn.com/cdnbot/iter49_vs_cpu1_Gouki_vs_Chun-Li_win_7c06e16b.mp4">Watch a trained Gouki defeat CPU Chun-Li.</a>
</video>

You can find the [full code here](https://github.com/modal-projects/sf3/tree/main/src/train).

## Creating the environment and reward function

We create our environment by defining a custom generate function.
Here, there are two fighters in a match which sample button sequences
from the current policy to feed into the
[Gymnasium](https://gymnasium.farama.org/)-style environment
concurrently.

For the reward, each move choice is a discounted score based on its net damage (i.e., dealt minus taken) from that point to the end of the round. So in ideal play, we actually expect the sum of all rewards to be 0.

```python
MODEL = Qwen3_VL_8B()
GAMMA = 0.9

_fights: dict[tuple[int, int], asyncio.Task] = {}


async def sf3_generate(args, sample, sampling_params):
    from slime.utils.types import Sample

    key = (sample.group_index, sample.index // 2)
    fight = _fights.pop(key, None)
    if fight is None:
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=256)
        )
        fight = _fights[key] = asyncio.create_task(
            _play_fight(args, sample, sampling_params)
        )
    try:
        return (await asyncio.shield(fight))[sample.index % 2]
    except Exception:
        traceback.print_exc()
        sample.status = Sample.Status.ABORTED
        return [sample]


async def _play_fight(args, sample, sampling_params):
    characters = random.sample(ROSTER, 2)
    identities = [{"character": c, "superArt": SUPER_ART} for c in characters]
    env = await asyncio.to_thread(
        create_environment,
        SF3EnvironmentConfig(
            characters=tuple(characters),
            outfits=(OUTFIT, OUTFIT),
            super_arts=(SUPER_ART, SUPER_ART),
            step_ratio=6,
        ),
    )
    try:
        observation, _ = await asyncio.to_thread(env.reset)
        encoder = FrameEncoder()
        recent = [deque(maxlen=RECENT_MOVE_LIMIT), deque(maxlen=RECENT_MOVE_LIMIT)]
        moves = [[], []]
        damage, rounds, round_index = [], [], 0
        while True:
            fighters = [
                player_state(observation, identities[seat], f"P{seat + 1}")
                for seat in range(2)
            ]
            pixels = observation["frame"]
            frame_size = pixels.shape
            frame = encoder.data_url(pixels)
            turn = await asyncio.gather(
                *(
                    _move(
                        args,
                        sample,
                        sampling_params,
                        seat,
                        fighters,
                        frame,
                        frame_size,
                        recent,
                    )
                    for seat in range(2)
                )
            )
            buttons = []
            for seat, move in enumerate(turn):
                moves[seat].append(move)
                move_buttons, move_name = resolve_move_with_fallback(
                    characters[seat],
                    MODEL.parse_response(move.response).content,
                    fighters[seat].side,
                )
                recent[seat].append(move_name)
                buttons.append(move_buttons)
            turn_damage = 0.0
            for p1_button, p2_button in zip_longest(*buttons, fillvalue=0):
                observation, step_damage, terminated, _, info = await asyncio.to_thread(
                    env.step, {"agent_0": p1_button, "agent_1": p2_button}
                )
                turn_damage += step_damage
                if terminated or info["round_done"]:
                    break
            damage.append(turn_damage)
            rounds.append(round_index)
            if info["round_done"]:
                round_index += 1
                for seat_recent in recent:
                    seat_recent.clear()
            if terminated:
                break
    finally:
        await asyncio.to_thread(env.close)
    returns, G = [0.0] * len(damage), 0.0
    for t in reversed(range(len(damage))):
        if t + 1 < len(damage) and rounds[t + 1] != rounds[t]:
            G = 0.0
        G = damage[t] + GAMMA * G
        returns[t] = G / HEALTH_MAX
    for seat_moves, sign in zip(moves, (1, -1)):
        for move, G in zip(seat_moves, returns):
            move.reward = sign * G
    # slime's static batching needs every step's sample count divisible by DP.
    gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    keep = sorted(random.sample(range(len(damage)), len(damage) // gpus * gpus))
    if not keep:
        raise RuntimeError("fight too short to align with DP")
    return [[seat_moves[t] for t in keep] for seat_moves in moves]
```

## Setting up a self-play recipe

Since the above logic is implemented as custom hooks, we can plug them into
the `Qwen3_VL_8B_Recipe` class.


```python
NUM_ROLLOUTS = 100

ROLLOUT_BATCH_SIZE = 32
N_SAMPLES_PER_PROMPT = 2
GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT // 4

model = Qwen3_VL_8B()
recipe = Qwen3_VL_8B_Recipe(
    custom_generate_function=sf3_generate,
    dynamic_sampling_filter_path="src.train.rollout.sf3_valid_group",
    image_overlay=lambda image: create_gameplay_image(
        base_image=image,
        copy=True,
        add_python_source=True,
    ),
    actor_num_gpus_per_node=8,
    num_rollout=NUM_ROLLOUTS,
    save_interval=10,
    rollout_batch_size=ROLLOUT_BATCH_SIZE,
    n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
    global_batch_size=GLOBAL_BATCH_SIZE,
    rollout_max_response_len=MAX_TOKENS,
    extra_config={
        **Qwen3_VL_8B_Recipe().extra_config,
        "custom_megatron_init_path": "src.train.rollout.megatron_init",
    },
)

config = TrainConfig(
    model=model,
    dataset=OnlineRollout(n_rows=ROLLOUT_BATCH_SIZE),
    recipe=recipe,
)
```

## Evaluating checkpoints during training

The reward curves are not very informative for this use case. Instead, for each new checkpoint, we run custom [evals](https://github.com/modal-projects/sf3/tree/main/src/eval) that calculate ELO scores and track win rates to see if our model is improving.

```python
cache_volume_url = (
    f"https://modal.com/storage/{modal.Workspace.from_context().hydrate().name}"
    f"/{modal.Environment.from_context().hydrate().name}/{cache_volume.name}"
)


def eval_checkpoint(checkpoint):
    hf_checkpoint = convert_megatron_checkpoint_to_hf(checkpoint, model)
    print(f"converted checkpoint: {hf_checkpoint.path}")
    player = f"{CHECKPOINTS_MOUNT}/{hf_checkpoint.path_relative_to_volume}"
    for opponent in ("cpu:1", POLICY_MODEL_KEY):
        report = orchestrate.remote(players=(player, opponent))
        print(
            f"{checkpoint.name} vs {opponent}: "
            f"{cache_volume_url}/{report['config']['output_dir']}"
        )


with config.launch() as run, eval_app.run(), ThreadPoolExecutor() as evals:
    print(f"run id: {run.training_run_id}")
    checkpoint, pending = None, []
    while True:
        done = run.done()
        latest = run.latest_checkpoint()
        if latest is not None and (
            checkpoint is None or latest.name != checkpoint.name
        ):
            checkpoint = latest
            print(f"new checkpoint: {checkpoint.path}")
            pending.append((checkpoint.name, evals.submit(eval_checkpoint, checkpoint)))
        if done:
            break
        time.sleep(30)
    if checkpoint is None:
        raise RuntimeError("run produced no checkpoint")
    for name, future in pending:
        try:
            future.result()
        except Exception as exc:
            print(f"{name} eval failed: {type(exc).__name__}: {exc}")
```
