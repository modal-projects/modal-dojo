---
order: 8
github: https://github.com/modal-projects/sf3
---

# Fighting LLMs in Street Fighter III

Instead of collecting prompts to train your model against, you can synthetically generate data using an [environment](https://gym.modal.dev/guides/recipe#environment).

Here, we show how we train [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) to play Street Fighter III. Fights run in a [Modal Sandbox](https://modal.com/docs/guide/sandbox), and the model trains with curriculum learning against increasingly difficult CPU opponents. To enable this, we use the [OnlineRollout](https://gym.modal.dev/reference/onlinerollout) class and custom rollout, generate, and reward hooks.

<video controls autoplay muted loop playsinline width="100%">
  <source src="https://modal-cdn.com/cdnbot/yang_vs_chunli_ckpt5_20_100_d1b99ed9.mp4" type="video/mp4">
  <a href="https://modal-cdn.com/cdnbot/yang_vs_chunli_ckpt5_20_100_d1b99ed9.mp4">Watch Yang go from losing to beating a level 8 CPU Chun-Li after 5, 20, and 100 rollouts.</a>
</video>

You can find the [full code here](https://github.com/modal-projects/sf3/tree/main/src/train).

## Fighting in Sandboxes

Each fight runs the emulator in a Sandbox. The worker inside it sends the current frame and the model's last four moves to a `/move` endpoint on the trainer, which samples the next move from the policy with SGLang and returns the move's tokens and log probabilities. The worker then converts the move into button presses, waits a fixed reaction delay so the model plays at human speed, and steps the emulator.

```python
while True:
    fighters = [
        player_state(observation, identities[f"player{seat + 1}"], f"P{seat + 1}")
        for seat in range(2)
    ]
    frame = encoder.data_url(observation["frame"])
    response = client.post(
        f"{endpoint}/move",
        json={"sample": sample, "fighters": ..., "frame": frame, "recent": list(recent)},
    )
    generated = response.json()
    bad = generated["name"] not in generated["available"]
    buttons, move_name = resolve_move_with_fallback(
        fighters[0].character,
        generated["name"] if not bad else "No-Move",
        fighters[0].side,
    )
    delay = model_reaction_frames(buttons, inference_latency_ms)
    observation, hp_diff, terminated, truncated, info = env.step_frames(
        [{"agent_0": button} for button in [0] * delay + buttons]
    )
    ...
```

Starting a Sandbox that runs this worker is only a few lines of code.

```python
async def run_arcade_sandbox(config, endpoint, sample, sampling_params, inference_latency_ms):
    payload = {
        "config": asdict(config),
        "endpoint": endpoint,
        "sample": sample,
        "sampling_params": sampling_params,
        "inference_latency_ms": inference_latency_ms,
    }
    async with _slots:
        app = await modal.App.lookup.aio(APP_NAME)
        sandbox = await modal.Sandbox.create.aio(
            "sleep",
            str(WORKER_TIMEOUT),
            app=app,
            image=modal.Image.from_id(os.environ[WORKER_IMAGE_ENV]),
            cpu=1,
            memory=2 * GB,
            region=os.environ.get("MODAL_REGION", CONTAINER_REGION),
            runtime="gvisor",
            timeout=WORKER_TIMEOUT,
            env={WORKER_INPUT_ENV: json.dumps(payload)},
        )
        try:
            process = await sandbox.exec.aio(
                "python", "-m", "src.train.worker", timeout=WORKER_TIMEOUT
            )
            await process.wait.aio()
            data = await sandbox.filesystem.read_bytes.aio(RESULT_PATH)
            return json.loads(gzip.decompress(data))
        finally:
            await asyncio.shield(sandbox.terminate.aio())
```

Finally, the rollout function serves the `/move` endpoint through a [Modal Tunnel](https://modal.com/docs/guide/tunnels), and our custom generate hook passes the tunnel's URL to a new Sandbox for each fight.

```python
@asynccontextmanager
async def _inference_endpoint(args):
    app = web.Application()
    app.router.add_post("/move", generate)
    ...
    async with modal.forward.aio(port) as tunnel:
        yield tunnel.url


def sf3_rollout(args, rollout_id, data_source, evaluation=False):
    async def collect():
        async with _inference_endpoint(args) as endpoint:
            args.sf3_inference_url = endpoint
            output, aborted = await generate_rollout_async(
                args, rollout_id, data_source.get_samples
            )
            ...
            return output

    return run(collect())


async def sf3_generate(args, sample, sampling_params):
    config = EnvironmentConfig(
        characters=random.sample(ROSTER, 2),
        vs_cpu=True,
        cpu_difficulty=_sample_cpu_difficulty(),
        ...
    )
    result = await run_arcade_sandbox(
        config,
        args.sf3_inference_url,
        {"index": sample.index, "group_index": sample.group_index},
        sampling_params,
        INFERENCE_LATENCY_MS,
    )
    ...
```

## Defining rewards

Each move's reward combines the round's outcome, discounted by how long before the end of the round the move was made, with a smaller, clamped term for the net damage (damage dealt minus damage taken) from that move onward. Invalid move names are penalized.

```python
DISCOUNT_HALF_LIFE_FRAMES = 60 * 60
DAMAGE_REWARD_WEIGHT = 0.1
ROUND_WIN_BONUS = 1.0
INVALID_MOVE_PENALTY = 20 / HEALTH_MAX


def _assign_rewards(moves, invalid, damage, durations, winner):
    outcome = ROUND_WIN_BONUS * {"P1": 1, "P2": -1, "draw": 0}[winner]
    damage_return = 0.0
    for move, bad, hp_diff, frames in reversed(
        list(zip(moves, invalid, damage, durations, strict=True))
    ):
        discount = 0.5 ** (frames / DISCOUNT_HALF_LIFE_FRAMES)
        outcome *= discount
        damage_return = hp_diff / HEALTH_MAX + discount * damage_return
        move.reward = (
            outcome
            + DAMAGE_REWARD_WEIGHT * max(-1.0, min(1.0, damage_return))
            - INVALID_MOVE_PENALTY * bad
        )
```

## CPU curriculum

The CPU difficulty for each fight comes from a [PFSP](https://www.nature.com/articles/s41586-019-1724-z)-style curriculum where the next level unlocks once the model wins at least half of its recent fights. Levels are sampled in proportion to `p * (1 - p)`, where `p` is the model's win rate; i.e., most fights are against opponents it's evenly matched with.

```python
def _sample_cpu_difficulty():
    hardest_won = max(
        (
            d
            for d in range(1, MAX_CPU_DIFFICULTY + 1)
            if len(r := _results[d]) >= MIN_CPU_RESULTS and _win_rate(r) >= 0.5
        ),
        default=0,
    )
    levels = range(1, min(hardest_won + 1, MAX_CPU_DIFFICULTY) + 1)
    rates = [_win_rate(_results[d]) for d in levels]
    return random.choices(levels, [p * (1 - p) for p in rates])[0]
```

Rewards vary widely between difficulty levels, so advantages are computed against a stratified moving average baseline.

```python
def sf3_reward_post_process(args, samples):
    rewards = [sample.reward for sample in samples]
    groups = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[sample.metadata["cpu_difficulty"]].append(i)
    advantages = list(rewards)
    for difficulty, members in groups.items():
        baseline = _reward_baselines.get(difficulty, 0.0)
        for i in members:
            advantages[i] -= baseline
        mean = statistics.fmean(rewards[i] for i in members)
        _reward_baselines[difficulty] = 0.9 * baseline + 0.1 * mean
    return rewards, advantages
```

## Cooking up a recipe

We can plug the above logic into the `Qwen3_VL_8B_Recipe` class as custom hooks.

```python
NUM_ROLLOUTS = 100

ROLLOUT_BATCH_SIZE = MAX_WORKERS
N_SAMPLES_PER_PROMPT = 1
GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT

model = Qwen3_VL_8B()
recipe = Qwen3_VL_8B_Recipe(
    rollout_function=sf3_rollout,
    custom_generate_function=sf3_generate,
    custom_reward_post_process_function=sf3_reward_post_process,
    custom_rollout_log_function=sf3_log_rollout,
    dynamic_sampling_filter_path="src.train.rollout.sf3_valid_group",
    image_overlay=lambda image: image.uv_pip_install(
        *PROCESSOR_PACKAGES
    ).add_local_python_source("src", copy=True),
    actor_num_gpus_per_node=8,
    num_rollout=NUM_ROLLOUTS,
    save_interval=5,
    capture_trace=True,
    rollout_batch_size=ROLLOUT_BATCH_SIZE,
    n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
    global_batch_size=GLOBAL_BATCH_SIZE,
    rollout_max_response_len=MAX_TOKENS,
    rollout_temperature=TEMPERATURE,
    rollout_top_p=TOP_P,
    extra_config={
        **Qwen3_VL_8B_Recipe().extra_config,
        "micro_batch_size": 4,
        "custom_megatron_init_path": "src.train.rollout.megatron_init",
        "sglang_mm_attention_backend": "sdpa",
        "sglang_cuda_graph_backend_prefill": "disabled",
    },
)

config = TrainConfig(
    model=model,
    dataset=OnlineRollout(n_rows=ROLLOUT_BATCH_SIZE),
    recipe=recipe,
)

if __name__ == "__main__":
    recipe.environment = {
        **recipe.environment,
        WORKER_IMAGE_ENV: build_worker_image(),
    }
    run = config.launch()
    print(f"run id: {run.training_run_id}")
```

## Results

The full run took about 9 hours on 8 H200s. The reward stays constantly positive throughout the run; in the presence of a curriculum, this indicates that the model is consistently improving.

![Training dashboard for the run](https://modal-cdn.com/cdnbot/dash-2c0r6kdph_f3b74fa5.webp)

The win rate against the level 8 CPU rose from 53% for the base model to 76% for the final checkpoint. Most of the gain comes from characters the base model couldn't win a single fight with, such as Yang, Makoto, Q, and Oro.

![Win rate against the level 8 CPU by character](https://modal-cdn.com/cdnbot/win_rate_by_character-1aqhhyjj6_b1ab5ddb.webp)

A method to track gameplay style is by counting how much of each fight consists of a single, spammed move. Early on, that one move makes up over 70% of a fight. With training, it drops to around 20% by rollout 60. Later on, some characters, such as Ibuki, Urien, and Elena, settle back on a single Target combo for over 80% of their moves. This is akin to the model first exploring the space of possible movesets, then exploiting a strategy that seems to work.

![Most-used move's share of a fight by character](https://modal-cdn.com/cdnbot/move_diversity_by_charactere9qxmzrt_d4764acb.webp)
