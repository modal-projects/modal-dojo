# ---
# order: 7
# github: https://github.com/modal-projects/sf3/tree/main/src/train
# ---
#
# # Fighting LLMs in Street Fighter III
#
# Instead of collecting prompts to train your model against,
# you can synthetically generate data using an
# [environment](https://gym.modal.dev/guides/recipe#environment).
# 
# Here, we show how we train [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
# on Street Fighter III using self-play. To enable this, we use the
# [OnlineRollout](https://gym.modal.dev/reference/onlinerollout) class and a
# custom generate function.
# 
# You can find the full code [here](https://github.com/modal-projects/sf3).
# 
# ## Creating the environment and reward function
#
# We create our environment by defining a custom generate function.
# Here, there are two fighters in a match which sample button sequences
# from the current policy to feed into the Gymnasium-style environment
# concurrently. In this self-play setup, we simply give the winner a 
# reward of 1 and the loser -1, with 0 for draws. This implies that in
# ideal play, we expect the sum of all rewards to be 0.
# 
# ```python
# MODEL = Qwen3_VL_8B()
# REWARDS = {"P1": (1.0, -1.0), "P2": (-1.0, 1.0), "draw": (0.0, 0.0)}
# 
#
# async def sf3_generate(args, sample, sampling_params):
#     from slime.utils.types import Sample
#
#     key = (sample.group_index, sample.index // 2)
#     fight = _fights.pop(key, None)
#     if fight is None:
#         fight = _fights[key] = asyncio.create_task(
#             _play_fight(args, sample, sampling_params)
#         )
#     try:
#         return (await asyncio.shield(fight))[sample.index % 2]
#     except Exception:
#         traceback.print_exc()
#         sample.status = Sample.Status.ABORTED
#         return [sample]
# 
# 
# async def _play_fight(args, sample, sampling_params):
#     characters = random.Random(sample.group_index).sample(ROSTER, 2)
#     identities = [{"character": c, "superArt": SUPER_ART} for c in characters]
#     env = await asyncio.to_thread(
#         create_environment,
#         EnvironmentConfig(
#             characters=tuple(characters),
#             outfits=(OUTFIT, OUTFIT),
#             super_arts=(SUPER_ART, SUPER_ART),
#             step_ratio=6,
#         ),
#     )
#     try:
#         observation, _ = await asyncio.to_thread(env.reset)
#         encoder = FrameEncoder()
#         recent = [deque(maxlen=RECENT_MOVE_LIMIT), deque(maxlen=RECENT_MOVE_LIMIT)]
#         moves = [[], []]
#         while True:
#             fighters = [
#                 player_state(observation, identities[seat], f"P{seat + 1}")
#                 for seat in range(2)
#             ]
#             pixels = observation["frame"]
#             frame_size = pixels.shape
#             frame = encoder.data_url(pixels)
#             turn = await asyncio.gather(
#                 *(
#                     _move(
#                         args,
#                         sample,
#                         sampling_params,
#                         seat,
#                         fighters,
#                         frame,
#                         frame_size,
#                         recent,
#                     )
#                     for seat in range(2)
#                 )
#             )
#             buttons = []
#             for seat, move in enumerate(turn):
#                 moves[seat].append(move)
#                 move_buttons, move_name = resolve_move_with_fallback(
#                     characters[seat],
#                     MODEL.parse_response(move.response).content,
#                     fighters[seat].side,
#                 )
#                 recent[seat].append(move_name)
#                 buttons.append(move_buttons)
#             for p1_button, p2_button in zip_longest(*buttons, fillvalue=0):
#                 observation, _, terminated, _, info = await asyncio.to_thread(
#                     env.step, {"agent_0": p1_button, "agent_1": p2_button}
#                 )
#                 if terminated or info["round_done"]:
#                     break
#             if info["round_done"]:
#                 for seat_recent in recent:
#                     seat_recent.clear()
#             if terminated:
#                 break
#     finally:
#         await asyncio.to_thread(env.close)
#     for seat_moves, reward in zip(moves, REWARDS[info["winner"]]):
#         for move in seat_moves:
#             move.reward = reward
#     return moves
# ```
# 
# ## Setting up a self-play recipe
# 
# Since the above logic is implemented as custom hooks, we can simply plug them into
# the `Qwen3_VL_8B_Recipe` class with ease. When you start training, watching the reward
# curves will not be that useful: instead, you'll want to watch the `Metrics` tab in the 
# [dashboard](https://gym.modal.dev/guides/dashboard) in addition to running
# [offline evals](https://github.com/modal-projects/sf3/tree/main/src/eval).
# 
# ```python
# NUM_ROLLOUTS = 10
#
# ROLLOUT_BATCH_SIZE = 4
# N_SAMPLES_PER_PROMPT = 2
#
# model = Qwen3_VL_8B()
# recipe = Qwen3_VL_8B_Recipe(
#     custom_generate_function=sf3_generate,
#     dynamic_sampling_filter_path="src.train.rollout.sf3_valid_group",
#     image_overlay=lambda image: create_gameplay_image(
#         base_image=image,
#         copy=True,
#         add_python_source=True,
#     ),
#     num_rollout=NUM_ROLLOUTS,
#     rollout_batch_size=ROLLOUT_BATCH_SIZE,
#     n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
#     global_batch_size=ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT,
#     rollout_max_response_len=MAX_TOKENS,
#     extra_config={
#         **Qwen3_VL_8B_Recipe().extra_config,
#         "micro_batch_size": 8,
#         "rewards_normalization": False,
#         "custom_megatron_init_path": "src.train.rollout.megatron_init",
#     },
# )
#
# config = TrainConfig(
#     model=model,
#     dataset=OnlineRollout(n_rows=ROLLOUT_BATCH_SIZE),
#     recipe=recipe,
# )
# 
# with config.launch() as run:
#     print(f"run id: {run.training_run_id}")
#     checkpoint = None
#     while True:
#         done = run.done()
#         latest = run.latest_checkpoint()
#         if latest is not None and latest != checkpoint:
#             checkpoint = latest
#             print(f"new checkpoint: {checkpoint.path}")
#         if done:
#             break
#         time.sleep(30)
#     if checkpoint is None:
#         raise RuntimeError("run produced no checkpoint")
#     print(convert_megatron_checkpoint_to_hf(checkpoint, model).path)
# ```
