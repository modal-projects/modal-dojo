# Kimi-K3 LoRA 256k on H200: validation report (in progress)

Status: **not validated yet**. No training step has finished. Recipe: `Kimi_K3_LoRA_256k_H200_Recipe` (16 x H200:8, TP16 inference, 256k context). Dashboard: https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run (env `helena-dev`).

## Runs

| Run | App | Trainer layout | Result |
| --- | --- | --- | --- |
| [fixed-fathom-65c285afb988](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/fixed-fathom-65c285afb988) | `ap-J7DSUkrFG6SgG0tfvKAgWH` | TP8 x CP4 x PP4, EP32, 65536 tok/GPU, routing replay on | Cancelled after SGLang engine init crash |
| [stable-plaster-5f9920d29d31](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/stable-plaster-5f9920d29d31) | `ap-sqrCXIKbt2SE5qaJrFiYEg` | TP8 x CP4 x PP4, EP32, 65536 tok/GPU, routing replay off | FAILED at Optimizer step, step 1/2: CUDA OOM in trainer forward |
| [crimson-brass-1832e798c59a](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/crimson-brass-1832e798c59a) | `ap-GyGeQjKO0yA5WcZicYxHvw` | TP4 x CP8 x PP4, EP32, 32768 tok/GPU, routing replay off | FAILED at Optimizer step, step 1/2 (reward 0.125): same trainer OOM site |

All runs use `validate.py --rollouts 2 --launch` in capacity mode: each sequence is filled to at least 261,120 tokens.

## fixed-fathom: routing replay crashes on K3

- SGLang allocated its KV pool before the crash: `KV Cache is allocated. dtype: torch.bfloat16, #tokens: 402497, KV size: 10.36 GB` (all 16 TP ranks).
- Then every engine failed in `RoutedExpertsCapturer`: `AttributeError: 'KimiLinearConfig' object has no attribute 'num_experts_per_tok'. Did you mean: 'num_experts_per_token'?`
- Cause: `use_rollout_routing_replay=True` in the K3 base recipe. radiant-tom-b8a5302e43ee (64k, completed) ran with it `false`.
- Fix: 422e3f08 sets `use_rollout_routing_replay=False` in the H200 256k recipe.

## stable-plaster: trainer OOM at 64k tokens/rank

- Inference init and the first rollout completed: the failure was inside `actor_model.train(rollout_id, rollout_data_pack)` (`miles/train.py:213`). The actual sequence lengths in that rollout weren't captured in the retained logs.
- Trainer OOM in the K3 MoE activation `situ_and_mul` (`miles_plugins/models/kimi_k3/ops.py:49`, called from Megatron `experts.py:895` `bias_act_func`): `Tried to allocate 5.19 GiB. GPU 5 has a total capacity of 139.79 GiB of which 6.86 GiB is free ... 112.66 GiB is allocated by PyTorch ... 8.88 GiB is reserved by PyTorch but unallocated`. Full per-layer recompute was on (`full`/`uniform`/1).
- After the OOM: `NonRetryableError: All cells failed in this training attempt`, followed by `ServerCell ... was garbage collected without dispose()` assertions for each inference cell. These are cleanup errors that came after the OOM.
- Log window: 2026-10-13 07:57:59 to 08:00:23 UTC.

## Change for crimson-brass (a6a360fb)

- Miles sizes each micro-batch at `max_tokens_per_gpu * cp_size` (`miles/utils/dp_schedule.py:52`, `miles/backends/training_utils/data/rollout.py:529`). That makes per-rank tokens `seq_length / CP`.
- TP8 x CP4 put 65,536 tokens on each rank. radiant-tom at 64k on B300 used 32,768 (`max_tokens_per_gpu=32768`, CP2).
- TP4 x CP8 x PP4 brings it back to 32,768 tokens/rank. EP32 and PP4 stay the same, so expert weights per GPU don't change. Miles' K3 attention only needs `num_attention_heads % tp_size == 0` (96 / 4 works). KDA CP goes through `fla` `build_cp_context` (`miles_plugins/models/cp_utils.py`), which has no CP-size limit (checked against radixark/miles HEAD af2b2d0, not the exact image commit).
- crimson-brass reused the converted checkpoint: `Found existing torch_dist checkpoint at /checkpoints/Kimi-K3_torch_dist; skipping conversion.`

## crimson-brass: 256k rollout and log probs pass, train forward still OOMs

- Got its 16 x H200 gang at about 16:01 UTC, after roughly 50 min queued (`Function 'train' ... is waiting to be scheduled on a GPU_H200 worker`). Trainer checkpoint reads took 390-850 s per rank (`DOJO_CKPT_READ`).
- Rollout 0 (`perf 0`): `episode_total_response_length/mean` 261,916.25, `episode_raw_reward` 0.125, `truncated_ratio` 0.875, generated response length 1,276-1,278.
- `K3_CONTEXT_MEMORY`: `before_log_prob` peak allocated up to 80.5 GiB; `before_train:0:0` peak allocated up to 94.5 GiB, reserved up to 119.3 GiB; host up to 1,253.9 of 1,920 GiB.
- The train forward OOMed in `situ_and_mul` again (`ops.py:49`, which upcasts the fc1 output to fp32): `Tried to allocate 4.93 GiB ... 6.46 GiB is free ... 106.54 GiB is allocated by PyTorch`, and on another rank `Tried to allocate 2.71 GiB ... 4.39 GiB is free ... 111.63 GiB`. Only 2 trainer actors (10.100.0.13 and 10.100.0.15) raised `OutOfMemoryError`.
- Many ranks also logged non-fatal `TorchMemorySaver::malloc return OOM` warnings for 2.9-11.1 GB requests.

Suspected cause (not measured): the capacity probe filled ~260k tokens by repeating `"Diagnostic record: all checks completed.\n"`. Repeated filler concentrates MoE routing on a few experts, so the EP ranks that host them get far more tokens than average. That happened in GLM-5.3 long-context validation, and it fits only 2 of 128 ranks failing. Next run: the filler is built from 4,096 varied dapo-math problems (`filler_tokens` in `validate.py`).

## 8 nodes

Not tried. It was ruled out by estimate (see `model_setup.md`), so it isn't proven to fail. Inference (not measured): stable-plaster OOMed on 16 nodes at 64k tokens/rank, and 8 nodes would need either 2x the tokens/rank or 2x the model shard per GPU.

## Continuation, 2026-10-13

- Source at takeover: `b5b5e3014aa3973b474187900e42a8dd08c8729e`.
- `crimson-brass` was queued, not converting: the conversion cache-hit log was already present while the dashboard still showed `Converting model`. Modal function call `fc-01M4TX0203JCE4DHJZGH7QBS8N` was enqueued at 15:09:48.429 UTC and started at 15:58:45.632 UTC (48m57s queue time). No replacement run was launched.
- Runtime resources verified through `modal function info`: 16-node RDMA cluster, 8 H200/node, 1,966,080 MiB host memory limit, no region/cloud restriction, no retries, 86,400s execution timeout.
- Trainer initialization reached rank-32 LoRA construction on all 128 ranks and began loading `/checkpoints/Kimi-K3_torch_dist` at 16:03:45 UTC. TP/PP reshards from the conversion layout `(32, 1)` to `(4, 4)`; the RNG-state warning is expected for that layout change. SGLang is gated while trainer initialization completes.
- Checkpoint loading succeeded around 16:18:40 UTC and KDA warmup began with `(32768, 24, 128, 8, -5.0, torch.bfloat16)`. Roughly 20 minutes from worker start is close to the previous attempt's 18m18s checkpoint-loading baseline. At 16:09 UTC the head-node GPUs each used 53,533 MiB; this is a loading snapshot, not a training peak.
- Corrected stale TP8/CP4 comments and the validation helper's 64k default; the running experiment's configuration is unchanged.
- Fixed the optional `--concurrency-per-engine` argument, which previously passed an unknown recipe field. It now sets the SGLang request limit, server concurrency, and five KDA cache slots per request. Only the default concurrency of one is under GPU validation; larger overrides have only been checked for local configuration validity.
- Added `preflight.py` and `preflight.json`: real cached K3 tokenizer and four DAPO rows, mocked HTTP generation. Five message-boundary cases each produced 261,128 total tokens, with only real generated tokens unmasked. Checked finite logprobs, length alignment, original-problem rendering, and reward-wrapper isolation from the observation/prompt. This is **not** GPU validation or validation of the upstream math scorer.
- Local checks: recipe/config instantiation (128 GPUs), Ruff, and 18 tests across `test_miles_recipe_hooks.py`, `test_miles_runtime_env.py`, and `test_miles_patches.py` passed.

## Still open

- A completed 2-step proof: nonzero reward, near-256k sequence lengths, finite log probs, correct loss masks, checkpoint save, and a second rollout on the updated adapter.
- The 10-step smoke run (`--rollouts 10`).
- GPU and host memory peaks at 256k.
