# Kimi-K3 LoRA 256k on H200: validation report (in progress)

Status: **not validated yet**. No training step has finished. Recipe: `Kimi_K3_LoRA_256k_H200_Recipe` (16 x H200:8, TP16 inference, 256k context). Dashboard: https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run (env `helena-dev`).

## Runs

| Run | App | Trainer layout | Result |
| --- | --- | --- | --- |
| [fixed-fathom-65c285afb988](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/fixed-fathom-65c285afb988) | `ap-J7DSUkrFG6SgG0tfvKAgWH` | TP8 x CP4 x PP4, EP32, 65536 tok/GPU, routing replay on | Cancelled after SGLang engine init crash |
| [stable-plaster-5f9920d29d31](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/stable-plaster-5f9920d29d31) | `ap-sqrCXIKbt2SE5qaJrFiYEg` | TP8 x CP4 x PP4, EP32, 65536 tok/GPU, routing replay off | FAILED at Optimizer step, step 1/2: CUDA OOM in trainer forward |
| [crimson-brass-1832e798c59a](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/crimson-brass-1832e798c59a) | `ap-GyGeQjKO0yA5WcZicYxHvw` | TP4 x CP8 x PP4, EP32, 32768 tok/GPU, routing replay off | FAILED: eight 262,140–262,142-token rollouts, reward 0.125; SiTU activation OOM in first training update |
| [linear-list-7f87c318b9b9](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/linear-list-7f87c318b9b9) | `ap-wr6wEkjAxIxyqTInBvRq2q` | TP4 × CP8 × PP4, EP32, fused SiTU, original filler | Submitted; no training result yet |

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

## 8 nodes

Not tried. It was ruled out by estimate (see `model_setup.md`), so it isn't proven to fail. Inference (not measured): stable-plaster OOMed on 16 nodes at 64k tokens/rank, and 8 nodes would need either 2x the tokens/rank or 2x the model shard per GPU.

## crimson-brass: 256k inference works; CP8 does not fix trainer OOM

- All eight rollout samples reached **262,140–262,142 total tokens**, with 1,276–1,278 generated tokens and the rest dominated by masked synthetic observation. Rollout time was **144.61 seconds**, mean reward **0.125** (one correct answer out of eight).
- Downloaded and inspected step 0 using `modal-dojo run trace`. The prompts and math reasoning are coherent; sample 2 finished with `Answer: \\(\\boxed{-3}\\)`. Other samples mostly exhausted the deliberately short final-generation window. This is a capacity stress test, not evidence of poor math capability or of 256k autonomous generation.
- Initial adapter sync succeeded. The log-probability pass completed in **219.1 seconds**. The training phase started at 17:02:10 UTC and OOMed at 17:03:38 before a completed update.
- Primary traceback: `miles_plugins/models/kimi_k3/ops.py:49`, `linear_beta * torch.tanh(linear / linear_beta)`, requested **4.93 GiB** with **6.46 GiB free**, **106.54 GiB allocated**, and **10.83 GiB reserved but unallocated**. Other ranks OOMed allocating **2.71 GiB** in expert FC2's LoRA `F.linear` during backward. Full per-layer recomputation was already enabled.
- Logged pre-training snapshots had 51–55 GiB allocated but as much as 119.28 GiB reserved; observed host use reached 1,298.22 GiB against a 1,920 GiB limit. These are sampled phase-boundary measurements, not final exhaustive peaks. The OOM traceback gives the later memory state.
- Run marked failed at 17:06:05 UTC. Modal app confirmed **stopped at 17:11:01 UTC**. No optimizer update, saved checkpoint, or second rollout was validated.
- Compact trace and memory evidence: `crimson_brass_result.json`. Raw downloaded trace: eight samples, 12,561,645 bytes; retained locally outside git.
- Follow-up experiment: benchmark the same SiTU math under `torch.compile` on **one H200**, including output/gradient comparisons and peak allocation at a shape matching the failed ~4.93 GiB FP32 half-tensor. Only use it in the full recipe if the focused test passes and demonstrates a memory reduction. Keep topology, dataset, and two-rollout horizon fixed.

## SiTU fusion proof (one H200)

- `ap-EJSzhXNieClMpp6LHx2olU`, completed 2026-10-13 17:23 UTC, image `radixark/miles:dev-202609251434`, PyTorch `2.13.0+cu130`. An earlier benchmark (`ap-hTejvGf4HbCT8vDvn1DQmR`) completed its GPU checks but failed to deserialize `torch.__version__` locally; converting that value to plain `str` fixed the result transport.
- `torch.compile(fullgraph=True, dynamic=True)` wraps the **unchanged upstream SiTU formula**. Small FP32/BF16 output and gradient comparisons passed at input scales 0.01, 1, 10, and 100; large-shape first/middle/last-row comparisons also passed. BF16 maximum observed differences were 0.03125 in output and 0.0078125 in gradient (within the stated tolerances); this is numerical agreement, not bitwise identity.
- At `[368640, 7168]` BF16 input (4.92 GiB per FP32 half), extra peak allocated memory for forward+backward dropped from **46.76 GiB to 7.44 GiB**; forward live intermediates dropped from **31.99 GiB to 2.46 GiB**. Baseline input/gradient allocation was 7.38 GiB in both cases. At 131,072 rows, forward+backward extra peak fell from 16.63 GiB to 2.63 GiB.
- The large compiled case includes first-shape compilation (~2.67s forward and 0.52s backward); these are not steady-state training timings or a speedup claim. Full results: `situ_benchmark.json`.
- Added `patch_k3_situ_compile` only to `Kimi_K3_LoRA_256k_H200_Recipe`, plus pinned input/output fixtures, idempotence/drift tests, and the snapshot refresh hook. **19 local Miles tests passed**, along with Ruff and recipe instantiation. Full 128-GPU validation remains required.

## linear-list: fused SiTU, original filler (running)

- Run [`linear-list-7f87c318b9b9`](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/linear-list-7f87c318b9b9), app `ap-wr6wEkjAxIxyqTInBvRq2q`, call `fc-01M4V4WMST2FGF7AFH7Z21WGAJ`, submitted 2026-10-13 17:25 UTC from `a49431b`.
- Two-rollout capacity test on the same 16×H200:8 topology, with fused SiTU. The image build printed `PATCHED_K3_SITU_COMPILE`; conversion reused the existing checkpoint. No training result yet.
- This run deliberately retains the **original repeated diagnostic filler**, to compare against crimson-brass with the activation implementation as the main change.
- Concurrent commit `d56e28d` changes the default probe to filler drawn from 4,096 varied DAPO problems. Its hypothesis is that repeated filler concentrates MoE routing, consistent with only two trainer actors OOMing; expert-token imbalance has **not been measured**. Both approaches are preserved here, but this running experiment predates the varied-filler change. Many crimson-brass ranks also logged non-fatal TorchMemorySaver allocation warnings for 2.9–11.1 GB requests.

## Continuation, 2026-10-13

- Source at takeover: `b5b5e3014aa3973b474187900e42a8dd08c8729e`.
- `crimson-brass` was queued, not converting: the conversion cache-hit log was already present while the dashboard still showed `Converting model`. Modal function call `fc-01M4TX0203JCE4DHJZGH7QBS8N` was enqueued at 15:09:48.429 UTC and started at 15:58:45.632 UTC (48m57s queue time). No replacement run was launched.
- Runtime resources verified through `modal function info`: 16-node RDMA cluster, 8 H200/node, 1,966,080 MiB host memory limit, no region/cloud restriction, no retries, 86,400s execution timeout.
- Trainer initialization reached rank-32 LoRA construction on all 128 ranks and began loading `/checkpoints/Kimi-K3_torch_dist` at 16:03:45 UTC. TP/PP reshards from the conversion layout `(32, 1)` to `(4, 4)`; the RNG-state warning is expected for that layout change. SGLang is gated while trainer initialization completes.
- Checkpoint loading succeeded around 16:18:40 UTC and KDA warmup began with `(32768, 24, 128, 8, -5.0, torch.bfloat16)`. Roughly 20 minutes from worker start is close to the previous attempt's 18m18s checkpoint-loading baseline. At 16:09 UTC the head-node GPUs each used 53,533 MiB; this is a loading snapshot, not a training peak.
- KDA forward/backward warmup completed across the cluster; the inference gate opened around 16:24 UTC and all eight TP16 engines began loading weights. Representative warmup forward/backward times were 25/79–104 seconds, with slower groups around 69/122 seconds.
- **Memory-rationale correction:** CP8 halves tokens before TP splitting, but halving TP doubles local attention heads. The observed KDA shapes have identical token-head products: `65536 * 12 == 32768 * 24`. With sequence parallelism, `262144 / (TP * CP)` remains 8,192, and EP32 is unchanged. Thus this is not a 2x reduction in overall activation memory; the full-sequence CP relayout buffers also remain. Whether enough other buffers shrink to avoid the MoE OOM is an empirical question for this run.
- Pinned-image checkpoint inspection confirms `save_checkpoint_with_lora` uses `iter_<iteration>/adapter` with optimizer/scheduler state; the base checkpoint is not the intended save payload.
- Corrected stale TP8/CP4 comments and the validation helper's 64k default; the running experiment's configuration is unchanged.
- Fixed the optional `--concurrency-per-engine` argument, which previously passed an unknown recipe field. It now sets the SGLang request limit, server concurrency, and five KDA cache slots per request. Only the default concurrency of one is under GPU validation; larger overrides have only been checked for local configuration validity.
- Added `preflight.py` and `preflight.json`: real cached K3 tokenizer and four DAPO rows, mocked HTTP generation. Five message-boundary cases each produced 261,128 total tokens, with only real generated tokens unmasked. Checked finite logprobs, length alignment, original-problem rendering, and reward-wrapper isolation from the observation/prompt. This is **not** GPU validation or validation of the upstream math scorer.
- Local checks: recipe/config instantiation (128 GPUs), Ruff, and 18 tests across `test_miles_recipe_hooks.py`, `test_miles_runtime_env.py`, and `test_miles_patches.py` passed.

## Still open

- A completed 2-step proof: nonzero reward, near-256k sequence lengths, finite log probs, correct loss masks, checkpoint save, and a second rollout on the updated adapter.
- The 10-step smoke run (`--rollouts 10`).
- GPU and host memory peaks at 256k.
