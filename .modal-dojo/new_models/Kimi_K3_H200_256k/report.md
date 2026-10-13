# Kimi-K3 LoRA 256k on H200: validation report (in progress)

Status: **not validated yet**. No training step has finished. Recipe: `Kimi_K3_LoRA_256k_H200_Recipe` (16 x H200:8, TP16 inference, 256k context). Dashboard: https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run (env `helena-dev`).

## Runs

| Run | App | Trainer layout | Result |
| --- | --- | --- | --- |
| [fixed-fathom-65c285afb988](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/fixed-fathom-65c285afb988) | `ap-J7DSUkrFG6SgG0tfvKAgWH` | TP8 x CP4 x PP4, EP32, 65536 tok/GPU, routing replay on | Cancelled after SGLang engine init crash |
| [stable-plaster-5f9920d29d31](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/stable-plaster-5f9920d29d31) | `ap-sqrCXIKbt2SE5qaJrFiYEg` | TP8 x CP4 x PP4, EP32, 65536 tok/GPU, routing replay off | FAILED at Optimizer step, step 1/2: CUDA OOM in trainer forward |
| [crimson-brass-1832e798c59a](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/crimson-brass-1832e798c59a) | `ap-GyGeQjKO0yA5WcZicYxHvw` | TP4 x CP8 x PP4, EP32, 32768 tok/GPU, routing replay off | Running (startup) |

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

## Still open

- A completed 2-step proof: nonzero reward, near-256k sequence lengths, finite log probs, correct loss masks, checkpoint save, and a second rollout on the updated adapter.
- The 10-step smoke run (`--rollouts 10`).
- GPU and host memory peaks at 256k.
