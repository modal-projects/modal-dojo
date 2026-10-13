# GLM-5.3-Flash bridge LoRA @ 256k context on 8x H200:8: experiment report

Status: **not yet validated.** At 256k, rollout, weight sync and log-prob work. The training
step still OOMs in the backward pass on the hottest expert-parallel rank. The ~10-step smoke
run has not been launched.

All runs: workspace `modal-labs`, environment `helena-dev`, 1-step proofs launched with
`validate_256k.py --steps 1 --launch`. Detailed notes: `model_setup.md`.

## Recipe

`GLM_5_3_Flash_LoRA_256k_Recipe` (`modal_dojo/train_recipes/miles_recipe/glm_5_3_flash_256k.py`)
subclasses `GLM_5_3_Flash_LoRA_Recipe`. The image, Miles ref and SGLang ref are unchanged.

| | base (#639) | 256k |
|---|---|---|
| nodes | 3x H200:8 | 8x H200:8 |
| TP / SP / PP / CP | 8 / on / 1 / 1 | 8 / on / 1 / 1 |
| EP | 24 | 32 (288 experts / 32 = 9 per rank) |
| seq_length, max_tokens_per_gpu, sglang + rollout context | 32768 | 262144 |
| rollout batch | | 2 prompts x 4 samples = 8 (one 256k sample per DP rank) |

CP must stay 1: the pinned DSA kpool selector raises `NotImplementedError` when CP > 1, and
upstream also hardcodes `--context-parallel-size 1`. So each TP8 group holds a full 256k
sequence, and the extra nodes only add DP and expert sharding.

## Image patches (applied only by the 256k recipe)

| patch | target in image | what it does |
|---|---|---|
| `patch_glm_5_3_flash_kpool_row_chunk.py` | `megatron/bridge/models/glm5_next/kpool_indexer.py` | Computes DSA indexer logits 8192 query rows at a time (2 GiB fp32 per chunk instead of 64 GiB); per-row top-k is unchanged. Falls back to the original path when indexer replay is on. |
| `patch_glm_5_3_flash_moe_row_chunk.py` | `megatron/core/transformer/moe/experts.py` | Runs grouped-expert forward over 262144-row chunks with per-chunk expert splits, writing into a preallocated output. |
| `patch_glm_5_3_flash_moe_seq_chunk.py` | `megatron/core/transformer/moe/moe_layer.py` | Routes once over the full sequence (routing replay order and shapes unchanged), then runs preprocess/dispatch/experts/combine over a fixed 8 token chunks. When grad is enabled, each chunk is wrapped in `tensor_parallel.checkpoint`. Supports allgather and alltoall dispatchers. Logs `GLM256K_MOE_SEQ_CHUNK` once per layer. |

Local checks (CPU glue tests against the unchunked path with fake dispatchers): forward and
gradients match to float rounding (max diff <= 3.1e-5 expert chunking; <= 1.3e-6 MoE-layer
chunking). Each patch applies to the image's real source file, is idempotent and compiles.

## Proof runs

| # | run | change vs previous | rollout reward | log-prob | result |
|---|---|---|---|---|---|
| 1 | [plain-shepherd-9800a782c6b1](https://modal.com/apps/modal-labs/helena-dev/ap-WPHLtEmwxZn2SlPt9G9DYt) | baseline 256k recipe | n/a | n/a | harness bug: dapo reward returns a dict. Rollout 151.8s, weight sync 39.6s, SGLang context_len 262144 |
| 2 | [short-cafe-fcac0d70b394](https://modal.com/apps/modal-labs/helena-dev/ap-KgsH5uaUuPdBVsLNKOO2AX) | reward_key, 8k response budget | -1.0 (harness bug: trailing `<\|user\|>`) | n/a | MoE OOM 16.08 GiB (repeated filler text -> routing imbalance) |
| 3 | [smart-caulk-bd72a9deae58](https://modal.com/apps/modal-labs/helena-dev/ap-nnf5sftRnjiXcg5wERjU0V) | varied DAPO filler, reward strips `<\|user\|>` | -0.5 | OOM | DSA indexer OOM 64.00 GiB (`tilelang_indexer_fwd.py:149`) |
| 4 | [mean-cobblestone-1c1b18810c7e](https://modal.com/apps/modal-labs/helena-dev/ap-aQYTUGiGuKvPDUgy69Ecpp) | kpool row chunk (miles_plugins copy) | -0.75 | OOM | same indexer OOM: Bridge mode uses `megatron/bridge` copy |
| 5 | [worn-carriage-e2fb1adcb178](https://modal.com/apps/modal-labs/helena-dev/ap-ErctRCZSbIaGzUmva7AGkL) | kpool patch retargeted to Bridge | -0.5 | OOM | indexer fixed; LoRA expert adapter OOM 13.63 GiB (118.8 GiB allocated) |
| 6 | [exponential-ascender-9eeb460e74fc](https://modal.com/apps/modal-labs/helena-dev/ap-T7yf8G26eoKQl6zQFbDdYJ) | expert row chunk | -0.75 | pass, 149.2s | train OOM in patch's `torch.cat` (13.61 GiB) |
| 7 | [tart-file-746505c84a49](https://modal.com/apps/modal-labs/helena-dev/ap-KPMbapKRvfHMxjFLRTsI2p) | preallocated chunk output | -0.5 | pass, 165.9s | train OOM on 2.00 GiB chunk, 117.55 GiB allocated |
| 8 | [blocky-number-be6798191273](https://modal.com/apps/modal-labs/helena-dev/ap-dbgCgv97eY6sMLokp3azBU) | MoE-layer seq chunk (allgather only) | -0.5 | pass, 159.6s | patch never engaged: Bridge LoRA forces the alltoall dispatcher; same OOM as proof 7 |
| 9 | [warped-darter-c4cdea978db2](https://modal.com/apps/modal-labs/helena-dev/ap-qmk5PXtKCUHKzCwSky4Upx) | seq chunk supports alltoall, fixed 8 chunks | -0.25 | pass, 179.1s, peak alloc ~48 GiB (was 62-79) | train forward passes; backward OOM (708 MiB, 120.37 GiB allocated) in full-recompute re-forward |
| 10 | [polite-plugin-2333a8f16b0e](https://modal.com/apps/modal-labs/helena-dev/ap-AdijxrI5xDE8Ju4emlXhRm) | per-chunk checkpoint under grad | -0.75 | pass, 169.0s | backward OOM inside a single chunk's recompute (976 MiB, 120.93 GiB allocated) |
| 11 | [presto-control-1be7107093e8](https://modal.com/apps/modal-labs/helena-dev/ap-5In4xIMiynodpShT6mDo0l) | diagnostics only (backward memory per MoE layer/chunk, allocation summary on OOM) | -0.75 | pass, 139.5s | backward OOM (836 MiB) on rank 13 at the 3rd layer of backward; allocation breakdown below |

Measured across proofs 3-10: rollout 190-194s for 8 samples with ~254k-token prompts, weight
update 39-49s, 1-3 of 8 responses truncated at the 8k response budget, coherent math reasoning
ending in `Answer: \boxed{N}`.

## What is proven vs open

Proven:
- At 256k, SGLang rollout, LoRA weight sync and Megatron log-prob all complete on 8x H200:8.
- After whole-MoE chunking, the training forward completes (proof 9).
- In proof 10 the backward OOM happens while only one chunk's expert activations should be live.
  At that point the rank already had ~121 GiB allocated, against a 34.7 GiB baseline before the
  training step.

Proof 11: what holds memory in backward (rank 13, the OOM rank). Live allocations at the OOM,
grouped by allocating call site, from `GLM256K_OOM` (117.2 GiB listed of ~119 GiB allocated):

| GiB | allocations | call site |
|---|---|---|
| 44.00 | 44 | `megatron/core/transformer/hyper_connection.py:144 native_h_post_bda` (inductor kernel), 1 GiB each |
| 21.24 | 2417 | allocated before history recording started (weights, LoRA/optimizer state, etc.) |
| 14.12 | 9 | `tensor_parallel/mappings.py:145 _gather_along_first_dim` via `gather_from_sequence_parallel_region` |
| 10.50 | 42 | `hyper_connection.py:71 native_h_aggregate` (inductor kernel), 0.25 GiB each |
| 4.00 | 2 | `hyper_connection.py:332 _projection_and_get_norm` |
| <= 2.00 each | | everything else; every MoE chunk allocation is <= 0.82 GiB |

`GLM256K_BWD_MOE`: rank 13 already had 105.5 GiB allocated when backward re-ran the last MoE
layer, and it ran out of memory two layers later. Its chunks there had 99k-107k rows (the average
is ~32k), at 114.6 GiB allocated.

So the MoE layer is no longer what fills memory. Inference, not proven: 1 GiB =
32768 local tokens x 4096 hidden x hc_mult 4 x 2 B. The 44 x 1 GiB group is most likely the mHC
(hyper-connection) residual stream that full recompute saves at each layer boundary. The
0.25 GiB-per-layer group matches one 32768 x 4096 bf16 aggregated hidden state per layer. Together
they are ~55 GiB of per-layer state that grows linearly with sequence length. The 14 GiB of
sequence-parallel all-gathers are not attributed to a specific module yet.

## Harness

`validate_256k.py` builds the 256k probe dataset. Each DAPO-math question is preceded by
~1,900 other DAPO problems as filler (~254k prompt tokens, no loss on the filler). Repeated
filler text causes artificial MoE routing imbalance and should not be used. Rewards use dapo
`compute_score(...)["score"]` with the trailing `<|user|>` stripped. Memory hooks print
`GLM256K_MEMORY` before log-prob and before the training step.

## Continuation (2026-10-13)

Proof 11 failed at 15:45:59 UTC and its app stopped at 15:46:56 UTC.
Rollout 191.4s, weight publication 44.5s, log-prob 139.5s. Mean total
sequence length 258990.9; mean response 5117.9; four of eight responses
truncated, one scored correctly. Exact memory diagnostic lines are saved
in `proof11_memory.log`.

### Proof 12: CPU offload of outer recompute checkpoints (failed)

`patch_glm_5_3_flash_checkpoint_cpu.py` wraps only the BF16 full-layer
`tensor_parallel.checkpoint` call in `torch.autograd.graph.save_on_cpu(pin_memory=True)`.
The initial checkpoint forward runs under `no_grad`, so its saved inputs move to host
memory. The scope exits before backward; inner MoE checkpoint recomputation remains
on GPU. This targets ~44–45 GiB/rank of residual streams without changing sequence
length, routing, batch size, parallelism, or model math. Expect roughly 360 GiB/node
of additional host checkpoint storage at eight GPUs/node; the recipe already allows
2 TiB host RAM/node. Full-model measurements are recorded below.

Validation before launch:
- Patch applies to `transformer_block.py` read from the pinned image, is idempotent,
  and the patched source compiles.
- [Single-H200 check](https://modal.com/apps/modal-labs/helena-dev/ap-bvryTqfNWFRgSv0mBbxAvf)
  used the image's real Megatron checkpoint function, six outer checkpoints, nested
  checkpointing, and dropout. Outputs and every input/parameter gradient matched
  exactly (`rtol=atol=0`). Retained forward GPU tensors fell from 100663296 bytes
  (96 MiB) to 16777216 bytes (16 MiB). The helper completed and stopped.
- Reproducer: `validate_checkpoint_cpu.py`. Image-source inspection used CPU-only
  helpers (`inspect_image.py`); these also completed.
- Proof 11 trace step 0 downloaded and inspected: final questions remain after the
  filler; responses attempt those questions. Some reasoning is erroneous or
  rambling, and four responses truncate. This proves workload shape, not quality.

Full-model one-step proof: [saucy-muscat-8124df5b55c1](https://modal.com/apps/modal-labs/helena-dev/ap-aS49sdjPArQuFoitFCJGW4),
created 2026-10-13 16:20 UTC. Same 8x H200:8 / TP8 / EP32 / PP1 / CP1
and workload as proof 11. Build log confirms `PATCHED_MODAL_DOJO_GLM53_CHECKPOINT_CPU`
applied to `/root/Megatron-LM/megatron/core/transformer/transformer_block.py`.
No full-model success is claimed yet.

Local preflight passed after installing the missing `jinja2` dependency in the
isolated virtualenv. Recipe import, dataset construction, patch compile/idempotence,
ruff lint/format and 18 recipe-hook/runtime/patch tests passed.

Proof 12 outcome: failed at 16:55:41 UTC during backward at layer 23/rank 47,
versus layer 43/rank 13 in proof 11. Rollout 185.7s, weight sync 42.2s,
log-prob 138.6s; all eight rewards -1, two truncated. Traces show incorrect
answers/unfinished reasoning for the intended final question, not the prior
trailing-token reward bug. No optimizer step completed.

Offload engaged: first backward MoE entry was 60.55 GiB (proof 11: 105.50 GiB).
However, same-rank memory increased from 60.55 GiB at layer 45 to 102.64 GiB
at layer 23, about 2 GiB per layer. OOM requested 1.95 GiB with 107.67 GiB
allocated and 15.39 GiB reserved but unused. Snapshot evidence:

| GiB | count | allocation |
|---:|---:|---|
| 22.46 | 23 | restored checkpoint inputs, `CheckpointFunction.backward` |
| 21.48 | 22 | mHC residual gradients, `native_fused_add_3` |
| 21.24 | 2409 | pre-history model/optimizer tensors |
| 13.79 | 9 | sequence-parallel gathers |
| 11.23 | 46 | aggregated hidden states (forward and recompute) |

Exact lines: `proof12_memory.log`. This suggests per-layer autograd graph
retention, rather than a single oversized expert chunk. The all-to-all dispatcher
keeps `self.probs` after combine, including its graph during recompute. Candidate
13 invokes the pinned dispatcher's `reset_transient_forward_state()` after each
completed chunk. Autograd retains its own saved tensors, and the next dispatch
reconstructs the transient metadata. A two-GPU regression checks this hypothesis
with the real dispatcher, nested checkpointing and differentiable routing before
launching another eight-node proof.
