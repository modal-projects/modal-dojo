# GLM-5.3-Flash LoRA @ 256k context, H200

Base: `GLM_5_3_Flash_LoRA_Recipe` (modal-dojo PR #639): 3x H200:8, TP8/SP, EP24, PP1, CP1, 32k.
Image/refs unchanged: `radixark/miles:glm53next@sha256:66725f74...`, miles `5a5353d3`, sglang `94c97e3a`.

## Model facts (HF config zai-org/GLM-5.3-Flash)
- 45 layers: 34 KDA linear-attention, 11 DSA (MLA kv_lora 512, indexer 32 heads x 128, kpool=4, topk=2048)
- 288 routed experts (moe_intermediate 2048), top-8, 1 shared; first 3 layers dense
- hidden 4096, mHC hc_mult=4, vocab 154880, max_position_embeddings 1048576 -> 256k needs no RoPE change

## Blocker: no context parallelism
`miles_plugins/models/glm5_next/dsa.py::_kpool_select` raises NotImplementedError when CP>1
(upstream `scripts/run_glm5_3_flash_lora.py` hardcodes `--context-parallel-size 1`).
So one TP8 group must hold a full 262144-token sequence. More nodes add DP / shard experts; they do
not reduce per-GPU sequence memory.

## Topology
8x H200:8 = 64 GPUs. TP8 + SP (intra-node), PP1, CP1, EP32 (288/32 = 9 experts/rank; 288/64 not integral),
expert-DP 2, DP8. SGLang: one TP8/EP8 engine per node (8 engines).
Batch: 2 prompts x 4 samples = 8 = one 256k sample per DP rank.

## Per-GPU training memory estimate (train phase, rollout offloaded)
- frozen expert weights 9 x 42 x 25.2M x 2B ~= 19 GiB, + attention/dense/shared TP-sharded few GiB
- full-recompute checkpoints: 32k tok/rank (SP) x 4096 x hc4 x 2B ~= 1 GiB/layer x 45 ~= 45 GiB
- DSA kpool indexer logits (no_grad, transient): index_q is SP-gathered -> fp32 [262144, 65536] = 64 GiB
  per DSA layer call (`tilelang_indexer_fwd.indexer_fwd_interface` allocates [seq, seq_kv] fp32)
- output logits vocab shard 19360 x 262144 tokens: ~10 GiB bf16 / ~20 GiB fp32
Total peak plausibly 130-150 GiB vs 141 GiB H200 -> **OOM risk dominated by the indexer logits**.
If the proof OOMs there, the upstream-style fix is to chunk `kpool_select_topk` over query rows
(topk is row-independent, so exact) rather than raising memory.

## Validation workload
Capacity probe: DAPO-math prompt with ~257k tokens of filler prepended in the user turn (prompt
tokens, no loss), `reasoning_effort=low`, response <= 4096 tokens (capped by rollout_max_context_len),
deepscaler reward on the response. Exercises 256k SGLang prefill + 256k Megatron fwd/bwd/logprob.
Memory reported via before_log_prob / before_train_step hooks.

## Expected timings (estimates, to be replaced by measured)
- image build: cached layers + bridge pip ~5-10 min; model load/convert ~15-25 min
- rollout: 256k prefill per engine ~1-3 min + <=4k decode ~1-2 min
- ref/old logprob: ~1-3 min; train fwd/bwd one 256k sample per DP rank ~3-8 min
- weight (LoRA) sync: < 1 min
- step total ~8-15 min

## Log
- proof1 plain-shepherd-9800a782c6b1 (ap-WPHLtEmwxZn2SlPt9G9DYt): SGLang up, max_total_num_tokens=1319936,
  context_len=262144. Rollout of 8 x ~258k prompts: perf/rollout_time 151.8s, response mean 2301 / max 4099,
  truncated_ratio 0.5. Weight sync 39.6s. FAILED in _normalize_rewards_by_rollout: rm_type=dapo returns a
  dict -> need reward_key=score. Harness bug, not recipe. Fix: extra_config reward_key=score; response
  budget 4096 -> 8192 (prompts ~254k) to cut truncation.
- proof2 short-cafe-fcac0d70b394 (ap-KgsH5uaUuPdBVsLNKOO2AX): rollout 189.8s, response mean 3083 / max 8194,
  truncated_ratio 0.25, episode_raw_reward -1.0 (all 8 samples scored -1; no learning signal).
  before_log_prob: 27.8 GiB allocated / 32.4 GiB peak per rank. FAILED: OutOfMemoryError in train()
  (actor.py:572 train_actor) -> MoE routed_experts_compute -> bridge LoRA _forward_te_grouped_linear ->
  TE grouped_linear torch.empty of 16.08 GiB; 97 GiB allocated + 15 GiB reserved-unallocated on rank 52.
  Not the DSA indexer.
  Responses (trace step_0000.json) are coherent math reasoning ending "Answer: \boxed{N}<|user|>"; prompts
  253948-253949 tokens. Reward -1 for all is a harness bug: dapo compute_score on "Answer: \boxed{34}<|user|>"
  vs "34" -> pred "34<|user|>", -1.0; same string without the trailing <|user|> -> +1.0 (local check).
  OOM inference (unproven): moe_ffn_hidden=2048 -> fc1 out 4096 bf16/row; 16.08 GiB ~= 2.1M token-expert rows on
  one EP rank vs ~262k if balanced. Likely the ~250k copies of one repeated filler line all routing to the same
  experts. Fixes for proof3: filler = ~1900 other DAPO problems (varied text, ~254k tokens/prompt); custom
  dapo_rm strips the trailing <|user|> before compute_score.
- proof3 smart-caulk-bd72a9deae58 (ap-nnf5sftRnjiXcg5wERjU0V): varied filler + reward fix. rollout 194.4s,
  episode_raw_reward -0.5 (nonzero signal), truncated_ratio 0.25. Got past the MoE layer (supports the
  routing-imbalance explanation for proof2). FAILED: OOM 64.00 GiB in megatron/bridge/models/glm5/tilelang/
  tilelang_indexer_fwd.py:149 logits=torch.empty([seq_len, seq_len_kv], fp32); 36.9 GiB allocated, 21.5 GiB free.
  This is the predicted DSA indexer blocker -> needs row-chunked indexer.
  Fix: patch_glm_5_3_flash_kpool_row_chunk (256k recipe only) scores kpool logits in 8192-row chunks when
  indexer replay is off (2 GiB fp32 per chunk vs 64 GiB); per-row topk/expand is unchanged, so selection is exact.
- proof4 mean-cobblestone-1c1b18810c7e (ap-aQYTUGiGuKvPDUgy69Ecpp): patch applied in image, but rollout 190.2s,
  reward -0.75, truncated 0.125, then same OOM (61.04 GiB) at tilelang_indexer_fwd.py:149 called from
  megatron/bridge/models/glm5_next/kpool_indexer.py:321 via bridge dsa.py:279. Bridge mode uses Bridge's copy of
  the kpool indexer, not miles_plugins -> patch must target the Bridge file.
  Fix: patch retargeted to megatron/bridge/models/glm5_next/kpool_indexer.py (Bridge 6527b18), same row chunking.
- proof5 worn-carriage-e2fb1adcb178 (ap-ErctRCZSbIaGzUmva7AGkL): Bridge patch applied; indexer no longer OOMs.
  rollout 194.3s, reward -0.5, truncated 0.375. FAILED in compute_log_prob: MoE routed_experts_compute ->
  bridge LoRA adapter `expert_output * (alpha/dim)` tried 13.63 GiB with 118.8 GiB already allocated (rank47).
  13.63 GiB / (4096 * 2 B) ~= 1.79M rows on one EP rank vs ~262k balanced -> expert load imbalance persists
  with varied filler (inference). Next options: row-chunk the expert/LoRA compute, or lower context.

proof6 exponential-ascender-9eeb460e74fc (ap-T7yf8G26eoKQl6zQFbDdYJ):
Adds patch_glm_5_3_flash_moe_row_chunk (grouped experts forward over 262144-row
chunks, per-chunk expert splits). Local CPU equivalence test vs unchunked: fwd
max diff 7.6e-6, grad max diff 3.1e-5. Build log confirms both patches applied
(/root/Megatron-LM/megatron/core/transformer/moe/experts.py).
Result: rollout 191.8s, raw reward -0.75, response mean 3108 / max 8295,
truncated 0.25, mean total length 256981. Weight update 39.2s.
log_probs PASSED (149.2s); before_train allocated ~37 GiB, peak_reserved up to
123 GiB. Train step forward (inside full-recompute checkpoint, MoE
routed_experts_compute) OOMed at the patch's torch.cat of chunk outputs:
13.61 GiB requested, 112.68 GiB allocated, 5.09 GiB free (another rank OOMed on
a 2 GiB chunk alloc with 117 GiB allocated). 13.61 GiB = ~1.78M token-expert
rows at width 4096 on one EP rank (avg ~262k).
Next: write chunks into a preallocated output (no sum-of-chunks + cat peak).

proof7 tart-file-746505c84a49 (ap-KPMbapKRvfHMxjFLRTsI2p): preallocated-output MoE chunking.
Result: rollout reward -0.5, response mean 4754.5 / max 8295. Weight update 45.6s.
log_probs PASSED again (165.9s). before_train allocated ~37 GiB, peak_reserved
up to 123.8 GiB. Train step forward (full-recompute checkpoint, MoE
routed_experts_compute -> chunked experts forward, experts.py:925) OOMed on a
2.00 GiB chunk allocation: 117.55 GiB allocated by PyTorch, 137.48 GiB in use,
2.21 GiB free. So the full-output cat was not the binding cost; the hot EP rank's
train forward baseline (~117 GiB) leaves no room for even one 2 GiB chunk.

proof8: adds patch_glm_5_3_flash_moe_seq_chunk (moe_layer.py). After the single
router call (routing replay order/shape unchanged), allgather-dispatched MoE runs
preprocess/dispatch/experts/combine/combine_postprocess over ceil(tokens/4096)
equal token chunks (8 at 32768 local tokens). Token count is identical across the
TP*EP group, so every rank issues the same number of all-gather/reduce-scatter
calls with equal sizes. Gated off for non-allgather dispatchers, partial
fwd_execution_map, latent MoE, delayed wgrad, inference mode. Verified: patches the
image's moe_layer.py, idempotent, compiles; CPU glue test vs unchunked: fwd diff 0,
grad diff 4.8e-7.

proof8 blocky-number-be6798191273 (ap-dbgCgv97eY6sMLokp3azBU): build log confirms
PATCHED_MODAL_DOJO_GLM53_MOE_SEQ_CHUNK applied. Rollout reward -0.5, response mean
4120; update_weights 44.8s; log_probs PASSED (159.6s); before_train allocated ~37 GiB,
peak_reserved up to 122.9 GiB. Train forward OOMed on a 2.00 GiB alloc (117.77 GiB
allocated) via moe_layer.py:776 -> the ORIGINAL expert_compute block, i.e. the chunk
gate returned 1. Root cause (image source): Bridge LoRA mode forces
provider.moe_token_dispatcher_type = "alltoall"
(/root/miles/miles/backends/megatron_utils/bridge_lora_helpers.py:149), and the GLM5
bridge sets moe_shared_expert_overlap=True / moe_permute_fusion=True
(megatron/bridge/models/glm5/glm5_bridge.py:256-262). The Miles args print
(moe_token_dispatcher_type=allgather) does not reflect the built model.
Fix for proof9: gate also accepts MoEAlltoAllTokenDispatcher (no drop_and_pad, no
cuda graphs) and uses a FIXED 8 chunks (token counts differ across EP ranks, so a
row-derived count could desync collectives). Shared-expert overlap runs inside the
dispatcher per chunk (per-token MLP, TP shards split equally). CPU glue test with a
fake alltoall dispatcher + shared expert: fwd diff 0, grad diff ~1e-7. Gate decision
now logged once per layer as GLM256K_MOE_SEQ_CHUNK.

proof9 warped-darter-c4cdea978db2 (ap-qmk5PXtKCUHKzCwSky4Upx): alltoall-aware fixed-8
MoE seq chunking. Logs confirm the gate engaged on every MoE layer:
GLM256K_MOE_SEQ_CHUNK chunks=8 shape=(32768|32640, 1, 4096)
dispatcher=MoEAlltoAllTokenDispatcher shared_overlap=False.
Rollout reward -0.25 (response mean 4389); update_weights 48.5s; log_probs PASSED
(179.1s). before_train: allocated ~34.4 GiB, peak_allocated ~48 GiB, peak_reserved
~66 GiB (proof7/8: peak_allocated 62-79 GiB, peak_reserved ~123 GiB), so whole-MoE
chunking roughly halved log-prob peak memory.
Train step OOMed inside the chunked path (moe_layer.py:949 _modal_dojo_chunked_moe ->
routed_experts_compute -> experts.py:822): 708 MiB requested, 120.37 GiB allocated by
PyTorch, 1.09 GiB free (rank on 10.100.0.7 GPU 5).
Full proof9 traceback: the OOM is in BACKWARD, not the training forward
(schedules.py:576 backward_step -> tensor_parallel/random.py:702 CheckpointFunction.backward
-> full-recompute layer re-forward with grad -> chunked MoE -> Bridge peft/utils.py:1532).
The training forward now fits. In the recompute re-forward every chunk's expert
activations stay saved for autograd, so the hot rank reaches the unchunked total.
Fix for proof10: when grad is enabled each chunk (preprocess/dispatch/experts/combine/
combine_postprocess) is wrapped in tensor_parallel.checkpoint, so backward holds one
chunk's expert activations at a time. Collectives are re-issued per chunk in backward
in the same order on every rank (same graph on every rank; Megatron's own
moe_layer_recompute also checkpoints dispatch collectives). Cost: one extra routed-MoE
forward per layer in backward. CPU glue test (fake alltoall + reentrant checkpoint,
.backward()): fwd diff 2.4e-7, grad diff 7.2e-7, 8 chunk recomputes in backward, no-grad
path unchanged.

proof10 polite-plugin-2333a8f16b0e (ap-AdijxrI5xDE8Ju4emlXhRm): per-chunk checkpoint.
Rollout reward -0.75 (response mean 4785); update_weights 40.5s; log_probs PASSED
(169.0s); before_train allocated ~34.7 GiB, peak_allocated ~48.7 GiB, peak_reserved ~66-68 GiB.
Backward OOM INSIDE a single chunk's recompute (random.py:711 -> random.py:702 inner
CheckpointFunction.backward -> moe_layer.py:957 lambda -> _modal_dojo_moe_chunk ->
experts.py:925 linear_fc2 -> Bridge peft/utils.py:1532): 976 MiB requested, 120.93 GiB
allocated by PyTorch, 138.12 GiB in use, 1.58 GiB free (10.100.0.5 GPU 1).
Per-chunk checkpointing engaged, yet the rank was already at ~121 GiB while only one
chunk's expert activations should be live; what holds the other ~86 GiB above the
before_train baseline is NOT yet measured (needs per-chunk memory logging / snapshot).
