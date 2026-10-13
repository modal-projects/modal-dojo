# Kimi-K3 LoRA, 256k context, H200

Base: `Kimi_K3_LoRA_64k_Recipe` (modal-dojo#618), image `radixark/miles:dev-202609251434` (K3 support in Miles/Megatron/SGLang; unchanged).

The running `crimson-brass` image reports Miles commit `41c5e38b94ea23677de93b01a4a77d55677a8f09`. Its `miles_plugins/models/cp_utils.py` accepts arbitrary CP group sizes provided padded sequence lengths divide CP, relayouts Megatron zigzag shards via all-gather, and calls FLA `build_cp_context`. This removes the earlier uncertainty about using upstream HEAD for the source inspection; it does not prove CP8 execution or memory capacity. The full-sequence relayout buffers mean that doubling CP does not halve every activation allocation.

## Evidence driving topology
- radiant-tom-b8a5302e43ee (64k, 8x B300:8, TP4/PP8/CP2/EP8): host used 2.20-2.34 TiB/node of 3 TiB; GPU resident ~82-107 GiB, peak allocated 146.6 GiB.
- H200: 140.4 GiB/GPU, 2 TiB host. Compact TP16 MXFP4 weights measured 111.5 GiB/rank (adagio-hull, #618 history).
- concave-frisbee-8a15d3a104f0 (H200 64k): OOM in the unbounded PP adapter broadcast.

## 8 nodes rejected
- Host: TP16 inference backup ~0.9 TiB/node + 1/8 of the 5.6 TB BF16 trainer ~0.7 TiB/node + overhead (~0.4 TiB in radiant-tom) > 2 TiB.
- GPU: 256k with TP x CP <= 8 at 64 GPUs is 4x the 64k per-rank sequence on half the memory.

## Chosen layout (16 x H200:8)
- Trainer TP4 x CP8 x PP4 (23/24/24/22 layers), EP32, DP1, `max_tokens_per_gpu=32768` (32k tokens/rank, same as radiant-tom at 64k). First tried TP8 x CP4 (64k tokens/rank), which OOMed; see `report.md`.
- Inference: 8 TP16 engines, one 256k request each, `sglang_max_total_tokens=524288` (fixed-fathom allocated 402,497 tokens / 10.36 GB per rank), mem fraction 0.95.
- Batch: 2 prompts x 4 samples = 8 sequences = one per engine.
- Patches: 64k set + compact TP16 Marlin padding + bounded PP adapter broadcast.

## Validation
`validate.py --context-length 262144 --rollouts {2,10} --launch` (capacity mode: masked observation fills each sequence to >= 261120 tokens).

## Observed timing baseline
- `stable-plaster`: queued 04:09:44–06:55:44 UTC (2h46m), checkpoint loaded at 07:14:02 (18m18s after worker start), trainer OOM at about 07:58 (roughly 62m after worker start). Its 3h54m submission-to-failure time should not be treated as all initialization or training.
- `crimson-brass`: queued 15:09:48–15:58:45 UTC (48m57s), checkpoint loaded around 16:18:40 (roughly 20m after worker start), then KDA warmup used `(32768, 24, 128, 8, -5.0, torch.bfloat16)`. No measured training-step duration yet.
