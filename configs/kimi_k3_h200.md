# Kimi-K3 LoRA on 64 H200s

This configuration is independently reviewable from the B300 recipe. It uses
`configs.kimi_k3_h200.build_recipe`; it does not import a B300 configuration.
The context budgeting and validation helpers are identical in the two PRs.
The original recipe class and checkpoint-volume identity are retained so the
existing converted BF16 checkpoint can be reused.

```python
from configs.kimi_k3_h200 import build_recipe
from modal_dojo import Kimi_K3, TrainConfig

config = TrainConfig(
    model=Kimi_K3(),
    dataset=your_dataset,
    recipe=build_recipe(num_rollout=2, rm_type="deepscaler"),
)
```

## Configuration

- Eight H200:8 nodes; BF16 rank-32 LoRA adapters and native MXFP4 inference.
- Transformer Engine trainer layers use FP8 hybrid with delayed scaling and
  FP8 parameter storage (`fp8_param_gather`); other layers remain BF16.
  Delayed scaling is required by the pinned optimizer CPU-offload path.
- Trainer TP2 x PP8 x CP4, DP1, EP8; four TP16 inference engines.
- Total context 65,536 tokens; one active request per engine.
- Training/log-probability packing budget 16,384 tokens per GPU at CP4.
- One prompt with four completions per update (`global_batch_size=4`) bounds
  the number of live pipeline microbatches; this is an H200 capacity default.
- Full uniform activation recomputation one layer at a time.
- PyTorch cache garbage-collection threshold 0.8 leaves space for Triton and
  communication allocations outside the PyTorch caching allocator.
- All optimizer state offloaded to CPU; active-model offload before restoring
  the other model; host RAM 1,792 GiB requested and 1,920 GiB maximum per node.
- FlashInfer MLA decoding and compact Marlin expert storage.

The generator subtracts the prompt/history and the scheduler's one-token
margin from the response budget. Agents must additionally bound complete tool
trajectories, preserve generated token IDs and logprobs, and mask observations
out of the loss. A smaller packing budget cannot shrink one long sequence.

## Memory and runtime fixes

Compact Marlin allocation preserves the supported 192-wide K3 TP16 expert
shard instead of padding it to 256. Streaming adapter updates use independent
CPU copies rather than accumulating a whole adapter on GPU. Generation health
checks select the one registered policy and release their leases.

The included pipeline-broadcast fix uses the existing 256 MiB transfer target
before allocating its buffer; it previously flattened a whole 5.96 GiB stage.
The frozen trainer checkpoint now merges directly on CPU, avoiding repeated
failed GPU allocations and cache/garbage-collection fallback work. This does
not enable `low_memory_resume` or change optimizer restoration.

FP8 checkpoint loading dequantizes one tensor at a time into CPU staging.
Previously, Megatron accumulated a second BF16 model on GPU before loading
the checkpoint. Frozen FP8 base parameters also release their initialization
copies on CPU when LoRA freezes them; trainable optimizer inputs are retained.

K3's original `situ_and_mul` formula is compiled with dynamic shapes to fuse
its FP32 intermediates. This addresses the activation allocation that failed
in the first full-model log-probability pass. Forward and backward numerical
comparisons use dtype-appropriate tolerances, rather than bitwise identity.

The expert LoRA add also fuses concatenation with addition, avoiding a
full-width delta temporary without modifying Transformer Engine's view output.
Its explicit linear backward uses no saved activations and avoids the extra
full-width gradient materialized by a compiler-generated backward.

## Validation evidence and limits

Earlier full-model attempts failed at GPU adapter staging and then a
pipeline-broadcast allocation. The subsequent `inverse-antagonist-c8d1f746f40e`
run completed eight near-64k samples but OOMed inside `situ_and_mul` while
computing log probabilities, before any optimizer update.

The H200 activation probe reproduced the eager OOM. At 221184 x 6144 BF16
input, peak temporary memory fell from 12.66 GiB to 1.32 GiB with compilation;
forward completed under an allocator cap where eager failed. BF16/FP32 output
and gradient comparisons passed for contiguous and strided inputs. The CPU
checkpoint-merge probe reproduced the 42 MiB allocation failure and verified
identical tensors with zero additional GPU allocation in four dtype/layout
cases. Other isolated GPU tests cover compact expert reload, adapter integrity,
health, 64k prefill/decode, release/resume, and bounded eight-rank broadcasts.

The next full-model attempt,
[relative-cottage-dcec7a81a761](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/relative-cottage-dcec7a81a761),
completed eight near-64k rollouts in 174 seconds and log-probability computation
in 199 seconds. It then failed in the first training pass: a 2.67 GiB expert
LoRA addition and 1.97 GiB backward allocations exhausted memory. No optimizer
update completed. App `ap-0pUTy1g6CGleSfRyHyO4gg` stopped with zero workers.

The first revised candidate kept 64 H200s and BF16 but used TP2/CP4 instead of
TP4/CP2 to reduce activation pressure. The pinned-runtime argument, tokenizer,
masking and reward preflight passes. A real H200 test of Transformer Engine
grouped linear plus the upstream K3 LoRA wrapper passes output, input-gradient
and adapter-gradient comparisons for BF16/FP32 and both zero/nonzero adapters.
At 233472 x 6144 BF16, fused-add forward temporaries fall from 5.34 GiB to
2.73 GiB; backward peak is unchanged at 8.02 GiB in this isolated test. The
eager allocation fails under a cap where the fused version succeeds. This is
not yet proof that the complete model fits during training.

That BF16 TP2/CP4 retry, `crispy-rake-2ba575441549`, produced eight near-64k
samples in 175 seconds, including one correctly rewarded boxed answer. Log
probabilities completed in 903 seconds, but a 1.40 GiB expert down-projection
LoRA addition then OOMed before an optimizer update. Its app stopped.

The next candidate adds FP8 hybrid/delayed scaling with FP8 compute parameters.
An isolated H200 proof using actual Transformer Engine grouped layers and K3
LoRA completed two forward/backward/AdamW cycles. It loaded BF16 parameter
values into FP8 tensors and verified exact values after CPU offload/resume.
Peak memory in this small block fell from 720 MiB to 579 MiB; FP8 parameter
initialization alone did not halve memory. Output relative RMSE was about 7.2%,
input-gradient RMSE 9.9%, and worst adapter-gradient RMSE 8.0% versus BF16.
This is compatibility evidence, not full-model numerical-quality validation.
Pinned-runtime FP8/CPU-offload argument and data/reward preflight returned
successfully; the helper logged a segfault during process teardown afterward.

The FP8 loader regression reproduces the original OOM with 96 MiB of extra
GPU allocation allowed. The patch peaks at 42 MiB, loads exact BF16 checkpoint
values, and copies them back into FP8 parameters within the tested quantization
tolerance. Clearing frozen initialization copies leaves parameter values intact.

A native 12-layer stage on eight H200s (TP2/CP4/EP8, 64k synthetic sequence)
failed backward with three-layer recomputation. One-layer recomputation passed
forward/backward on all eight ranks with finite outputs and 156 gradients per
rank; peak live PyTorch allocation was 113.38–114.16 GiB. This test excludes
optimizer/DDP, pipeline communication, real-weight routing skew and colocated
inference. Exercise multiple live microbatches before treating the single-stage
result as evidence for the full pipeline:

```bash
uv run modal run --env helena-dev --detach scripts/validate_k3_h200_stage_memory.py --extra-resident-gib 8
```

The expanded test caught an OOM with eight live microbatches. Each additional
forward graph retained about 1.64 GiB, motivating the four-completion default
(approximately 6.6 GiB less retained memory). Four microbatches without proactive
cache reclamation then failed during Triton backward autotuning. The next test
combined cache reclamation with 8 GiB of persistent allocations per GPU to
approximate omitted full-run overhead; the local network connection failed
before its result was collected, and no pass is claimed. The combined candidate
therefore still requires full-model validation. Local configuration/patch tests:
95 passing.

```bash
# Inspect the config locally:
uv run -m scripts.validate_kimi_k3_h200

# Allocate 64 H200s for the two-step capacity proof:
uv run -m scripts.validate_kimi_k3_h200 --rollouts 2 --launch
```

Capacity mode inserts a masked synthetic observation between short generations.
It tests near-64k sequence capacity, not autonomous 60k-token generation or
learning quality. Meaningful rewards and a longer smoke test are still needed.
Use `--mode math` for ordinary DAPO rollouts with the context ceiling.

Full retries use deployment-compatible launcher `1e4c5bdd2`, with matching
flags, patches, environment, topology and RAM. Current main's shared Training
Gym-to-Dojo environment migration is outside this model change.
