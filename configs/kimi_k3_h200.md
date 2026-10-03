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

- Eight H200:8 nodes; BF16 rank-32 LoRA training and native MXFP4 inference.
- Trainer TP4 x PP8 x CP2, DP1, EP8; four TP16 inference engines.
- Total context 65,536 tokens; one active request per engine.
- Training/log-probability packing budget 32,768 tokens per GPU at CP2.
- Full uniform activation recomputation in groups of three layers.
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

K3's original `situ_and_mul` formula is compiled with dynamic shapes to fuse
its FP32 intermediates. This addresses the activation allocation that failed
in the first full-model log-probability pass. Forward and backward numerical
comparisons use dtype-appropriate tolerances, rather than bitwise identity.

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

Full-model validation is running as
[relative-cottage-dcec7a81a761](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/relative-cottage-dcec7a81a761),
app `ap-0pUTy1g6CGleSfRyHyO4gg`. It uses the same recipe settings and patch output
as this branch. Optimizer updates, checkpoint save, and a subsequent rollout
with updated weights remain pending.

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

The active run was launched before the PR split from deployment-compatible
launcher `1e4c5bdd2`, with matching flags, patches, environment, topology and RAM.
It is unaffected by this Git reorganization. Current main's shared Training
Gym-to-Dojo environment migration is outside this model change.
