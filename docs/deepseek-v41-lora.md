# DeepSeek V4.1 Flash native LoRA

`DeepSeek_V4_1_Flash_LoRA_Recipe` trains rank-32 adapters on 32 H200s,
with TP4/EP32 training and four TP8 rollout engines. It uses the existing
32K window, batch 8 × 4, learning rate 1e-4 and alpha 32. Supply the dataset,
rollout implementation and reward through the usual `TrainConfig` interface.

## Adapter coverage and compatibility

The default targets are the complete supported layout from
[Miles #4205](https://github.com/radixark/miles/pull/4205):

- Attention: `layers.*.attn.{wq_a,wkv,wq_b,wo_a,wo_b}`.
- Shared-expert MLP: `layers.*.ffn.shared_experts.{w1,w3,w2}` (gate, up, down).

These are eight projections per layer; previously the recipe selected four
attention projections. `wo_a` uses upstream's grouped adapter so head groups
do not mix. Fully qualified selectors avoid similarly named compressor and
indexer modules. Routed experts, compressor, indexer and Engram remain frozen.
This is broader adaptation, not full-model LoRA or demonstrated better reward.

The runtime pins are Miles `6c8e2378cfe3dc5f99633e9719a80586dac86683` and
SGLang `34033fe90b90f1bb64e63337268beada57f21f84`
([SGLang #45231](https://github.com/sgl-project/sglang/pull/45231)). Both revisions
are still PR sources. Adapter checksum verification and strict serving loads
remain enabled. The Miles update brings cancellation-safe startup release and
the pinned-adapter/two-slot registration required when users enable DP attention;
this recipe retains its existing DP-attention setting.

**Existing four-target checkpoints require their original `target_modules`.**
Do not use the broader default to resume them: the new adapter tensors and their
optimizer state do not exist in those checkpoints. For a continuation, reproduce
the entire saved configuration, including the target list:

```python
recipe = DeepSeek_V4_1_Flash_LoRA_Recipe(
    target_modules="layers.*.attn.wq_a,layers.*.attn.wkv,"
    "layers.*.attn.wq_b,layers.*.attn.wo_b",
)
```

## Efficiency changes

- **Initial adapter synchronization:** release KV cache and CUDA graphs while
  retaining frozen serving weights. This reaches the same resident-memory state
  as release-all/restore-weights, without the initial full-base disk round trip.
  Later training transitions still back up and restore the base normally.
  The patch preserves Miles' new cancellation-safe release task; the old
  30-minute controller-tick override is removed.
- **Fixed one-sample microbatches:** pad each sequence to its own aligned length
  instead of the longest sequence on its worker. This reduces wasted token
  computation and routing-replay padding. Real tokens, masks and expert routes
  are unchanged. The optimization applies only to native DeepSeek LoRA with
  fixed microbatch size 1 and PP1; other configurations retain upstream behavior.

These two scoped patches derive from the Tau experiment's runtime fixes and
have no dependency on the Tau harness, prompts or data. Optimizer/RNG/sampler
continuation changes and optional frozen-base snapshot reuse are maintained
separately in [Dojo #732](https://github.com/modal-projects/modal-dojo/pull/732).
Snapshot reuse is not enabled by this recipe update.

## Validation

Local validation:

```bash
uv run pytest tests/test_deepseek_lora_coverage.py \
  tests/test_miles_recipe_hooks.py tests/test_miles_runtime_env.py tests/test_miles_patches.py
uv run -m compileall -q modal_dojo/
```

CPU validation builds the exact recipe image, including both source overlays
and all Dojo patches, then tests target resolution/export, grouped `wo_a`,
shared-expert output/gradient correctness, serving target recognition,
microbatch masks/routing replay, startup state transitions and upstream server
configuration. It allocates no GPUs and downloads no model weights:

```bash
MODAL_ENVIRONMENT=training-team uv run python scripts/check_deepseek_lora_coverage.py
```

The full 40-layer model with the broader adapter layout still needs a GPU proof
before promotion. Check adapter manifests, initial and post-update train/rollout
log-probability differences, finite gradients, output quality, memory headroom,
and dashboard step/substep timings. The earlier attention-only Tau run does not
validate this new coverage or establish a speedup for it.
