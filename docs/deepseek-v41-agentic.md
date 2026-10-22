# DeepSeek V4.1 Flash LoRA for agent rollouts

Use `DeepSeek_V4_1_Flash_LoRA_Agentic_Recipe` for multi-turn workloads that
provide their own dataset, rollout function and reward. It extends the native
DeepSeek LoRA recipe; the existing math recipe remains the default model preset.

```python
from modal_dojo import DeepSeek_V4_1_Flash_LoRA_Agentic_Recipe

recipe = DeepSeek_V4_1_Flash_LoRA_Agentic_Recipe(
    num_rollout=10,
    eval_interval=5,
    # Add your workload's rollout_function and evaluation hook here.
)
```

The variant reserves 30,720 prompt tokens and 2,048 response tokens per call.
It retains 32 H200s, training TP4/EP32, four TP8 serving engines, rank-32
attention LoRA, learning rate 1e-4, and eight tasks with four draws per update.
Manual minimum-load routing retains conversation affinity when the rollout
adapter supplies stable session IDs. It enables trace capture, optimizer/RNG
checkpoint saves and sampled policy-update diagnostics.

Dojo owns the native runtime fixes: per-microbatch padding, zero normalization
weight for loss-masked scheduling fragments, the initial adapter sync without
a frozen-weight disk round trip, native optimizer/RNG/scheduler and sampler
restore, and effective LoRA/policy-update diagnostics. Patches fail on unexpected
upstream source and are tested against the pinned Miles revision. A workload
that pads fragment batches must reject batches with no supervised tokens.
The harness still owns conversation fragmentation, padding sample construction,
model input/output parsing, tool execution, compaction, task data and rewards.

New checkpoints store RNG/topology under `dojo_resume_state`; the earlier Tau
`tau_resume_state` format is accepted for continuation. Exact RNG restoration
requires the saved topology. Set `lora_adapter_path` to the native adapter
directory while retaining the frozen base checkpoint in `load`.

## Optional frozen-weight snapshot reuse

`reuse_frozen_base_backup=True` builds a pinned allocator extension and enables
reuse of the initial frozen base snapshot on later offloads. Base allocations
are read-only after sealing; adapter allocations remain writable and are saved
again. This option defaults to **false**. CPU lifecycle and isolated CUDA tests
cover exact restoration, adapter changes, graph replay and write protection;
they do not establish full-model serving compatibility or end-to-end speedup.

## Validation

Local tests require no GPUs:

```bash
uv run --frozen pytest tests/test_deepseek_agentic_recipe.py tests/test_native_lora_runtime.py tests/test_frozen_weight_backup.py
```

Run native framework and allocator lifecycle tests in the actual image on CPU:

```bash
MODAL_ENVIRONMENT=training-team uv run --frozen python scripts/check_deepseek_agentic.py --reuse-frozen-base-backup
```

The optional `scripts/check_frozen_weight_backup.py` checks CUDA protection and
restoration on one H200 with a ten-minute function timeout. It uses no model or
shared checkpoint volumes. Run it only within an approved GPU budget.

The original harness implementation completed a ten-update Tau smoke with
evaluation reward 0.361 → 0.504 → 0.509 at updates 0/5/10, finite nonzero
gradients and stable trainer/serving log-probability differences. Snapshot reuse
was disabled. This is evidence for the source behavior before extraction, not
a separate GPU validation of this relocated recipe. The extraction is checked
for recipe parity and exercised in the native CPU image before publication.
