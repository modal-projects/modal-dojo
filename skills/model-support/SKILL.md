---
name: model-support
description: Use when adding, debugging, validating, or productionizing support for a new base model or
  model-specific recipe in modal-dojo, especially Slime and Miles recipes and model configs.
---

## Adding a new model config

When asked to add a new model example to SlimeRecipe or MilesRecipe, you should output artifacts in a temporary directory in `.gym/new_models/[model_name]/` folder. Once you are finished, add the finished config to the recipes folder.

Read the reference for the framework whose recipe you are adding — each carries its own four phases and common gotchas:

| Framework | Reference |
| --- | --- |
| Slime (`SlimeRecipe`) | [references/slime.md](references/slime.md) |
| Miles (`MilesRecipe`) | [references/miles.md](references/miles.md) |

### Gates for every validation run

Phase 2's single step and Phase 3's smoke test both have to show:
1. The model output is not gibberish and actually makes sense.
2. Step time **and substep times** on the dashboard — patches that break the Modal Dojo's observability fail this gate even when training works.
3. A non-zero raw reward, and by Phase 3 an increasing one. GSM8K saturates for strong bases, so pick a harder task rather than reading a flat curve as breakage.

If reward doesn't climb, isolate which layer is at fault: data (`DatasetConfig` — check rendered prompts read correctly and labels match source rows), the run setup (reward function shape, too few steps / too small batches / too low lr), the recipe (sampling, stop tokens, response budget, masking, optimizer, parallelism), or framework plumbing (weight sync, stale weights, checkpoint conversion).

### Common gotchas (both frameworks)

Naming convention: For the model name in artifacts, it should be `_` separated by model family identifiers and replacing `.` for versioning (e.g. `Qwen3_4b`, `Qwen3_6_35b`).

`--max-tokens-per-gpu` is a flag for training, whereas `--rollout-max-response-len` is a flag for rollouts.

If it is a large MoE, you may need `--optimizer-cpu-offload`, `--use-precision-aware-optimizer`, and `--overlap-cpu-optimizer-d2h-h2d`

### Validate Model Configs

`.github/workflows/validate-models.yml` guards against performance regressions and is manually triggered — required before merging a new model, so ping someone who can dispatch it (escalate to Joy Liu). Register the model in `common/models/validation.py: VALIDATION_CONFIGS` with its `Framework`. The recipe is used as `get_base_recipe` returns it, image included, so validating a candidate image means bumping it on a branch and dispatching.
