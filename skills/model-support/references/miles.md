# Adding a new model config to miles

Read [../SKILL.md](../SKILL.md) first for the artifacts directory, the gotchas that apply to both frameworks, and the shared validation gates.

Always read the common gotchas.

### Phase 1: Discovery

First try looking for the existing model running on miles. Upstream ships inside the pinned image at `/root/miles`: `scripts/models/<model>.sh` (the `MODEL_ARGS` a recipe sources), `scripts/run_<model>.py` (the validated cluster shape, parallelism and hyperparameters), `docs/models/<vendor>/<model>.md`, `miles_plugins/models/<model>/`, and `miles/backends/megatron_utils/megatron_to_hf/<model>.py` (the weight mapping selected by `model_name`). Probe the image to read them — `scripts/fetch_miles_patch_snapshots.py` is the working pattern. If you cannot find an existing model, find the model with the most similar architecture. Reference huggingface for model architecture.

**Check image/version compatibility FIRST — it is the most common blocker.** The gym pins the miles image per recipe (`docker_image` on `MilesRecipe`, overridden on the recipe subclass), so a bump changes only the recipe you edit. A model added to miles *after* that image was built will not run on it. Verify:
- **When support landed upstream** — the model plugin, the `megatron_to_hf` mapping, **and** the sglang inside the image all have to be new enough. A model with custom kernels needs an sglang that can serve them, not just the plugin.
- **That the tag runs on Modal** — the named tag an upstream model doc recommends may be arm64-only (GB300), which will not run on H100/H200. The dated `dev-*` nightlies are multi-arch; prefer one pushed shortly after the upstream merge.
- **That the tag still exists** — radixark prunes dated dev tags, so a pin that once worked can 404. Modal's cache still serves existing apps, so a stale pin is invisible until a cold pull in a fresh environment fails. Report one you find on a recipe you're not otherwise changing rather than bumping it silently.

Then, output your first artifact, which is a file called `model_setup.md`, containing:
- Is there this model or a model with the same architecture that is already validated on miles?
- **Does upstream support postdate the recipe's pinned `docker_image`?** If so, note the required image bump + sglang version.
- Is this model validated to be supported in megatron?
- Is this model validated to be supported in sglang?
- What is your plan for train configuration?
- How long do you expect each step in training take?
- How long do you expect each substep to take (e.g. rollout server initialization, weight sync, rollouts, etc)

**Scale/cost gate.** The models that land on miles are large by construction. Do not launch a multi-node run without explicit user sign-off on GPU budget and cluster availability; write the recipe + config first (Phases 1 & 4) and leave the live run to the operator if unconfirmed.

### Phase 2: Implementation

Output a miles config you believe will work, then follow the one-step proof in
[agent-driven-training](../../agent-driven-training/SKILL.md). Output the config
in `configs` directly and track progress in
`progress_log_[attempt_count].md`.

While tracking the progress, also make sure the timing lines up with your expectations in the `model_setup.md` artifact.

If this step does not work, go back to phase 1: what assumptions did you make in phase 1 that were incorrect and caused this? Output an artifact if it fails with `failure_analysis_[attempt_count].md`.

Record how long the step took, and how long each substep took. Make sure the model parser works and it is not generating gibberish.

### Phase 3: Validation

Follow the smoke-test loop in
[agent-driven-training](../../agent-driven-training/SKILL.md) with about 10
steps. If it fails, create a minimal reproduction and address the cause.

Record how long the step took, and how long each substep took. Make sure the model parser works and it is not generating gibberish.

### Phase 4: Productionize

Create a doc describing the miles config changes, and justify any patches you have made. If it's possible to not patch, do not patch.


# Common gotchas

`patch_files` (patch scripts applied at image build) and `local_miles` (a local checkout mounted over the image's copy, no rebuild) are the two patching levers. Gate model-specific patches on the model — the hardcoded 30s router bind timeout and the VL prompt preprocessing are both patched that way. Patching miles' own sources has a snapshot contract — `tests/testdata/miles/*.input|.output` plus `validate-miles-patch-snapshots.yml`, refreshed with `uv run modal run scripts/fetch_miles_patch_snapshots.py` — so changing `docker_image` can break that CI even with no patch change. Prove a patch applied by printing the patched source out of the image; "didn't apply" and "applied and didn't help" look identical in a training log.

`megatron_to_hf_mode` picks the checkpoint path: `"bridge"` uses megatron-bridge, `"raw"` turns on HF → torch_dist conversion, `""` disables export. Only the non-bridge path needs Megatron's torch_dist save patches.

Miles registers every sglang `ServerArgs` option under a `--sglang-` prefix, so any rollout-engine knob is reachable as an `sglang_*` field with no gym code.

## How the recipe maps to CLI flags (add flags without touching gym code)

`MilesRecipe` inherits `BaseTrainRecipe.cli_args`, which emits `--<field-name-with-dashes> <value>` for **every dataclass field** not listed in `_MILES_SKIP` (recipe.py). So the way to add an arbitrary miles/sglang flag is simply to **declare it as a field on your recipe subclass** — no edits to `recipe.py` or the launcher. The existing recipes do exactly this for their `sglang_*` and perf flags. Rules `cli_args` follows:
- `True` → bare flag (`--foo`); `False` / `None` / `""` → omitted entirely. So default an unwanted flag to `None`/`False`/`""`.
- `list` → `--foo a b c`.
- Fields in `YAML_CONFIG_FIELDS` (`eval_config`, `extra_config`, `sglang_config`) may be passed as a **dict** — `prepare_miles_config` materializes it to a YAML file at runtime and rewrites the value to the path. `JSON_CONFIG_FIELDS` (`train_env_vars`, `apply_chat_template_kwargs`, `multimodal_keys`) are passed as JSON. `extra_config` is the escape hatch: its keys become miles args and override same-named fields.
- Things in `_MILES_SKIP` (e.g. `miles_model_script`, `megatron_conversion_hf_checkpoint`, `environment`) are launcher instructions, not CLI flags — they won't appear in `cli_args` output. That is expected, not a bug.
- `${MODEL_ARGS[@]}` are emitted **before** the recipe's flags (`build_train_cmd`), so a recipe field overrides the same flag baked into the model script. That is how `custom_model_provider_path` swaps a provider without forking the script.

## `miles_model_script` vs. `ModelArchitecture`

When a model's args aren't representable in `ModelArchitecture` (custom kernels, exotic MoE routing, a custom model provider), set `miles_model_script = "scripts/models/<model>.sh"`. The launcher then `source`s that script and passes `${MODEL_ARGS[@]}`, and the upstream args are used verbatim. The model class then leaves `architecture = None`. Arch args are not re-derived for the conversion step, so the script is the single source of truth. Also set `model_name` when the megatron→HF weight mapping is selected by name rather than by config.

## Checkpoint conversion (torch_dist) constraints

`get_checkpoint_conversion_policy` (modal_helpers/utils.py) decides the HF→torch_dist conversion layout. Gotchas:
- It emits **TP and PP** always, and **EP/ETP only when `conversion_expert_model_parallel_size` / `conversion_expert_tensor_parallel_size` are set explicitly**. torch_dist is reshardable, so training parallelism can differ from conversion — converting at TP8/PP1/EP8 while training at TP4/PP8/EP4 reloads fine, which is why most models leave the conversion EP unset and convert at the implicit EP1. Set them only when the full expert set does not fit a rank at EP1 (Inkling-Small, 256 experts); `etp * ep * pp` must divide `tp * pp` or the policy raises. The `decoder-first/last-pipeline-num-layers` pair is dropped when the conversion layout is PP1. `CONVERT_KEEP_PP1=1` stops `convert_hf_to_torch_dist.py` auto-bumping PP toward the rank count and rewriting the decoder split.
- **A Modal Volume cannot absorb a large sharded write.** The writer dies in `inline_container.cc` with `unexpected pos` when `--save` points at the Volume at 42 layers (fine at 4 and 8). `convert_via_local_staging` writes to local disk and moves shards over afterwards; budget `convert_ephemeral_disk_mb` for the whole checkpoint plus the in-flight shard, and expect the Volume copy to dominate (~45 min of a ~58 min conversion at 550 GB).
- **A crashed conversion can register as a cache hit.** `.metadata` is written last, so it is what separates a finished save from a dead one; without that check partial weights feed training.
- **`no_save_optim` must be paired with `no_load_optim`**, or resuming a params-only checkpoint dies with `KeyError: 'optimizer'`. Saving params only is often mandatory at scale — params plus distributed-optimizer state runs to terabytes, and the Volume buffers all of it to container-local disk — at the cost of restarting Adam moments on resume.

## Shipped callables and hooks

Miles takes custom functions as import paths; the gym ships the callable by value and writes the resolved path, via `_HOOK_PATH_FLAGS`, `_HOOK_PATH_CONFIG_KEYS` and `_HOOK_WRAPPER_PATHS` in recipe.py. The wrappers live in `frameworks/miles/phase_reporting.py` and run phase reporting and dashboard capture before delegating to yours — so **setting a raw `--*-path` yourself replaces the wrapper and the run trains fine while reporting no substep times**, failing the Phase-2 dashboard gate. Pass the callable on the recipe field instead. Prefer `custom_reward_post_process_function` over a dotted path: a `__main__` function has no importable module name and miles' `import_module` fails inside the Ray actor. `capture_trace` + `trace_sample_limit` attach a per-sample generate/reward/tool-call timeline, useful when diagnosing gibberish.

## LoRA

Ship a full-parameter and a `*_LoRA_Recipe` variant on a shared private base, matching whichever upstream gates in CI. They differ on more than the adapter: full-param pins `use_dynamic_batch_size=False` + `micro_batch_size=1` (dynamic packing exposes a PP-p2p × EP-all-to-all NCCL race on varlen shapes) and offloads the optimizer, while LoRA packs dynamically, keeps both runtimes resident (`no_offload_train` / `no_offload_rollout`) and syncs only the adapter — ~3 s vs ~50 s per rollout.
- **Upstream's LoRA learning rate is a trap.** Its 5e-6 default reads as "not learning" because zero-initialized B factors need hundreds of rollouts to show a delta; 2e-4 is the validated value. Suspect this before the reward function.
- Set `sglang_moe_runner_backend = "triton"` for MoE LoRA. sglang's `auto` picks marlin for INT4 checkpoints, whose LoRA MoE path hits an illegal memory access capturing decode CUDA graphs at every batch size.
- Keep `sglang_max_lora_rank == lora_rank` and `sglang_max_loras_per_batch = 1`; RL serves exactly the current policy's adapter.
- `experts_shared_outer_loras=True` shares one outer factor across routed experts, with expert-specific factors following EP.
- PEFT adapter export is not supported for every architecture, so "train LoRA, ship an adapter" may not be deliverable even when training works.

## Multimodal

Three layers have to line up: the **model** (a multimodal `custom_model_provider_path` overriding the text provider from the model script — the towers load from `--hf-checkpoint` and never enter the torch_dist checkpoint, so no re-conversion), the **rollout** (`sglang_enable_multimodal`, plus `use_rollout_routing_replay` indexed over the media-expanded sequence so replay stays aligned after each `<image>` expands), and the **data** (`multimodal_keys`, which `MultimodalDataset` emits and `MilesRecipe` forwards). Override `_fields` on the recipe to select the multimodal provider automatically when the attached dataset has `multimodal_keys`.

The failure modes are silent, so check the rendered prompt before believing a bad reward: a missing `<image>` placeholder means the image never reaches the model and every sample scores unparseable, images must be materialized files rather than URLs, `apply_chat_template` must be off, and a coordinate-scale mismatch floors every sample (a model answering on a 0–1000 grid against a reward expecting 0–1 fractions scores zero on correct answers).

## Known upstream conflicts

`--use-dynamic-batch-size` conflicts with `--qkv-format bshd`. `--use-rollout-routing-replay` reads `num_experts_per_tok`, which some configs don't expose. Activation recompute can hit `save_for_backward` on tuple-returning layers. Raise `rollout_health_check_first_wait` a lot when deepgemm compiles, or engines get killed during startup.

`WandbConfig` is effectively unsupported: the launcher passes an empty run id because the driver and the `RolloutManager` both `wandb.init()` under a shared `WANDB_RUN_ID` and the second dies with "run ID … is in use". Use the dashboard for step/substep timing.

## Registration checklist (Phase 4)

Wiring a new `<Model>` + `<Model>_Recipe` (usually plus `<Model>_LoRA_Recipe`) requires edits in all of:
1. `modal_dojo/common/models/<model>.py` + export in `common/models/__init__.py` (import + `__all__`).
2. `modal_dojo/train_recipes/miles_recipe/<model>.py` + export in `miles_recipe/__init__.py` (import + `__all__`) — export every variant.
3. Top-level `modal_dojo/__init__.py`: add to `_EXPORTS` (lazy map) **and** `__all__`.
4. `MilesRecipe.get_base_recipe` (recipe.py): add the `model_name → Recipe()` branch. Without it the model gets no preset and every caller must pass a recipe explicitly.
5. `common/models/validation.py: VALIDATION_CONFIGS`: `_ValidationConfig("<Name>", <Model>, Framework.MILES)`. Step 4 is a prerequisite — `build_miles_validation` raises if `get_base_recipe` returns `None` — and the dataset it picks is DAPO-Math-17k, so a non-math recipe needs that backend widened.

Verify with: `uv run -m compileall`, `uv run ruff check <files>`, `uv run pytest tests/test_miles_recipe_hooks.py tests/test_miles_runtime_env.py tests/test_miles_patches.py`, and a quick `python -c "from modal_dojo import <Model>, <Model>_Recipe; r=<Model>_Recipe(); print(r.gpu_allocation.summary())"` — instantiating the recipe runs the GPU-allocation and parallelism validators, catching bad TP/PP/EP/node math before any Modal run.
