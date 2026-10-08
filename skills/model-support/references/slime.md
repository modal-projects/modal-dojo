# Adding a new model config to slime

Read [../SKILL.md](../SKILL.md) first for the artifacts directory, the gotchas that apply to both frameworks, and the shared validation gates.

Always read the common gotchas.

### Phase 1: Discovery

First try looking for the existing model running on slime. You can find examples in [slime model scripts](https://github.com/THUDM/slime/tree/main/scripts/models) or [slime examples](https://github.com/THUDM/slime/tree/main/examples). If you cannot find an existing model, find the model with the most similar architecture. Reference huggingface for model architecture.

**Check image/version compatibility FIRST — it is the most common blocker.** The Modal Dojo pins the slime image by digest (`SLIME_IMAGE` in `modal_dojo/frameworks/slime/launcher.py`). A model added to slime *after* that image was built will not run on it. Verify:
- **When support landed upstream** — date the model script / plugin / bridge via the GitHub API:
  `curl -s "https://api.github.com/repos/THUDM/slime/commits?path=scripts/models/<model>.sh&per_page=5"` (also check `slime_plugins/models/...` and any `slime_plugins/mbridge/...`).
- **When the pinned image was built** — map the `SLIME_IMAGE` digest to its nightly tag/date on Docker Hub:
  `curl -s "https://hub.docker.com/v2/repositories/slimerl/slime/tags?page_size=30&ordering=last_updated"`.
- **The rollout backend's sglang version** — slime's Dockerfile pulls `slimerl/sglang:<tag>`; new attention backends (e.g. NSA/DSA) require a matching sglang bump, so check the Dockerfile at both the pinned commit and the model's support commit.
- If the model postdates the image, bumping `SLIME_IMAGE` is a **shared-infra change** (re-bases every slime recipe) → flag it for the user and smoke-test an existing small recipe (Qwen3-0.6B) after the bump. A per-recipe base image override is NOT cleanly possible: the sglang binary lives in the base image, and `image_overlay` / `image_run_commands` / `image_env` / `local_slime` only layer on top of the fixed `SLIME_IMAGE`.

Then, output your first artifact, which is a file called `model_setup.md`, containing:
- Is there this model or a model with the same architecture that is already validated on slime?
- **Does upstream support postdate the pinned `SLIME_IMAGE`?** If so, note the required image bump + sglang version.
- Is this model validated to be supported in megatron?
- Is this model validated to be supported in sglang?
- What is your plan for train configuration?
- How long do you expect each step in training take?
- How long do you expect each substep to take (e.g. rollout server initialization, weight sync, rollouts, etc)

**Scale/cost gate.** Very large models (700B+ needing dozens of nodes) make the Phase-2/3 runs expensive, cluster-reserving operations. Do not launch them without explicit user sign-off on GPU budget and cluster availability; write the recipe + config first (Phases 1 & 4) and leave the live run to the operator if unconfirmed.

### Phase 2: Implementation

Output a slime config you believe will work, then follow the one-step proof in
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

Create a doc describing the slime config changes, and justify any patches you have made. If it's possible to not patch, do not patch.


# Common gotchas

Slime by default use mbridge (`megatron_to_hf_mode=""`) instead of bridge (`megatron_to_hf_mode="bridge"`), which requires it to preconvert the weights. To determine if we should use bridge mode or mbridge, look upstream at the slime codebase at what was used for similar models.

## How the recipe maps to CLI flags (add flags without touching Modal Dojo code)

`SlimeRecipe.cli_args` emits `--<field-name-with-dashes> <value>` for **every dataclass field** not listed in `_SLIME_SKIP` (recipe.py). So the way to add an arbitrary slime/sglang flag is simply to **declare it as a field on your recipe subclass** — no edits to `recipe.py` or the launcher. `glm_4_7.py` does exactly this for its `sglang_*` and perf flags. Rules `cli_args` follows:
- `True` → bare flag (`--foo`); `False` / `None` / `""` → omitted entirely. So default an unwanted flag to `None`/`False`/`""`.
- `list` → `--foo a b c`.
- Fields in `YAML_CONFIG_FIELDS` (`eval_config`, `extra_config`, `sglang_config`) may be passed as a **dict** — `prepare_slime_config` materializes it to a YAML file at runtime and rewrites the value to the path (this is how SGLang PD-disaggregation `server_groups` are supplied). `JSON_CONFIG_FIELDS` (`train_env_vars`, `apply_chat_template_kwargs`, `multimodal_keys`) are passed as JSON.
- Things in `_SLIME_SKIP` (e.g. `slime_model_script`, `megatron_conversion_hf_checkpoint`, `environment`) are launcher instructions, not CLI flags — they won't appear in `cli_args` output. That is expected, not a bug.

## `slime_model_script` vs. `ModelArchitecture`

When a model's args aren't representable in `ModelArchitecture` (DSA, MLA q/kv-lora, `--spec ...`, `--allgather-cp`, `--enable-experimental`, exotic MoE routing), set `slime_model_script = "scripts/models/<model>.sh"`. The launcher then `source`s that script and passes `${MODEL_ARGS[@]}`, and `recipe._model_to_fields` is **skipped** — the upstream args are used verbatim. This mirrors `GLM_4_7_Recipe`. The `ModelArchitecture` on the model class becomes informational (still useful for the `num_experts ÷ expert_model_parallel_size` validator).

## Checkpoint conversion (torch_dist) constraints

`get_checkpoint_conversion_policy` (modal_helpers/utils.py) decides the HF→torch_dist conversion layout. Gotchas:
- It emits only **TP and PP** (plus `decoder-first/last-pipeline-num-layers`, `mtp-num-layers`, `make-vocab-size-divisible-by`) — **never EP/ETP**. torch_dist is reshardable, so training EP can differ from conversion; the model reloads fine. (This is why `GLM_4_7`/`GLM_5_2` convert without an explicit expert-parallel size.)
- There is a **single** `decoder_first/last_pipeline_num_layers` pair shared between conversion and training. For models with pipeline-stage-placement assertions (e.g. DSA cross-layer index sharing requires each PP stage to *start* on a computing layer), keep **conversion PP == training PP** so the same split is valid for both. Overriding only conversion PP (`conversion_tensor_model_parallel_size` / `conversion_pipeline_model_parallel_size`) will pass a first/last split that no longer sums to the layer count.
- MTP special case: if `tp==1 and pp==1 and mtp_num_layers`, conversion runs single-rank (world=1) to avoid corrupting the sharded state dict with the MTP head's duplicated embedding.

## Dual-checkpoint models (BF16 train + FP8 rollout)

Some models train in BF16 but roll out in FP8 (separate HF repos). The launcher's `download` step only calls `model.download()`, yet both `megatron_conversion_hf_checkpoint` (BF16) and `hf_checkpoint` (FP8) are later resolved with `local_files_only=True`. So **override the model's `download()` to fetch every repo the recipe references** (see `GLM_5_2.download()`), or a later step fails with a cache miss.

## MTP / EAGLE speculative decoding

Do NOT blindly copy `GLM_4_7`'s `_disable_mtp_in_config` (it zeroes `num_nextn_predict_layers`). That is a workaround for a PP>1 embedding collision in a model that does *not* use EAGLE. If the model uses its MTP layer for EAGLE spec decoding (`--sglang-speculative-algorithm EAGLE`), you must **keep** MTP — zeroing it breaks the draft model.

## Registration checklist (Phase 4)

Wiring a new `<Model>` + `<Model>_Recipe` requires edits in all of:
1. `modal_dojo/common/models/<model>.py` + export in `common/models/__init__.py` (import + `__all__`).
2. `modal_dojo/train_recipes/slime_recipe/<model>.py` + export in `slime_recipe/__init__.py` (import + `__all__`).
3. Top-level `modal_dojo/__init__.py`: add to `_EXPORTS` (lazy map) **and** `__all__`.
4. `SlimeRecipe.get_base_recipe` (recipe.py): add the `model_name → Recipe()` branch.
5. `common/models/validation.py: VALIDATION_CONFIGS`: `_ValidationConfig("<Name>", <Model>, Framework.SLIME)`.

Verify with: `uv run -m compileall`, `uv run ruff check <files>`, and a quick `python -c "from modal_dojo import <Model>, <Model>_Recipe; r=<Model>_Recipe(); print(r.gpu_allocation.summary())"` — instantiating the recipe runs the GPU-allocation and parallelism validators, catching bad TP/PP/EP/node math before any Modal run.
