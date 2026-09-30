# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`modal-dojo` is a pip-installable Python package that provides framework-aware launchers for distributed training on Modal's multi-node GPU clusters. The current entrypoint is `TrainConfig` + a recipe (`SlimeRecipe` / `MilesConfig`), then `.train()` / `.launch()` — the package handles image construction, cluster topology, Ray/NCCL bring-up, volume mounts, and checkpointing.

## Commands

```bash
# Setup
uv sync                              # install deps (Python 3.12 required)
uv run pre-commit install            # register local hooks

# Lint (ruff — tutorials/ is excluded via pyproject.toml)
uv run ruff check modal_dojo/
uv run ruff format --check modal_dojo/

# Type check
uv run pyright modal_dojo/    # if pyright is available

# Compile check (no GPU needed)
uv run -m compileall modal_dojo/ tutorials/

# Docs (Astro/Starlight site at docs-next/)
uv run scripts/generate_all.py --skip-build   # regen models table and docs pages
cd docs-next && npm ci && npm run dev                 # local dev server
uv run scripts/generate_all.py                 # full regen + build

# Models table (generated from the recipe registries)
uv run scripts/generate_models_table.py         # regenerate
uv run scripts/generate_models_table.py --check # CI freshness check

# Deploy
# IMPORTANT: These commands are only for development of the Modal Dojo itself.
# Consumers of the Modal Dojo should use `modal-dojo setup` instead.
# Features such as requiring proxy authentication only work with the CLI
# and will stop working if the dashboard is deployed with `modal deploy`.
uv run modal deploy docs-next/docs_next_app.py        # docs site → dojo.modal.dev
uv run modal deploy dashboards/app.py                  # observability dashboard

# Validate model configs / map a diff to affected tutorials
uv run scripts/validate_model_configs.py list                         # models + frameworks
uv run scripts/validate_model_configs.py list --names-only            # CI matrix names
uv run scripts/validate_model_configs.py check -m qwen3-4b
# miles models go through the same script; the registry picks the framework
uv run scripts/validate_model_configs.py check -m Qwen3.5-4B-Miles
git diff | uv run python -m scripts.diff_impact
```

## Architecture

### TrainConfig + recipe

`TrainConfig` composes `dataset`, `model`, and a recipe (`SlimeRecipe` or `MilesConfig`). Call `.train()` or `.launch()` — there is no public `build_app()`. Recipes carry Modal infra + framework CLI flags (`extra="forbid"`).

### Train pipeline

All frameworks expose a single `train()` entry point. Calling `train()` handles model download, dataset preparation, and training automatically — if the model isn't cached or the dataset isn't materialized, `train()` runs those steps first.

### Dataset API

Instantiate `HuggingFaceDataset(...)` directly; do not subclass it. Its `input_format` is `"text"` for plain text that the dataset should convert to messages, `"messages"` for an already formatted message column, and `"raw"` for input that must bypass the model chat template. Custom `DatasetConfig` classes implement the methods `input_key()`, `label_key()`, and `rows()`; override `apply_chat_template()` only when the default `True` is wrong. Launchers create the destination directory, then call `write(path)` to materialize framework input; the default implementation writes `rows()` as JSONL. `cache_key()` controls materialization reuse: stable equal keys share a path, while `None` gives every training run a fresh path and forces it to attempt materialization independently.

`TrainConfig.eval_dataset` is an optional, separate dataset for the framework's internal evaluation loop. Explicit offline evaluation is independent and should iterate `dataset.rows()` directly.

### Volume layout

Every framework mounts three Modal Volumes:
- `/root/.cache/huggingface` — shared HF model cache (read-mostly)
- `/data` — training data (framework-specific, per-app)
- `/checkpoints` — training outputs (per-app, persists across runs)

### Model presets

Known-model presets live under `train_recipes/`, such as `Qwen3_4B_Recipe`. `TrainConfig` trains with the recipe it receives; choose a model-specific class when tuned defaults are intended.

### Model validation

One registry, one script, one workflow, across every framework.

`common/models/validation.py` holds `VALIDATION_CONFIGS`: each entry maps a model name to its `ModelConfig` and the framework whose `get_base_recipe` trains it. The framework has to be declared — `SlimeRecipe.get_base_recipe` returns a recipe for any model it's asked about, so the recipe classes can't answer "is this model mine?".

`scripts/validate_model_configs.py` owns everything framework-agnostic (CLI, result JSON, markdown summary, PR comment, and the `check` flags). `scripts/validation_backends/<framework>.py` owns the only two things that differ: which recipe trains the model and which dataset it trains on, returned as a pair from one `build_*_validation` function. Recipes are used as `get_base_recipe` returns them, image included — the image a miles model trains on is declared once, in `MilesRecipe`, and validating a candidate image means bumping it on a branch and dispatching, not passing a flag. Adding a framework is one module here plus registry entries.

`list` and `list --names-only` print every registered model. The workflow's blank-dispatch branch and `diff_impact.py` use that same set, so a PR or weekly synmon run covers the whole registry. `diff_impact.py` scopes re-validation per framework, so a miles-only change doesn't re-run the slime set.

### Cloudpickle caller resolution

Launchers use `resolve_caller_module()` (in `common/framework.py`) to find the user's tutorial module by walking the stack past `modal_dojo.*` frames. This enables cloudpickle to serialize inline `DatasetConfig`/`ModelConfig` subclasses by value to remote containers.

### TrainingRun persistence

`TrainingRun` is written to the metadata volume (`MetadataStore.TRAINING_RUNS`, keyed by `training_run_id`). `TrainConfig.train()` and `TrainingRun.from_id` return that handle. `.model` reconstructs a `ModelConfig` pointing at the latest megatron checkpoint. A dashboard-only blob may also be written to `MetadataStore.TRAIN_RESULTS`.

### Tutorial system

Tutorials are `tutorials/*.py` or `tutorials/<name>/main.py` with sibling helpers. Each entry starts with comment frontmatter containing `order` and optional comma-separated `deps`. `docs-next/src/lib/tutorial-docs-loader.ts` discovers the corpus, validates contiguous order values, and renders Markdown comments plus Python code into docs pages.

### API reference generation

`scripts/api_reference_manifest.py` contains a curated list of public classes. `scripts/generate_api_reference.py` introspects each class (fields, types, defaults, methods) and generates Starlight markdown pages. Run via `scripts/generate_all.py`.

### Dashboard

`dashboards/app.py` is a Modal app with a Svelte frontend (built at image-build time). Training runs and evals write metadata to a shared Modal Volume (`modal-dojo-metadata`) via `modal_dojo.utils.metadata`. The ASGI endpoint serves the pre-built SPA + JSON APIs (`/api/runs`, `/api/train-results`, `/api/evals`) that read summary JSON from the volume on demand.

## Working rules

- Use `uv` for all Python operations. Never install packages at the system level.
- Tutorial sources are `tutorials/*.py` or `tutorials/<name>/main.py`. Keep `order` values contiguous from zero.
- Never hand-edit the models table in README.md. It is generated from `__all__` of each `train_recipes/*_recipe/__init__.py`; add the recipe and matching `ModelConfig` export, then rerun `scripts/generate_models_table.py`.
- Ruff excludes `tutorials/**`.
- Python 3.12 is pinned. Modal's `serialized=True` requires local ↔ remote Python version match.
- Modal Secrets `huggingface-secret` (HF_TOKEN) and `wandb-secret` (WANDB_API_KEY) are optional: HF auth is only needed for gated/rate-limited Hub access, and `wandb-secret` only when a `WandbConfig` is passed.
- Custom SGLang and vLLM deployments (`CustomDeployment.launch()`) are public by default (`unauthenticated=True`). Pass `unauthenticated=False` to require Modal proxy auth (export `MODAL_KEY` (`wk-…`) / `MODAL_SECRET` (`ws-…`) in the launching shell, or eval/`generate`/teacher calls return HTTP 401). For calls from remote workers (custom rm/reward fns) to authenticated endpoints, also forward the pair into the worker via a `modal.Secret` — the driver shell env doesn't reach them.
- Every framework's Modal app is tagged with `_modal_framework`, `_modal_job_type=training`, and metric provider/project/group for dashboard auto-discovery (see `common/__init__.py: COMMON_DOJO_TAGS`).

## Agent skills

- Before acting, inspect the descriptions in `skills/*/SKILL.md` and read every
  skill that matches the request. Do not assume the explicitly named skills
  below are the only available skills.
- For training lifecycle work, read `skills/agent-driven-training/SKILL.md`
  before acting. This includes launching, monitoring, inspecting, diagnosing,
  continuing, or promoting a Modal Dojo run.
- For raw Modal infrastructure work, read
  `skills/modal-infrastructure/SKILL.md` before acting. Use it for apps,
  containers, volumes, scheduling, image builds, caches, and endpoint
  authentication.
- For model support work, read `skills/model-support/SKILL.md` before acting. Use it when adding, debugging, validating, or productionizing new model support, especially Slime recipes and model configs.
