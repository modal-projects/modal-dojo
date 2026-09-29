---
name: modal-dojo-overview
description: >-
  Explains modal-dojo repository architecture:
  package layout, TrainConfig, models, datasets, recipes, framework internals,
  cloudpickle caller resolution, tutorial generation, and shared tools. Use
  when modifying or explaining repository internals, not for running or
  debugging a normal training lifecycle.
when_to_use: >-
  User edits or asks about modal_dojo/ code, tutorials, framework
  configs, shared internals, or repository structure.
---

# Modal Dojo Overview

One-stop reference for agents asked to build, modify, or validate tutorials
and examples in this repo. Pairs with
[modal-infrastructure](../modal-infrastructure/SKILL.md) (raw Modal debugging) and
[example-validation](../example-validation/SKILL.md) (tiered example
validation).

## What this repo is

`modal-dojo` is a pip-installable Python package plus a catalog of
runnable training tutorials on Modal's multi-node GPU cluster product. End
users `pip install modal-dojo` once, then import framework-specific
launchers from their own scripts or notebooks.

## Top-level layout

```
modal_dojo/         <- installable package
├── common/                 <- cross-framework pure data + helpers
│   ├── dataset.py          <- dataset abstraction + built-in sources
│   ├── models/             <- ModelConfig hierarchy (see below)
│   ├── wandb.py            <- WandbConfig
│   ├── framework.py        <- resolve_caller_module, TOOLS_*
│   └── ray_cluster.py      <- ModalRayCluster helper (used by slime)
├── frameworks/             <- one subpackage per training framework
│   ├── slime/              <- slime GRPO (Ray + Megatron + SGLang)
└── tools/                  <- shared scripts mounted on every image at
                              /opt/modal-dojo/tools (see "Tools" below)

tutorials/                  <- runnable Python tutorial sources

tests/                      <- plain-script tests (uv run tests/<x>.py)
skills/                     <- packaged agent skills (you are here)
```

Edit `tutorials/*.py` or `tutorials/<name>/main.py` (sibling helpers allowed).
Each entry is both the runnable tutorial and the source for its docs page.

## Core abstractions

### `ModelConfig` / `HFModelConfiguration`

Model identity + optional architecture + optional local path + a
`download()` method. Found in `modal_dojo/common/models/`.

```python
class ModelConfig:                       # base; abstract download
    model_name: str = ""
    model_path: str | None = None
    architecture: ModelArchitecture | None = None
    def __init__(self, **kwargs): ...
    def download(self): raise NotImplementedError

class HFModelConfiguration(ModelConfig): # HF-hosted; snapshot_download
    def download(self):
        snapshot_download(repo_id=self.model_name)
```

Built-in subclasses:

| Class | HF repo | Architecture populated? | Notes |
|---|---|---|---|
| `Qwen3_4B` | `Qwen/Qwen3-4B` | yes | slime-ready |
| `GLM_4_7` | `zai-org/GLM-4.7` | yes | MoE; slime-ready |
| `Llama2_7B` | `meta-llama/Llama-2-7b-hf` | no | torchrun-based workflows |

**Rule of thumb for slime**: slime emits architecture fields as CLI flags,
so it requires `architecture` to be a populated `ModelArchitecture(...)`.
`SlimeRecipe._validate_custom_model_architecture` raises an
actionable `ValueError` if a user attaches a model with
`architecture is None`. Every other framework only needs `model_name`.

### `DatasetConfig`

In `modal_dojo/common/dataset.py`. Instantiate built-in sources such
as `HuggingFaceDataset` directly:

```python
dataset = HuggingFaceDataset(
    hf_repo="org/repo",
    hf_split="train[:100]",
    input_column="question",
    output_column="answer",
    input_format="text",
)
```

Use `input_format="text"` for plain text that should be converted into
messages, `"messages"` for an already formatted message column, and `"raw"`
only when the framework must not apply the model's chat template.

For custom data, subclass `DatasetConfig` and implement the methods
`input_key()`, `label_key()`, and `rows()`. `rows()` is also the interface for
explicit local or offline loops. The default `write(path)` serializes those
rows to JSONL; launchers create the destination directory and call `write()` to
materialize framework input on the shared data volume. Override
`apply_chat_template()` when its default of `True` is not appropriate.

`cache_key()` controls materialization reuse. Return the same stable key when
equivalent configurations can share written data; return `None` to make every
training run use a fresh path and attempt materialization independently.

### `TrainConfig` + recipe

`TrainConfig` composes `dataset`, `model`, and a recipe (`SlimeRecipe` /
`MilesRecipe`). It also accepts an optional, separate `eval_dataset` for the
framework's internal evaluation loop; this is independent from explicit
offline evaluation over `eval_dataset.rows()`. Recipes carry Modal infra +
framework CLI flags (`extra="forbid"`). Call `.train()` / `.launch()` — no
public `build_app()`.

```python
cfg = TrainConfig(
    dataset=MyDataset(...),
    model=Qwen3_4B(),
    recipe=Qwen3_4B_Recipe(gpu_type="H100", ...),
)
run = cfg.train()
```

### Caller resolution for cloudpickle

Launchers walk the call stack via
`common.framework.resolve_caller_module()` to find the true user-tutorial
module (skipping `modal_dojo.*` frames) and register that module
for cloudpickle by-value inlining -- this is how a user's inline
`DatasetConfig` / `ModelConfig` subclasses survive serialization to
the remote container.

## Framework catalog

| Framework | Uses `config.model` CLI-wise | Needs `architecture` | Best for |
|---|---|---|---|
| `slime` | hf_checkpoint CLI flag **plus** architecture flags | **yes** | GRPO |

## The `tools/` shared directory

Any cross-framework script lives at `modal_dojo/tools/`. Every
launcher mounts this directory at **`/opt/modal-dojo/tools`** on its
remote image(s) via `common.framework.mount_tools_dir`, so scripts are at a
predictable path regardless of which framework's container calls them.
Framework-agnostic `ModelConfig.download` overrides use this path.

To add a new tool: drop the script in `modal_dojo/tools/`, commit.
It's automatically mounted via `add_local_dir(TOOLS_LOCAL_PATH,
remote_path=TOOLS_REMOTE_PATH, copy=True)` on every framework image.

## Adding a new model

1. **Create the per-model module** at
   `modal_dojo/common/models/<name>.py`:

   ```python
   from .base import HFModelConfiguration, ModelArchitecture

   class MyModel(HFModelConfiguration):
       model_name = "org/repo"
       # Optional: populate only if slime users will consume this model.
       architecture = ModelArchitecture(
           num_layers=..., hidden_size=..., ...,
       )
       # Optional: override download only if the default HF
       # snapshot_download isn't enough.
   ```

2. **Export in `common/models/__init__.py`**:

   ```python
   from .my_model import MyModel
   __all__ = [..., "MyModel"]
   ```

3. **Verify** with a one-liner smoke (or an analogous snippet):

   ```python
   from modal_dojo.common.models import MyModel
   m = MyModel()
   assert m.model_name == "org/repo"
   ```

### When to override `download`

- Just HF snapshot -> inherit `HFModelConfiguration` (do nothing).
- Extra post-processing (format conversion, weight repacking, tokenizer
  tweaks) -> override `download` in the subclass. Reference
  `tools/<script>.py` via the canonical `/opt/modal-dojo/tools` path.
  Do **not** put this logic in a framework launcher -- keep model-specific
  quirks with the model spec, not the framework plumbing.

## Adding a new tutorial

1. **Pick the framework**. Use one of the catalog entries above in most cases.

2. **Create `tutorials/<name>.py`** with the next contiguous `order` value:

   ```python
   # ---
   # order: 0
   # ---
   #
   # # <Title>
   #
   # One-paragraph description of what this trains.

   from modal_dojo import TrainConfig

   run = TrainConfig(...).train()
   ```

   Markdown comment blocks become prose on the docs page. Python blocks become
   code cells.

3. **Validate the source**:

   ```bash
   uv run -m compileall tutorials/
   ```

4. **Keep training cheap by default**. Tutorials should smoke in a single
   step by default so Tier 2 validation is cheap:
   - Small `global_batch_size` / tiny dataset slice
     (e.g. `split="train[:4]"`).
   - For slime, prefer short recipes (`num_rollout=1`) so the first training step is visible.
   - Disable eval / save if they'd gate the first-step marker.

### Custom-model tutorial pattern (no built-in subclass)

If the tutorial's model isn't in the catalog, define a one-off subclass
inline:

```python
from huggingface_hub import snapshot_download
from modal_dojo import ModelArchitecture, ModelConfig


class MyTinyModel(ModelConfig):
    model_name = "HuggingFaceTB/SmolLM2-135M"
    # Slime also requires a populated model architecture.
    architecture = ModelArchitecture(...)

    def download(self):
        snapshot_download(repo_id=self.model_name)


model = MyTinyModel()
```

Better: inherit from `HFModelConfiguration` and skip the `download`
override (the one-line snapshot_download is the inherited default). See
existing slime tutorials for full examples.

## Validation

Always follow the tiered policy in
[example-validation](../example-validation/SKILL.md):

- **Tier 0 (local compile)** -- `uv run -m compileall modal_dojo/`.
- **Tier 1 (cheap drift checks):** Local instantiation smoke across the
  affected frameworks. No GPU.
- **Tier 2 (scheduled smoke)** -- one remote `modal run --detach` that
  reaches >=1 training step, then kill the detached app.
- **Tier 3 (full example validation)** -- canonical multi-node runs.
  Scheduled, not per-PR gating.

Per-change default: Tier 0 + Tier 1, plus Tier 2 for the new/modified
tutorial only. Don't expand to all tutorials on a single change.

## Gotchas

- **Python version pin**. The repo pins 3.12 (see `pyproject.toml` +
  `CLAUDE.md`). Modal's `serialized=True` functions require the remote
  image's Python to match the local one. If a framework image has Python
  3.11 (e.g. some ModelScope images), app build fails with `InvalidError`.
- **Framework image switches**. Recipes have no `image=`; use
  `image_overlay=` (slime/miles) or Miles `docker_image=` to customize.
  The launcher's `pip_install` chain reinstalls the framework fresh, so
  switching/overlaying the base is usually enough. Check whether transitive
  deps (megatron-core, pillow, tokenizers for transformers) are in the new
  image; the ModelScope image shipped many, bare CUDA/NGC images don't.
- **cloudpickle caller_module**. `TrainConfig.train()` / `.launch()` (and
  the internal `_build_app` path) delegate to the launcher, meaning
  `inspect.stack()[1]` inside `build_<f>_app` is not the tutorial.
  Launchers use `resolve_caller_module()` to walk past
  `modal_dojo.*` frames. Never use raw `inspect.stack()[1]` here.
- **Secrets for gated models and W&B**. Hugging Face auth is only needed for
  gated or rate-limited Hub access. Pass `WandbConfig` only when you want W&B.
- **Do not add framework-specific quirks to `TrainConfig`** that only matter
  for one model. Put those in the model's `download` override and
  make the tool script live in `modal_dojo/tools/`.

## Common file references

- Adding/modifying a model -> `modal_dojo/common/models/`.
- Adding/modifying a framework -> `modal_dojo/frameworks/<name>/`.
- Cross-framework scripts -> `modal_dojo/tools/`.
- Cross-framework helpers -> `modal_dojo/common/framework.py`.
- Tutorial sources live in `tutorials/*.py` or `tutorials/<name>/main.py`;
  their docs loader is `docs-next/src/lib/tutorial-docs-loader.ts`.
- Tests -> `tests/test_*.py`, run via `uv run tests/<file>.py`.
