# SPEC 02 — Round 2: architecture, modes, and the Flyte task pipeline

Status: **design spec (round 2). Implemented in `modal_dojo/flyte/`.**

## 1. Architecture

```
TrainConfig(dataset, model, recipe, backend="flyte")
    │
    ├─ backend="flyte" (default)        ├─ backend="modal" (legacy, unchanged)
    │                                   │
    └─ flyte.launch.flyte_launch()      └─ common/train.launch()  [existing]
         │
         ├─ flyte.pipeline.build_pipeline(config)
         │      → builds TaskEnvironments + registers @env.task stages
         │         (download_model, prepare_dataset, convert_checkpoint, train)
         │      → assembles the driver entrypoint task
         ├─ flyte.run(driver, …, mode, run_id, …)      # remote by default
         └─ returns FlyteTrainingRun handle (wraps flyte.Run)
```

Key invariant: **every config object is reused as-is.** `SlimeRecipe`,
`MilesRecipe`, `ModelConfig` subclasses, `DatasetConfig` subclasses are plain
Pydantic dataclasses / plain objects with no Modal import. Only the *launcher*
layer (`common/train.py` Modal path, `frameworks/*/launcher.py`,
`common/modal_lifecycle.py`) is Modal-bound. We add a Flyte launcher alongside,
and route `TrainConfig` to it via a `backend` selector. Nothing in the config
layer changes, so all model/dataset/recipe conveniences survive untouched.

## 2. Two training modes

The heavy slime/miles path needs a built GPU image (megatron, transformers,
vllm/sglang) and GPU nodes. That cannot be validated on a CPU devbox, and its
image build is a large, Modal-specific surface. We therefore ship **two modes**
under one pipeline; both exercise the *same* driver skeleton so validation is
meaningful:

| mode | stages | where it runs | validates |
|---|---|---|---|
| `minimal` | download_model → prepare_dataset → `train` (small HF `Trainer`) | devbox / Union CPU, or 1 GPU | end-to-end pipeline, io mapping, orchestration, handles |
| `slime` / `miles` | download_model → prepare_dataset → (convert_checkpoint) → `train` (framework CLI on GPU) | Union with GPU + built framework image | production training |

`mode` resolves per-call: explicit `TrainConfig.flyte_train(mode=…)`, else the
recipe's framework (`"slime"` / `"miles"`), else `"minimal"` for safety on
devbox. The stage signatures are identical, so swapping `train` is a one-line
change inside `pipeline.py`.

## 3. Task/stage definitions

All `@env.task`s live on one of two `TaskEnvironment`s (image/resource groups):

- `dojo_data_env` — CPU, small; hosts `download_model`, `prepare_dataset`,
  `convert_checkpoint` (CPU convert only; GPU convert reuses `dojo_train_env`).
- `dojo_train_env` — GPU; hosts `train`.

Stage signatures (all `async def`):

```python
@dojo_data_env.task
async def download_model(run_id: str, model: ModelConfig) -> flyte.io.Dir
@dojo_data_env.task
async def prepare_dataset(run_id: str, dataset: DatasetConfig) -> flyte.io.Dir
@dojo_train_env.task
async def train(run_id: str, model, dataset,
                recipe, data_dir: flyte.io.Dir,
                hf_cache_dir: flyte.io.Dir) -> flyte.io.Dir   # /checkpoints
```

Driver entrypoint (the thing `flyte.run` launches — our replacement for the
Modal `app.train` orchestration):

```python
@dojo_data_env.task(entrypoint=True)
async def main(run_id: str, model, dataset, recipe,
               mode: str,
               hf_cache_dir, data_dir, checkpoints_dir) -> flyte.io.Dir:
    hf = await download_model(run_id, model, hf_cache_dir)         # parallel-able
    data = await prepare_dataset(run_id, dataset, data_dir)
    ckpt = await train(run_id, model, dataset, recipe, data, hf, mode=mode)
    return ckpt
```

Stages that have no data dependency (download vs prepare_dataset) could fan out
with `asyncio.gather`; phase 1 keeps them sequential for determinism (SPEC-01:
"keep orchestration lightweight, avoid non-determinism").

## 4. Object-store "volumes"

`modal_dojo/flyte/io.py` returns deterministic object-key prefixes per run and
constructs `flyte.io.Dir(path=…)` handles:

```python
def hf_cache_prefix(model):   -> f"dojo/{model.model_name}/hf-cache"
def data_prefix(run_id):      -> f"dojo/{run_id}/data"
def checkpoints_prefix(run_id):-> f"dojo/{run_id}/checkpoints"
```

These are passed as `flyte.io.Dir` to stages, which treat them as a writable
mount path (`dir.local_path` / just read/write under it). `training_run_id` is
reused verbatim from `TrainConfig._generate_training_run_id()` so dashboard and
record linkage survive.

## 5. Resources, secrets, image

- `Resources(gpu="H100:8")`: built from `recipe.gpu_type` +
  `recipe.actor_num_gpus_per_node`, with `memory`/`cpu` from the recipe when set.
- Secrets: `flyte.Secret` / env vars for `HF_TOKEN`, W&B; devbox needs none.
- Image: `mode="minimal"` uses a small CPU image
  (`from_debian_base().with_pip_packages("transformers","datasets","torch","accelerate")`);
  GPU modes reference a framework image (built via `flyte.Image`; registry
  upload required — documented as the deployment step).

## 6. Handle & status

`FlyteTrainingRun` wraps `flyte.Run`: exposes `training_run_id`, `url`
(redirects to Flyte UI / Union console), `wait()`, `status`, and a
`training_run_id`-scoped `TrainingRun` record for API parity with the Modal
`TrainingRun`. `config.train()` blocks on `run.wait()`.

## 7. Open questions addressed here

- **Remote vs local:** `flyte run` defaults to remote; `flyte.run` likewise. We
  call `flyte.init_from_config()` (reads `~/.flyte/config.yaml` — Union — or
  `config-devbox.yaml` for the devbox) before `flyte.run`.
- **Determinism in the driver:** sequential awaits → stable action IDs.
- **`flyte.io.Dir` I/O:** stages read/write the local synced path under the
  object key; Flyte persists it back — this replaces `volume.commit()`.
