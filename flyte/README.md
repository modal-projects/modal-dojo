# modal-dojo Flyte 2 backend

`modal-dojo` keeps its user-facing training API (`TrainConfig` with
`dataset`/`model`/`recipe`, `.train()` / `.launch()`, model presets, recipes,
dataset materialization, GPU guardrails) while running the training pipeline on
a **Flyte 2** cluster instead of Modal — the local devbox
(`flyte start devbox`) or **Union** (`~/.flyte/config.yaml`).

This package lives in `modal_dojo/flyte/`. The design is captured in
[`flyte/DESIGN.md`](DESIGN.md) and the three iterative specs in
[`flyte/design/`](design/).

## What's preserved

Every config object is reused as-is — recipes (`SlimeRecipe`/`MilesRecipe`),
model presets (`Qwen3_4B`, …), datasets (`HuggingFaceDataset`, …) are plain
dataclasses with the same API. Only the orchestration backend changed.

```python
from modal_dojo import TrainConfig, Qwen3_4B_Recipe, HuggingFaceDataset

config = TrainConfig(
    dataset=HuggingFaceDataset(hf_repo="open-r1/OpenR1-Math", split="train"),
    model=Qwen3_4B(),
    recipe=Qwen3_4B_Recipe(actor_num_gpus_per_node=8, gpu_type="H100"),
)
run = config.launch()   # or config.train() to block
```

`TrainConfig` gained a `backend` field (`"flyte"` default, `"modal"` legacy) and
`MODAL_DOJO_BACKEND` env override. Explicit helpers are also exported:
`flyte_launch(config)`, `flyte_train(config)`, `Backend`, `TrainingMode`,
`FlyteTrainingRun`.

## Pipeline

Each stage is a Flyte 2 `@env.task` on a single module-level `TaskEnvironment`
(`dojo_data_env`), exposing the same stage contract as the Modal apps:

1. `download_model` → Hugging Face snapshot (HF-cache `Dir`)
2. `prepare_dataset` → materialized JSONL (`/data` `Dir`)
3. `convert_checkpoint` → HF→torch_dist (phase 2 wiring)
4. `train` → writes checkpoints (`/checkpoints` `Dir`)

The driver is the module-level `main` task (`modal_dojo.flyte.pipeline.main`),
which `flyte.run` launches. Modal `Volume`s are replaced by `flyte.io.Dir`
object-store persistence (deterministic key layout in `fields.py`).

### Modes

- `minimal` — a tiny, network-free GPT-2 training loop on CPU/1 GPU. This is the
  **end-to-end-testable** path (works on a devbox and Union CPU nodes).
- `slime` / `miles` — real framework launcher on GPU (phase 2: needs a GPU
  `TaskEnvironment` + framework image deployed; see Scope).

## Run on the local devbox

```bash
flyte start devbox                                   # ~/.flyte/config-devbox.yaml
uv add flyte@">=2.6.0"
MODAL_DOJO_FLYTE_CONFIG=~/.flyte/config-devbox.yaml \
  python flyte/validate_minimal.py
```

## Run on Union

Your Union config is `~/.flyte/config.yaml` (the default). Verify connectivity
with `flyte get project`, then:

```bash
python flyte/validate_minimal.py                     # minimal (CPU) pipeline
# or in your own script:
from modal_dojo import TrainConfig, flyte_launch    # flyte_launch(cfg, mode="minimal")
```

`flyte run` builds task images through Union's remote builder on first use, then
caches them; subsequent runs skip the build.

## Validation

`flyte/validate_minimal.py` runs the full pipeline (download → prepare → train →
checkpoint `Dir`) with bundled smoke configs (`modal_dojo.flyte.smoke`), no
large downloads. Verified end to end on Union
(see git history / the run URL it prints).

Unit tests: `tests/test_flyte_backend.py` (no cluster required).

## Scope & phase 2

Phase 1 (this change set) delivers the single-node CPU + object-store path and
the seam for the heavy path. Phase 2 (documented, not yet implemented):

- **Multi-node / GPU framework training** — a deployed framework module with a
  GPU `TaskEnvironment` (+ framework image) that replaces the Modal Ray/NCCL
  bring-up for `actor_num_nodes > 1` and `slime`/`miles` mode.
- **`CustomDeployment` serving** (vLLM/SGLang) as `flyte.serve` environments.
- **Dashboard/metric mirror parity** in the Flyte/Union console.

Permanently out of scope: Modal's sandbox scoring (`Sandbox`, `score_in_sandbox`)
stays on the Modal runtime as a launcher utility, invoked from any task via the
same HTTP client.
