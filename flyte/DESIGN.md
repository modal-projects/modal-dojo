# modal-dojo → Flyte2: authoritative design

This file is the single source of truth for the migration. It consolidates the
three iterative design rounds:

- `flyte/design/SPEC-01-inventory-and-mapping.md` — inventory + primitive map
- `flyte/design/SPEC-02-architecture.md` — architecture, modes, task pipeline
- `flyte/design/SPEC-03-phasing.md` — phasing, scope boundaries, run instructions

## Problem

`modal-dojo` is a framework-agnostic launcher that trains LLMs on Modal multi-node
GPU clusters. We want the **same public API and conveniences** (`TrainConfig`
with `dataset`/`model`/`recipe`, `.train()`/`.launch()`, model presets, recipes,
dataset materialization, checkpointing, GPU guardrails) but a **backend that
targets a Flyte 2 cluster** — local devbox (`flyte start devbox`) and
Union (`~/.flyte/config.yaml`).

## Guiding principle

**The config objects are the public API and are backend-agnostic.** Recipes and
model/dataset presets are plain Pydantic dataclasses / plain Python with no
Modal import. Only the launcher layer is Modal-bound. So we:

1. Keep every config class byte-for-byte identical.
2. Add a Flyte launcher (`modal_dojo/flyte/`) that mirrors the stage contract
   (download → prepare → convert → train) as `@env.task`s.
3. Route `TrainConfig` to it via a `backend` selector (default **flyte**),
   retaining the Modal launcher behind `backend="modal"` until phase-2 parity.

## Modal → Flyte 2 mapping (summary)

| Modal | Flyte 2 |
|---|---|
| `modal.App` + `@app.function` | `flyte.TaskEnvironment` + `@env.task` |
| `modal.Image` | `flyte.Image.from_debian_base().with_pip_packages()` |
| `modal.Volume` at `/root/.cache/huggingface`, `/data`, `/checkpoints` | `flyte.io.Dir` object-store prefixes |
| `app.run(detach=True)` + `app.train.spawn()` | `flyte.run(task, …)` → `Run` handle |
| `f"{gpu_type}:{n}"` / `cloud=` / `region=` | `flyte.Resources(gpu="H100:8")` |
| `Secrets("huggingface-secret")` | `flyte.Secret` / `HF_TOKEN` env |
| Ray + nodes bring-up (multi-node) | **phase 2**: Flyte clustered task env |

## Pipeline (each stage = one `@env.task`)

1. `download_model` → HF cache object key
2. `prepare_dataset` → JSONL in the `/data` object key
3. `convert_checkpoint` (optional, non-bridge) → torch_dist
4. `train` → write `/checkpoints` object key

Driver entrypoint task `main(...)` calls them sequentially (deterministic
action IDs for stable replay).

## Modes

- `minimal`: small self-contained PyTorch GPT-2 loop (no weights) — **end-to-end
  runnable on devbox and Union CPU**. The train stage can also be routed onto a
  real accelerator (Apple-Silicon `mps`, or CUDA on a `--gpu` devbox / DGX /
  Union nodepool) via the driver's `accelerator` input (`cpu`/`mps`/`gpu`),
  which picks the matching module-level train environment; the trainer
  auto-detects the final device. This is the testable path.
- `slime` / `miles`: real framework CLI on GPU — production path on a GPU
  Union cluster with the framework task image pushed.

## Layout

```
modal_dojo/flyte/
  __init__.py     # public: Backend, flyte_launch, flyte_train, build_pipeline, FlyteTrainingRun
  env.py          # TaskEnvironment + image + resource builders from a TrainConfig
  fields.py       # deterministic object-key prefixes + flyte.io.Dir helpers
  pipeline.py     # @env.task stages + main driver
  run.py          # FlyteTrainingRun handle (wraps flyte.Run)
  launch.py       # init_from_config + flyte.run + handle wiring
```

See `flyte/design/SPEC-03-phasing.md` for run instructions (devbox + Union) and
the explicit phase-2 / out-of-scope boundaries.
