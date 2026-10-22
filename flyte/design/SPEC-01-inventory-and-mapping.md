# SPEC 01 — Inventory & Modal → Flyte2 primitive mapping

Status: **Round 1 (design spec). Superseded by `flyte/DESIGN.md` where they
disagree.**

## 1. What must be preserved (the user-facing contract)

`modal-dojo` exposes a small, ergonomic public surface around LLM training.
`modal_dojo/__init__.py` re-exports ~90 names, but the *user-facing training
API* we must keep byte-for-byte compatible is:

```python
from modal_dojo import TrainConfig, Qwen3_4B_Recipe, HuggingFaceDataset

config = TrainConfig(
    dataset=HuggingFaceDataset(hf_repo="open-r1/OpenR1-Math", split="train"),
    model=Qwen3_4B(),
    recipe=Qwen3_4B_Recipe(actor_num_gpus_per_node=8, gpu_type="H100"),
)
run = config.launch()   # non-blocking handle
run = config.train()    # blocking; returns TrainingRun
```

Conveniences layered on top of that core:

| Convenience | Where | Must survive the port |
|---|---|---|
| Single `train()` = download model + prep dataset + train | `common/train.py` | yes — orchestrate as Flyte tasks |
| Recipes are Pydantic dataclasses, every field → CLI flag | `train_recipes/*` | yes — config objects are backend-agnostic |
| Model presets (`Qwen3_4B`, …) | `common/models/*` | yes — config objects, no Modal deps |
| `DatasetConfig` abstraction + `write()`/`rows()` | `common/dataset.py` | yes |
| `TrainingRun` handle (id, status, url, checks) | `common/run.py` | yes |
| Checkpoint resume (`resume_from_checkpoint`) | `common/checkpoint.py` | yes |
| GPU allocation / memory guardrails | `gpu_allocation.py` | yes (map to `flyte.Resources`) |
| Eval, W&B/Trackio metrics | `common/eval.py`, `common/metrics.py` | keep — metric providers are URL/env config |
| `CustomDeployment` (vLLM/SGLang serving) | `common/deployment.py` | **phase 2** (needs Flyte service env) |

## 2. What is Modal-specific and must be replaced

| Modal primitive | Used for | Flyte 2 replacement |
|---|---|---|
| `modal.App(app_name, tags=…)` + `@app.function` | Containerized remote functions | `flyte.TaskEnvironment` + `@env.task` |
| `modal.Image` / `.run_commands()` / `.pip_install()` | Container image definition | `flyte.Image.from_debian_base().with_pip_packages()` / `from_dockerfile(...).run(...)` |
| `modal.Volume` mounted at `/root/.cache/huggingface`, `/data`, `/checkpoints` | Shared, persistent storage across stages | `flyte.io.Dir`/`File` offloaded to the cluster object store (run-scoped "volumes") |
| `app.run(detach=True)` + `app.train.spawn()` | Detached cluster launch + waitable handle | `flyte.run(task, …)` returns a `Run` handle; `run.wait()`; `flyte.deploy` for durable entities |
| `modal.enable_output()` / `modal_app_dashboard_url` | Terminal streaming + dashboard URL | Flyte UI URL (`run.url`), Flyte exec logs |
| GPU spec `f"{gpu_type}:{n}"` / `cloud=` / `region=` | GPU scheduling | `flyte.Resources(gpu="H100:8")`; devbox/Union supply placement |
| `Secrets("huggingface-secret")` | HF auth | `flyte.Secret(key=…)` / env vars (`HF_TOKEN`) |
| `ModalClient` / `resolve_caller_module` cloudpickle-by-value | Serialize inline user classes to the container | Flyte `flyte.run` pickles task args; pass config objects directly |
| Ray/nodes bring-up (`cluster.start_ray()`, `SLIME_HOST_IP`) | Multi-node NCCL/colocated weight sync | **phase 2** — Flyte 2 node-group/clustered task env is experimental; v1 devbox is single-node |

## 3. Pipeline contract (stage = Flyte task)

The Modal app registers these callables; we keep the same logical stages as
Flyte tasks so recipes, status reporting, and checkpoints stay meaningful:

1. `download_model` — snapshot the HF model into the HF-cache "volume" (`Dir`).
2. `prepare_dataset` — materialize `dataset.rows()` to JSONL in the `/data` `Dir`.
3. `convert_checkpoint` (optional, non-bridge) — HF → Megatron torch_dist.
4. `train` — run the framework launcher on GPU resources; WRITE checkpoints to
   the `/checkpoints` `Dir`.

## 4. Volume → Flyte object-store mapping (core decision)

Modal `Volume`s are block-store-backed and stay alive across runs. Flyte 2
persists task I/O in its object store (S3/GCS/R2 on Union; the devbox's MinIO).
We model each modal "volume" as a **run-scoped object-store prefix** exposed to
tasks as a mounted-directory path via `flyte.io.Dir`:

| Modal path | Flyte object key prefix | Purpose |
|---|---|---|
| `/root/.cache/huggingface` | `dojo/<repo>/hf-cache/` | read-mostly, shared model cache |
| `/data` | `dojo/<training_run_id>/data/` | materialized training JSONL |
| `/checkpoints` | `dojo/<training_run_id>/checkpoints/` | training outputs, persists across runs |

Because Flyte tasks are isolated containers, shared caches are **not** a
mount-level guarantee the way a Modal Volume was; phase 1 keeps the HF cache
per-repo (same prefix for every run of a model) and reads/writes through the
object store, which Flyte syncs into each task's container automatically.

## 5. What "works on Union / devbox" means for the port

- **Union** (`~/.flyte/config.yaml` → `demo.hosted.unionai.cloud`): a full
  managed Flyte cluster with object store + GPU nodes; `flyte.run` executes the
  pipeline remotely with image builds pushed to a registry.
- **Devbox** (`flyte start devbox` → `~/.flyte/config-devbox.yaml`): a local
  Flyte control plane (MinIO + local k8s); image builder is `local`, so tasks
  run the images the SDK ships. GPU nodes are generally absent → validate with
  the `mode="minimal"` CPU path.

Phase-1 explicitly scopes **single-node** training plus GPU *resources
requests*. Multi-node megatron/Ray, GPU serving endpoints, and dashboard
mirroring are enumerated in SPEC-03 as follow-up phases.
