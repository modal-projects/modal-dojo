# SPEC 03 — Round 3: phasing, scope boundaries, and run instructions

Status: **design spec (round 3).**

## 1. Phasing

### Phase 1 (this change set)
- Add `flyte>=2.6` to `pyproject.toml` dependencies.
- New `modal_dojo/flyte/` backend: environments, io mapping, pipeline
  (`download_model` / `prepare_dataset` / `convert_checkpoint` / `train` +
  `main` driver), Flyte launch + `FlyteTrainingRun` handle.
- Two training modes (`minimal` for devbox/CPU end-to-end verification, and
  `slime`/`miles` GPU production mode wired to the framework CLI).
- `TrainConfig` gains `backend: Backend = "flyte"`; `train()` / `launch()`
  dispatch to the Flyte backend. Legacy Modal path remains importable and
  selectable (`backend="modal"`).
- Unit tests for the backend (resource mapping, io prefixes, arg building,
  dispatch) that run without a cluster.

### Phase 2 (documented, not yet implemented)
- Flyte-native multi-node training (node groups / clustered task env) to
  replace Modal's Ray + NCCL weight-sync bring-up for `actor_num_nodes > 1`.
- `CustomDeployment` (vLLM/SGLang) as `flyte.serve` environments —
  `FastAPIAppEnvironment`-style service task; auth via Flyte secrets.
- Dashboard/metric mirror parity ("Modal app URL" → Flyte/Union console URL).

### Out of scope
- Sandbox scoring (`modal*Sandbox`) stays on the Modal runtime until a Flyte
  sandbox story exists; it is a launcher utility, not part of the `TrainConfig`
  contract, and can be called from any task with the same HTTP client.

## 2. Scope boundaries & honest limits

The Modal codebase is ~228 modules, deeply coupled to `modal.Volume`,
`modal.Image`, Ray cluster bring-up, ServeFunction deployments, and a Svelte
dashboard. A faithful 1:1 rewrite of every stage is a multi-week effort. This
change set delivers a **correct, runnable Flyte2 backend for the core training
contract** (`TrainConfig(dataset, model, recipe).train()/.launch()`) with the
same stage skeleton and object-store-backed persistence, and it is the seam on
which the phase-2 items (multi-node, serving) are implemented. The pre-existing
Modal launcher is retained behind `backend="modal"` rather than deleted, so the
repo stays internally consistent and existing tests keep their subject.

## 3. How to run on the local devbox

```bash
# 1. Start the devbox (local Flyte control plane + MinIO)
flyte start devbox            # config → ~/.flyte/config-devbox.yaml

# 2. Install the SDK + this package
uv sync                        # pulls flyte + modal-dojo deps
uv add flyte@">=2.6"

# 3. Run the minimal pipeline end-to-end on the devbox (CPU)
MODAL_DOJO_FLYTE_CONFIG=~/.flyte/config-devbox.yaml \
  python - <<'PY'
from modal_dojo import TrainConfig, Qwen3_4B_Recipe, HuggingFaceDataset
from modal_dojo.common.models import Qwen3_4B
cfg = TrainConfig(
    dataset=HuggingFaceDataset(hf_repo="databricks/databricks-dolly-15k", split="train"),
    model=Qwen3_4B(),
    recipe=Qwen3_4B_Recipe(actor_num_gpus_per_node=0),  # cpu-selected by mode
)
run = cfg.flyte_launch(mode="minimal")
print(run.url)
run.wait()
PY
```

## 4. How to run on the Union cluster

```bash
# Union config is ~/.flyte/config.yaml (dns:///demo.hosted.unionai.cloud)
flyte get project           # verify connectivity / auth

# GPU mode; the framework task image must be pushed (see flyte/README.md)
python - <<'PY'
from modal_dojo import TrainConfig, Qwen3_4B_Recipe, HuggingFaceDataset
from modal_dojo.common.models import Qwen3_4B
cfg = TrainConfig(
    dataset=HuggingFaceDataset(hf_repo="open-r1/OpenR1-Math", split="train"),
    model=Qwen3_4B(),
    recipe=Qwen3_4B_Recipe(actor_num_gpus_per_node=1, gpu_type="H100"),
)
run = cfg.flyte_launch(mode="slime")
print(run.url)
run.wait()
PY
```

## 5. Design decisions — final

1. **Config layer is the API; launcher is a pluggable backend.** Model presets,
   recipes, and datasets are untouched; only orchestration moves.
2. **Object store replaces Modal Volumes** via `flyte.io.Dir` prefixes.
3. **Deterministic sequential driver** (no time-based branching) so Flyte
   replay/retries are stable.
4. **`mode` selects weight of `train`** so the pipeline is end-to-end verifiable
   on a CPU devbox and production-correct on a GPU Union cluster.
5. **Modal backend retained** behind a selector until phase-2 parity is done.
