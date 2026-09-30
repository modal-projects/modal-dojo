# Dashboard

Self-hosted observability dashboard for Modal Dojo. Aggregates training
runs and eval results from the `training-gym-metadata` Modal Volume into
a single Svelte SPA served by a Modal ASGI endpoint.

Deploy your own copy:

```bash
modal deploy dashboards/app.py
```

Modal prints the URL where the dashboard is served.

## Custom trajectory viewer

The rollout detail page includes a default `TrajectoryViewer.svelte` that
renders the conversation (messages, thinking, tool calls, and eval report).
It is environment-agnostic; replace it for a deployment with your own Svelte
component:

```bash
modal-dojo setup --trajectory-viewer ./MyTrajectoryViewer.svelte
```

The component is mounted over
`dashboards/frontend/src/components/TrajectoryViewer.svelte` before Vite
builds the dashboard. It receives these props:

- `sample`: the selected normalized sample.
- `samples`: all samples in the selected rollout/prompt group.
- `trajectory`: `sample.metadata.trajectory_messages`.
- `rewardEvents`: `sample.reward_events` when emitted.
- `rollout`: the expanded `TrainingRolloutResult`.
- `run`: the current `TrainingRun` summary.

The override path is saved in `~/.modal-dojo.toml`, so later `setup` or
password redeploys keep using it. To restore the built-in viewer, run
`modal-dojo setup --no-trajectory-viewer`.

## Run-scoped dashboard components

For a component that belongs to one run, attach it after launch instead of
changing the global dashboard setup:

```python
from modal_dojo import DashboardComponent, TrainingRun

run = TrainingRun.from_id("bristled-pine-a7c3e91d4b")
run.add_dashboard_component(
    name="my-viewer",
    component_type=DashboardComponent.TRAJECTORY_VIEWER,
    from_path="./MyTrajectoryViewer.svelte",
)
```

The source is stored content-addressably in the
`training-gym-dashboard-overlay` Modal Volume, with a per-run association
record under `runs/<training_run_id>/<name>.json`. The immutable artifact
manifest is also associated with the run under
`metadata.dashboard_components`. The dashboard mounts this Volume, verifies
the source against the manifest's `sha256`, and compiles the selected Svelte
component on demand in a separate Modal function that has no secrets or
Volumes. The compiled component is served as a self-contained page and
rendered in a sandboxed `<iframe>` (opaque origin, no network access), so it
receives its props over `postMessage` and cannot use the dashboard's
credentials. The built-in viewer remains the fallback when no run-scoped
component is available (or compilation fails). If several names are attached
for the same component type, the most recently attached one is used.
