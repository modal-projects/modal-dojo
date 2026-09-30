---
name: modal-infrastructure
description: >-
  Operates raw Modal infrastructure: runs, apps, containers, volumes,
  scheduling, image builds, caches, and endpoint authentication.
when_to_use: >-
  Use for explicit raw Modal operations or when the Modal Dojo CLI cannot
  explain an infrastructure failure. Use agent-driven-training for normal
  Modal Dojo lifecycle work.
---

# Modal infrastructure operations

This document captures durable repo-specific workflow for agents launching and debugging training jobs on Modal in this repository.

For routine Modal Dojo lifecycle work, use
[agent-driven-training](../agent-driven-training/SKILL.md). Use this runbook
when the request explicitly concerns Modal infrastructure or the Modal Dojo
CLI cannot explain the underlying failure.

## Scope

- Keep secrets out of this file.
- Record reusable workflow and runtime behavior, not one-off experiment notes.
- Prefer repo-root commands so the runbook works from a clean checkout.
- Keep the guidance generic enough that an agent can discover the right entrypoint from the repo rather than relying on a fixed list.

## Environment Setup

- Make sure the `modal` CLI is available and authenticated in the shell you are using.
- Set `MODAL_ENVIRONMENT` explicitly, or pass `--env <env>`, when the target environment matters.
- If your local setup relies on shell init for auth or helper tooling, ensure that setup is loaded before launching jobs.
- Store credentials in local shell config or Modal secrets, not in tracked files.
- **Proxy-auth tokens for served endpoints.** SGLang and vLLM endpoints from `CustomDeployment.launch()` are public by default (`unauthenticated=True`). Pass `unauthenticated=False` to require Modal proxy auth — then *any* call to that endpoint — `deployment.generate()`, an OPD teacher `/generate`, health/readiness polls — must send a proxy-auth token pair or it returns **HTTP 401**. Create a token in the Modal dashboard (Settings → Proxy Auth Tokens) and export it in the launching shell as `MODAL_KEY` (`wk-…`) and `MODAL_SECRET` (`ws-…`); the package turns these into `Modal-Key`/`Modal-Secret` headers. For calls issued from **remote workers** (e.g. a custom rm/reward function hitting a teacher endpoint), the driver's shell env does not reach the worker — forward the pair into the worker by attaching a `modal.Secret` (e.g. via the recipe's `train_function_kwargs={"secrets": [...]}`).

## Finding The Right Entrypoint

Discover the launch path from the repo itself instead of assuming a fixed command.

- Start with the workflow README and identify the documented `modal run` path.
- Inspect the corresponding launcher module to see whether it exposes intermediate stages such as dataset prep, model download, checkpoint conversion, benchmark, evaluation, or training.
- Check for cheaper discovery paths first, such as `--help`, `--list`, `--dry-run`, or a single-node variant.
- If the launcher uses a local entrypoint, make sure the documented invocation still maps to that entrypoint from the repo root.
- If the workflow requires secrets, volumes, or cached artifacts, treat those prerequisites as explicit stages rather than assuming training is the first runnable step.

## Launching Jobs

- Prefer detached launches for real training and any helper job that may run for a long time:

```bash
modal run --detach path/to/modal_train.py::function
```

- Use the smallest dataset subset that still exercises the full stack when proving a new path.
- Once one topology works, keep the topology fixed and change one tuning parameter at a time.
- Treat model downloads, dataset prep, conversion, and training as separate checkpoints when the example supports that split.
- Prefer proving the workflow stage-by-stage rather than jumping straight to the longest training command.

## Scheduling Expectations

- Large GPU jobs may wait several minutes before worker assignment.
- During capacity waits, the client may only show spinner output or a scheduling message.
- Empty logs do not mean failure if workers have not started yet.
- A detached app may briefly gain tasks and then drop back to zero while Modal retries placement.

## Detached App Workflow

- After the function invocation exists, `Ctrl-C` is safe for detached runs and the app keeps running on Modal.
- In attached mode, `Ctrl-C` stops the app even if containers are already active.
- `--detach` does not protect the run before the function invocation exists. Interrupting during image build or object creation can still kill the launch.
- Track long-running jobs by app ID.

Useful commands:

```bash
modal app list --json
modal app logs <app-id>
modal app stop <app-id>
```

Do not stop an app during a status or diagnosis-only request.

## Reading App State

- `modal app list --json` is the fastest way to confirm that an app exists and whether workers are active.
- The JSON output uses display-style keys such as `App ID`, `State`, `Tasks`, and `Stopped at`.
- `Tasks: 0` usually means workers have not landed yet or the placement is being retried.
- `Tasks: N` means containers are active and logs should start to matter.
- `State: stopped` with a recent stop time is the quickest confirmation that a detached run has exited.

## Logs, Containers, And Volumes

- `modal app logs <app-id>` only shows function and container logs after user code starts. It does not show the earlier image-build phase.
- For multi-node jobs, expect repeated startup blocks, one per node.
- In general, the examples in this repo build on training frameworks with notoriously noisy logs. It's often best to be proactively defensive with context by filtering the logs, either with `grep`/`rg`, or with the extra parameters available to `modal app logs`:

```bash
# Fetch the last 500 log entries and grep them
modal app logs <app-id> --tail 500 | rg "error|CUDA|OOM|loss"

# Fetch only stderr from a particular function call
modal app logs <app-id> --source stderr --function-call <function-call-id>

# Search for specific keywords in logs
modal app logs <app-id> --search "error"
modal app logs <app-id> --search "loss"

# Show container IDs alongside logs to distinguish output from different nodes in multi-node jobs
modal app logs <app-id> -f --show-container-id

# Filter logs to a single container, or function call, or function
modal app logs <app-id> --container <container-id>

# Fetch logs from a specific time window
modal app logs <app-id> --since 2h
modal app logs <app-id> --since 30m --until 10m
```

- When logs lag behind actual progress, inspect live containers directly:

```bash
modal container list --env <env>
modal container exec <container-id> -- nvidia-smi
```

- Use volume inspection when you need to confirm whether a job has started writing committed outputs:

```bash
modal volume ls <volume-name> / --env <env>
```

- `modal volume ls` takes the path inside the volume, not the in-container mount path.
- Live files inside a running container are not the same as committed volume state. If a producer app is still running, container inspection is often more accurate than volume listings.

## Image Build And Startup Behavior

- A lightweight helper function can still trigger image creation for heavier functions defined in the same app.
- When helper jobs feel stuck before user code runs, check whether the delay is image build time rather than training logic.
- For very recent upstream dependencies, pin exact commits or versions in the Modal image definition instead of relying on floating branches.

## Large Downloads And Shared Caches

- Model downloads can take much longer than the first visible log lines suggest.
- Progress bars may stall or update slowly even while shards are actively downloading.
- If a replacement app will reuse the same shared cache volume, stop the previous app first and confirm it has fully stopped before launching the next one.

## Debugging Strategy

1. Prove the launch path with the smallest useful dataset or shortest run.
2. Derive the exact command from the README and launcher instead of guessing.
3. Wait through scheduling before assuming failure.
4. Confirm app creation and active task count with `modal app list --json`.
5. Inspect logs only after workers appear.
6. If logs are ambiguous, check live containers and committed volumes separately.
7. After a known-good run, change one variable at a time.

## SGLang Qwen3.5 served endpoint tool-call parser

When serving Qwen3.5 models (`Qwen3_5_9B`, `Qwen3_5_4B`, etc.) behind an SGLang endpoint that will be used for tool calling, set the parser to `qwen3_coder` and add the reasoning parser:

```python
SglangRecipe(
    extra_server_args={
        "--tool-call-parser": "qwen3_coder",
        "--reasoning-parser": "qwen3",
    },
)
```

Do **not** use `--tool-call-parser qwen` or `qwen25` for Qwen3.5. Those parsers expect a JSON object inside `<tool_call>` tags, while Qwen3.5 emits XML-style calls like:

```xml
<tool_call>
<function=list_directory>
<parameter=path>/repo</parameter>
</function>
</tool_call>
```

With `qwen`/`qwen25`, SGLang logs `Failed to parse JSON part: ...` and the OpenAI response contains an empty `tool_calls` list, causing agent loops to hit `Reached max iterations without a final response.`. The `qwen3_coder` parser (and the `qwen3` reasoning parser for any inline thinking) handles the XML format correctly on the default `lmsysorg/sglang:v0.5.12` image.

(Qwen3.6-35B is a separate case: pass `--tool-call-parser qwen` on `SglangRecipe`, not `qwen3_coder`.)

## Updating This Runbook

- Add new notes when they describe durable Modal behavior or a stable repo workflow.
- Do not preserve temporary experiment history here once it has served its purpose.
- If a lesson only applies to one example, prefer that example's README over this shared runbook.
- Do not turn this file into a catalog of current example entrypoints; keep it focused on how to discover and debug them.
