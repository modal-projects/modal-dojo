---
name: agent-driven-training
description: >-
  Owns the complete Modal Dojo lifecycle or one requested stage: configure,
  prove, smoke test, monitor, diagnose, continue, and promote.
  Diagnoses reward, entropy, log-probability, gradient, evaluation, runtime,
  timing, and async scaling symptoms using the RL Quick Reference Handbook.
when_to_use: >-
  User asks to train, post-train, fine-tune, or improve a model; launch a
  config; inspect run status or logs; debug failure, reward, or performance;
  continue a checkpoint; or promote a Modal Dojo run.
---

# Agent-driven training

## Scope

- If the user asks to train or improve a model, own the complete loop:
  configure, prove, smoke, diagnose, iterate, and promote. Continue without
  waiting for routine approval while a safe next step remains.
- If the user asks for one stage, perform only that stage and stop at its
  natural terminal condition. A status request does not authorize a relaunch;
  a diagnosis request does not authorize a fix.
- Use the `modal-dojo` CLI as the normal observability interface. Escalate to
  raw Modal commands only when CLI evidence cannot explain an infrastructure
  problem.

## Diagnose by the observed symptom

Start with the symptom you see and read its reference. These are eight peer
entry points, not a sequence to work through. Follow links to another symptom
only when the evidence calls for it. Each reference covers evidence, possible
interventions, and verification for that symptom.

| No. | What you see | Debugging reference |
| --- | --- | --- |
| 01 | Reward is flat or unexpectedly low | [Reward](references/debug-reward.md) |
| 02 | Entropy behaves unexpectedly, or performance collapses | [Entropy](references/debug-entropy.md) |
| 03 | Trainer–rollout log-probability differences grow | [Log-probabilities](references/debug-logprobs.md) |
| 04 | Gradients are tiny, spike or become nonfinite | [Updates](references/debug-updates.md) |
| 05 | Training reward improves but evaluation does not | [Evaluation](references/debug-evaluation.md) |
| 06 | The run hangs, crashes or runs out of memory | [Failures](references/failure-signatures.md) |
| 07 | Steps are slow or timing totals look wrong | [Timing](references/debug-systems.md) |
| 08 | Async or more GPUs do not help | [Async and scaling](references/debug-async.md) |

The references adapt the supplied RL Quick Reference Handbook and incorporate
repository-specific operational checks. Its missing image assets are replaced
with text. Published sources and recorded examples provide background, not
evidence about the current run. Data, infrastructure, and algorithms interact;
metric patterns suggest hypotheses rather than establishing causes.

### Gather evidence for the selected investigation

Identify the run, framework/code revision, resolved recipe, dataset revision,
reward implementation, and relevant checkpoint versions. Compare the first
anomalous interval with a healthy interval or comparable run if available.

```bash
uv run modal-dojo run get <run-id> --verbose
uv run modal-dojo run params <run-id>
uv run modal-dojo run logs <run-id> --tail 200
```

Use traces as described below when the question concerns generated behavior or
scoring; use worker logs, timings, or memory evidence for systems failures.
Establish metric definitions, masks, denominators, and weight versions before
interpreting trends. Distinguish sampled, completed, accepted, and trained
populations. Missing or disabled metrics are unknown, not zero. Use supplied
local artifacts when sufficient; identify the smallest additional evidence
needed when they cannot distinguish the leading explanations.

### Test and report the explanation

Connect each leading hypothesis to evidence and a check that could disprove
it. Prefer replaying saved responses, inspecting an affected batch, or comparing
fixed prefixes when these can resolve the question. Verify framework support
and actual implementation before turning a handbook method into a recipe flag.

For an authorized experiment, change one causal factor and hold the relevant
workload, budgets, resources, and measurement protocol fixed. Judge success by
task quality as well as the repaired metric. Report run/checkpoint and interval,
artifact paths and sample IDs where relevant, confirmed findings, alternatives,
and the next test with expected distinguishing outcomes. Quantify prevalence
with counts and denominators; label missing evidence and unvalidated fixes.

## 1. Configure and preflight

If the user has not already chosen the model, dataset, reward function,
topology, and final training horizon, propose the missing pieces and ask the
user to confirm them before implementation. Present the staged plan 
explicitly: the one-step proof, the smoke test, and the proposed full run 
with its model, GPU topology, important recipe settings, and maximum step 
count. Proof or smoke-test approval does not authorize the full run.

Create or adapt the config only after that decision. Before spending GPU
capacity:

- run a local compile/import check,
- exercise dataset formatting on representative rows, and
- test custom reward extraction on correct, incorrect, malformed, and
  missing-answer responses.

The predicted answer must come from the model response; it must not be read from
prompt or reference fields.

## Trace monitoring

At every proof, smoke, and full-run monitoring stage, use `run trace` to pull
traces for completed steps. For diagnosis, select baseline, transition,
anomalous, and recent steps, including successful and failed samples. Preview
the download size when choosing the interval:

```bash
uv run modal-dojo run trace <run-id> --out ./traces --step <steps> --dry-run
uv run modal-dojo run trace <run-id> --out ./traces --step <steps> --yes
```

Read both the prompts and responses in the downloaded traces. Confirm that the
prompts and responses make sense in the context of the requested task before
advancing to the next stage. Inspect tool observations, termination reasons,
and verifier inputs/outputs as needed. `--step` accepts a comma-separated list
or an end-exclusive range; omitting it selects all available rollout steps.
These exports are training rollouts; locate evaluation artifacts separately,
and do not assume every attempted or rejected trajectory was retained.

## 2. Prove one step

Launch a fresh one-step run and monitor its new run ID:

```bash
uv run modal-dojo run get <run-id> --verbose
```

Monitor `run get` periodically rather than merely waiting on the launch
process. For automated monitors, request JSON and feed stdout to `jq`; keep
`uv` warnings and other stderr separate from the JSON stream. A parser error,
empty response, or failed CLI call is not a reward or progress event—verify it
with a direct CLI call before reporting it.

Advance only when the run completes one rollout, records a nonempty reward,
and has no traceback. Startup can take tens of minutes. If it fails or appears
stuck, read [failure-signatures.md](references/failure-signatures.md).

## 3. Smoke test

Launch a new run from the same config and topology with about 10 steps.
Continue active monitoring until it completes and reward data spans the smoke
test.

If a symptom appears, use the [eight debugging paths](#diagnose-by-the-observed-symptom)
to investigate it before deciding on a change.

Change one setting at a time and repeat the smoke test with a fresh run ID.

## 4. Promote

Promote only when the proof and smoke runs are healthy, the reward remains
informative, trace inspection confirms that prompts and responses make sense
for the task, and the user has confirmed the final configuration and maximum
step count. Launch a fresh full run from that exact config and monitor it until completion or an early-stop decision.

A full run is not a commitment to spend its entire configured horizon.
Reassess efficacy early using task metrics and sampled traces. Investigate any
observed symptom through the [debugging paths](#diagnose-by-the-observed-symptom)
and make an early-stop decision when enough comparable evidence shows no useful
progress or sustained deterioration. Do not stop on a single noisy point or an
algorithm-expected plateau, and do not let an ineffective run finish simply
because it has not crashed. Keep the target task metric fixed.

Keep checking every active run until it reaches a terminal state or a deliberate
stop decision. Record the launch time, last progress time, current phase, and
observed step duration. If startup or a step takes materially longer than the
run's prior timings or the expected window, inspect `run get --verbose` and
`run logs` immediately. Diagnose and fix an authorized bottleneck instead of
continuing to wait without a new progress signal.

Report the run ID, checkpoint, early-versus-late reward, task-specific success
rate, timing, and whether all apps stopped.

Continue from a checkpoint only when the run is healthy and preserving its
optimizer and scheduler state avoids discarding useful progress, such as after
an interruption or while reward is still improving at the configured horizon.
For Slime, keep the original model path and set `recipe.load` to the training
checkpoint; when extending the saved scheduler horizon, also set
`extra_config={"override_opt_param_scheduler": True}`. Launch with a fresh run
ID and prove one step before proceeding. Prefer a new run when changing the
objective or when reward is saturated, corrupted, or based on a broken reward
function.

## Stop or relaunch

When stopping or fixing the run is within the authorized scope, obtain the
Modal app ID from `run get`, preserve the evidence, then use:

```bash
uv run modal app stop <app-id>
uv run modal app list --json
```

Confirm the old app stopped before relaunching against shared volumes. Change
one setting at a time, use a fresh run ID, and repeat the appropriate proof or
smoke test before promoting. For diagnosis-only or status-only requests, report
the findings without stopping or relaunching. Use
[modal-infrastructure](../modal-infrastructure/SKILL.md) for raw infrastructure
investigation when the supported CLI cannot explain the failure.
