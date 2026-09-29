---
name: agent-driven-training
description: >-
  Owns the complete Modal Dojo lifecycle or one requested stage: configure,
  prove, smoke test, monitor, diagnose, continue, and promote.
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
traces for completed steps:

```bash
modal-dojo run trace <run-id> --out ./traces --step <steps> --yes
```

Read both the prompts and responses in the downloaded traces. Confirm that the
prompts and responses make sense in the context of the requested task before
advancing to the next stage.

## 2. Prove one step

Launch a fresh one-step run and monitor its new run ID:

```bash
modal-dojo run get <run-id> --verbose
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

- Failed or apparently hung: read
  [failure-signatures.md](references/failure-signatures.md).
- Flat, declining, saturated, or suspicious reward: read
  [debug-reward.md](references/debug-reward.md).
- Slow or unstable steps: read
  [debug-systems.md](references/debug-systems.md).

Change one setting at a time and repeat the smoke test with a fresh run ID.

## 4. Promote

Promote only when the proof and smoke runs are healthy, the reward remains
informative, trace inspection confirms that prompts and responses make sense
for the task, and the user has confirmed the final configuration and maximum
step count. Launch a fresh full run from that exact config and monitor it until completion or an early-stop decision.

A full run is not a commitment to spend its entire configured horizon.
Reassess efficacy early using both reward trajectories and sampled traces. If
reward remains flat, declines, or is otherwise uninformative, read
[debug-reward.md](references/debug-reward.md) and make an early-stop decision
from task metrics and traces rather than letting a healthy but ineffective job finish by default.

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

## References

Read only the reference matching the current decision:

- [failure-signatures.md](references/failure-signatures.md) — failed or
  apparently hung runs, cleanup, and relaunch.
- [debug-reward.md](references/debug-reward.md) — trajectory analysis, trace
  inspection, reward bugs, hacking, and saturation.
- [debug-systems.md](references/debug-systems.md) — phase timing, bottlenecks,
  tuning experiments, and final-step evaluation.
