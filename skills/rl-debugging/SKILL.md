---
name: rl-debugging
description: >-
  Diagnoses RL training problems using metrics, datasets, generated responses,
  verifier decisions, and controlled comparisons. Use for flat or misleading
  reward, training/evaluation gaps, entropy collapse, log-probability mismatch,
  unstable updates, or RL throughput and async scaling problems. Complements
  agent-driven-training for run lifecycle and modal-infrastructure for raw
  infrastructure operations.
---

# RL debugging

## Scope

Use the RL Quick Reference Handbook to turn a symptom into a supported
diagnosis and a testable next action. Data, infrastructure, and algorithms
interact; check actual attempts and scoring before assuming an algorithmic
failure. Metric patterns suggest hypotheses, not universal diagnoses.

For Modal Dojo run operations, also read
[agent-driven-training](../agent-driven-training/SKILL.md). That skill owns
launching, monitoring, stopping, continuing, and promoting runs. Keep work
within the requested stage and existing authorization; diagnosis alone does
not authorize another training run. Use
[modal-infrastructure](../modal-infrastructure/SKILL.md) when CLI evidence
cannot explain an infrastructure failure.

## Collect evidence

1. Identify the run, framework and code revision, resolved recipe, model and
   checkpoint versions, dataset revisions/splits, reward implementation, and
   intended task metric. Record the symptom's first occurrence and a healthy
   comparison interval or run, if available.
2. Inspect configuration, metric history, and logs through the existing CLI:

   ```bash
   uv run modal-dojo run get <run-id> --verbose
   uv run modal-dojo run params <run-id>
   uv run modal-dojo run logs <run-id> --tail 200
   ```

3. Inspect prompts, responses, tool observations, termination reasons, and
   verifier inputs/outputs from baseline, transition, anomalous, and recent
   rollouts. Include both successful and failed samples. Preview the download
   size, then fetch the relevant steps:

   ```bash
   uv run modal-dojo run trace <run-id> --out ./traces --step <steps> --dry-run
   uv run modal-dojo run trace <run-id> --out ./traces --step <steps> --yes
   ```

   `--step` accepts a comma-separated list or an end-exclusive range; omitting
   it selects all available rollout steps. These exports are training rollouts;
   locate evaluation artifacts separately. Do not assume they contain eval
   responses or every attempted/rejected trajectory.
4. Establish metric definitions, denominators, masks, and weight versions.
   Distinguish sampled, completed, accepted, and trained populations. Missing
   metrics or responses are unknown, not zero. State coverage and any evidence
   that the run did not retain.

Use local artifacts when supplied; live access is unnecessary if those
artifacts answer the question. If critical evidence is missing, identify the
smallest additional export or instrumentation needed to distinguish causes.

## Select the investigation

Read only the relevant sections of
[the handbook](references/handbook.md), following its numbered cross-references
when the evidence points to another symptom.

| Symptom | Investigation |
| --- | --- |
| Flat or low reward | Handbook §01: generation, verifier, within-group reward contrast, filtering |
| Unexpected entropy or quality collapse | Handbook §02: measurement, response diversity, update size, entropy bonus |
| Growing trainer–rollout score gap | Handbook §03: matching tokens and weights, engine disagreement versus policy movement |
| Tiny, spiking, or nonfinite gradients | Handbook §04: reward → advantage → mask → optimizer → published weights |
| Training reward rises, evaluation stalls or falls | Read [train–eval investigation](references/train-eval-gap.md), then handbook §05 for remedies |
| Hang, crash, OOM | Handbook §06: first failing worker, phase, allocation; use lifecycle/infrastructure skills for operations |
| Slow steps or inconsistent timing | Handbook §07: comparable workload, critical path, phase and worker timings |
| Async or more GPUs fail to help | Handbook §08: usable throughput, queues, policy lag, admitted task mix |

## Test the explanation

For each leading hypothesis, identify supporting evidence, competing
explanations, and a check that could disprove it. Prefer replaying a saved
response through the scorer, inspecting an affected batch, or comparing
fixed prefixes before proposing a new training run. Verify framework support
and the actual implementation before translating a handbook remedy into a
recipe flag; method names are not configuration keys.

When a change is authorized, change one causal factor and hold the relevant
data, budgets, resources, and evaluation protocol fixed. Judge success by task
performance and response quality as well as the metric being repaired.
Repeat uncertain comparisons before treating normal variation as a result.

## Return a diagnosis

Report the affected run/checkpoint and interval, evidence with artifact paths
and sample IDs, the most likely cause and alternatives, and the smallest next
test with its expected distinguishing outcomes. Separate confirmed findings
from hypotheses. Quantify how common a failure is using counts and denominators;
a selected bad response illustrates a failure but does not measure prevalence.
State what remains unavailable or untested, and whether a proposed fix has
actually been validated.
