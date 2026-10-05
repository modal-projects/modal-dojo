# Debug reward

Use this reference when reward is flat, declining, saturated, unexpectedly
volatile, or improving suspiciously fast.

For training reward improving without evaluation gains, follow the
[train–eval investigation](../../rl-debugging/references/train-eval-gap.md).
Compare the underlying datasets, responses, and scoring paths before inferring
overfitting or changing the reward or recipe.

## Characterize the trajectory

```bash
modal-dojo run get <run-id> --verbose
```

Compare the baseline, early steps, recent steps, sample counts, and variance.
Do not infer learning from the final value alone.

Before interpreting the curve, identify what population the algorithm reports.
Algorithms that oversample or dynamically filter candidates may report reward
over a changing retained population, so their reward need not rise
monotonically. DAPO is one example. A plateau alone is not a stop signal;
compare fixed task metrics, filter/retention counts, and representative traces.
Use `run params` to confirm the sampling configuration, `run get --verbose` to
inspect rollout summaries and task metrics, and `run trace` to inspect
filtering evidence and raw samples.

- Flat near chance: the model may not be learning, the signal may be sparse,
  or extraction may return zero broadly.
- Declining: inspect regressions, truncation, instability, and whether the
  optimization target matches the task metric.
- Near the ceiling from the first rollout: the base model may already solve the
  task, the evaluation set may be too easy, or the reward may leak or overmatch.
- Abrupt jump to perfect reward: inspect for answer leakage, permissive parsing,
  duplicated samples, and reward hacking before treating it as success.

Ceiling thresholds are task-relative. Require enough steps and samples to
separate saturation from normal variance.

## Decide whether to stop an active run

Set an early efficacy checkpoint proportional to the run length; for example,
reassess a 150-step run across roughly steps 10–40. If enough comparable
samples and fixed task metrics show that learning remains flat outside normal
noise, declines, or is otherwise uninformative, stop the run instead of waiting
for completion. Do not stop on a single noisy point or an algorithm-expected plateau, 
but do not keep a healthy yet ineffective job alive only because it has not failed.

To stop it, use `modal-dojo run get <run-id>` to obtain the Modal
app ID, then:

```bash
modal app stop <app-id>
```

Diagnose the trajectory, change one parameter, and launch a fresh smoke test
before promoting again.

## Download representative traces

Choose baseline, anomalous, transition, and recent steps based on the observed
reward trajectory, then download their traces after a dry-run:

```bash
modal-dojo run trace <run-id> --out ./traces --step <steps>
```

## Classify samples

Compare both low- and high-reward samples. Classify:

- correct answer scored incorrectly,
- incorrect or malformed answer scored as correct,
- prediction extracted from the prompt/reference instead of the response,
- response truncation or missing final answer,
- repeated tool-call or environment failures,
- parser mismatch with otherwise valid answers,
- repeated templates, copied references, or other reward-hacking behavior,
- infrastructure errors represented as task failures.

Recompute the reward locally for representative samples using the exact
prompt, response, and reference fields. Add fixture cases for every discovered
false positive or false negative.

## Decide the next experiment

- Reward implementation bug: fix it and rerun the one-step proof.
- Task is already saturated: make the task or evaluation more discriminative;
  do not launch a full horizon to produce an uninformative curve.
- Sparse but valid signal: adjust one reward or sampling choice and repeat the
  smoke test.
- Base model almost never crosses a required correctness gate: propose a
  stronger model or a justified graded signal/curriculum.
- Model behavior problem: change one training setting and compare fresh runs
  over equivalent steps and samples.

Never silently redefine the task metric merely to make the curve rise.
