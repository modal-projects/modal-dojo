# 05 — Training reward improves but evaluation does not

Training reward rises while held-out performance stalls or falls. First establish what the gap measures before diagnosing a failure to generalize.

## Establish what the gap measures

1. **Is evaluation valid?** Confirm the evaluated weight version, completed samples and execution errors. Align results to that version's training update, not their arrival time.
2. **What do the metrics measure?** Check prompt populations, verifier, sampling settings, token/turn/tool budgets and metric definitions. Pass@1 and pass@k answer different questions. Filtering or async arrival order can change the training task mix, raising batch reward without better performance on a fixed distribution.
3. **If the gap persists, where does learning fail to transfer?** Read high-reward training responses and failed evaluation responses. Check for verifier shortcuts, missing task coverage and regressions within particular task types. Inspect split overlap and distribution differences between train vs. eval datasets.

Training and evaluation settings may differ intentionally. State those differences; use matched conditions when investigating their effect on the gap.

## Inspect datasets and responses

Locate the evaluation's saved per-example results and responses, distinguishing
framework-internal `TrainConfig.eval_dataset` results from an independent
offline evaluator. Training rollout exports alone cannot explain eval failures.
Compare raw rows and rendered prompts, system instructions/chat templates,
reference answers or tests, tool/environment setup, and task/difficulty coverage.
Check split overlap, duplication, and answer leakage.

Aggregate discrepancies across all available records in the comparison window
where practical. Inspect a stratified set of actual responses across task types,
checkpoints, high/low training rewards, eval successes/failures, lengths, and
termination/error categories. Expand inspection where a pattern appears. Record
coverage and missing artifacts; selected examples do not measure prevalence.

For stable eval IDs, compare earlier and later responses; compare train and eval
within similar task categories without pairing unrelated rows by position.
Retain artifact paths, sample/attempt IDs, and checkpoint versions. Quantify
scoring disagreements, truncations, tool errors, and capability regressions with
counts and denominators. Compare submitted/completed/accepted task mixes when
available, since filtering and async arrival can hide hard or slow tasks.

Re-score identical saved responses with both scorers when their task definitions
are compatible, using the verifier checks in [01](debug-reward.md). If the gap
persists, evaluate baseline and later weights on a fixed training subset and a
fixed held-out subset under one matched protocol. Keep the target evaluation
alongside that diagnostic comparison; do not move held-out examples into training.

## Match the finding to a remedy

| Confirmed finding | Next action |
| --- | --- |
| Wrong weights, incomplete evaluation or execution errors | Repair evaluation and rerun the affected measurement. |
| Different budgets, sampling settings or reward definitions | Run a diagnostic comparison under matched conditions; retain and document intentional differences in the target evaluation. |
| High reward without genuine task success | Strengthen the verifier or reward specification. Test known correct responses and known failure cases. |
| Improvement concentrated on training examples or narrow task types | Evaluate fixed training and held-out subsets under the same protocol. Identify coverage gaps before changing the training mixture. |
| Some capabilities improve while others regress | Track them separately. Test a targeted task mixture or reference-policy constraint, then measure both gains and regressions. |

**Verify:** keep the held-out protocol fixed across comparisons and inspect per-task results. Repeat uncertain measurements before reacting to small changes; differences comparable to measurement variation do not establish improvement or deterioration. [Evaluation uncertainty](https://arxiv.org/abs/2108.13264)

When comparing the same prompts, use paired per-prompt outcomes or prompt-level
uncertainty estimates; repeated attempts at one prompt are not independent
prompts. Distinguish rescoring old responses from new generation or retraining.
