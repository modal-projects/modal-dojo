# 05 — Training reward improves but evaluation does not

Training reward rises while held-out performance stalls or falls. First establish what the gap measures before diagnosing a failure to generalize.

## Establish what the gap measures

1. **Is evaluation valid?** Confirm the evaluated weight version, completed samples and execution errors. Align results to that version's training update, not their arrival time.
2. **What do the metrics measure?** Check prompt populations, verifier, sampling settings, token/turn/tool budgets and metric definitions. Pass@1 and pass@k answer different questions. Filtering or async arrival order can change the training task mix, raising batch reward without better performance on a fixed distribution.
3. **If the gap persists, where does learning fail to transfer?** Read high-reward training responses and failed evaluation responses. Check for verifier shortcuts, missing task coverage and regressions within particular task types. Inspect split overlap and distribution differences between train vs. eval datasets.

If the gap persists, use [trace monitoring](../SKILL.md#trace-monitoring) for a
few training examples and obtain evaluation responses from the evaluator's
saved artifacts separately. Compare similar task types at matching checkpoints.

Training and evaluation settings may differ intentionally. State those differences; use matched conditions when investigating their effect on the gap.

## Match the finding to a remedy

| Confirmed finding | Next action |
| --- | --- |
| Wrong weights, incomplete evaluation or execution errors | Repair evaluation and rerun the affected measurement. |
| Different budgets, sampling settings or reward definitions | Run a diagnostic comparison under matched conditions; retain and document intentional differences in the target evaluation. |
| High reward without genuine task success | Strengthen the verifier or reward specification. Test known correct responses and known failure cases. |
| Improvement concentrated on training examples or narrow task types | Evaluate fixed training and held-out subsets under the same protocol. Identify coverage gaps before changing the training mixture. |
| Some capabilities improve while others regress | Track them separately. Test a targeted task mixture or reference-policy constraint, then measure both gains and regressions. |

**Verify:** keep the held-out protocol fixed across comparisons and inspect per-task results. Repeat uncertain measurements before reacting to small changes; differences comparable to measurement variation do not establish improvement or deterioration.
