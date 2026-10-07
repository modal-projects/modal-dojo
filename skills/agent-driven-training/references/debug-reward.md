# 01 — Reward is flat or unexpectedly low

## Inspect attempts and scoring

Compare the baseline, early steps, recent steps, sample counts, and variance.
Do not infer learning from the final value alone.

- Near the ceiling from the first rollout: the base model may already solve the
  task, the evaluation set may be too easy, or the reward may leak or overmatch.
- Abrupt jump to perfect reward: inspect for answer leakage, permissive parsing,
  duplicated samples, and reward hacking before treating it as success.

**Inspect the attempt → validate the verifier → interpret the reward distribution.**

| Inspect first | What should hold |
| --- | --- |
| [**Generation trace**](../SKILL.md#trace-monitoring) | Confirm the intended prompt and tool observations reached the policy. Distinguish wrong answers, budget exhaustion, repetitive loops and tool failures. Check the termination reason and end-of-sequence (EOS)/stop settings. |
| **Verifier input and decision** | Confirm the verifier received the intended answer or artifact. Inspect extraction and tests using known correct and incorrect examples. Separate verifier errors from valid failure judgments. |
| **Recorded reward** | Confirm the value matches the verifier's decision and the stated objective. Count truncations and execution errors separately. Repair generation or scoring errors before tuning learning. |

Recompute the reward locally for representative samples using the exact
prompt, response, and reference fields. Add fixture cases for every discovered
false positive or false negative.

**If attempts end too early:** fix unintended stop conditions; increase the budget (`max_length`) only if the task objective permits. If success must fit a fixed budget, keep that constraint explicit in scoring and evaluation. Masking truncated responses removes their direct policy-loss contribution, but their rewards may still affect other group members' advantages. Check the implementation.

## Inspect group contrast and choose an intervention

For Group Relative Policy Optimization (GRPO) and similar methods, next inspect rewards **within each prompt group**. The example uses binary rewards; with graded rewards, inspect within-group variation.

**Compare prompt groups, not only batch accuracy.** Batches with the same
overall accuracy can have different learning signal: mixed correct/incorrect
responses within each group provide reward contrast, while a batch of all-correct
and all-wrong groups does not. In uniform groups, centered advantages are zero:
this reward term supplies no policy-gradient signal, though auxiliary losses
may still contribute.

Uniform groups are normal. Investigate a sustained lack of reward contrast across the training data.

| Pattern across batches | Interpretation and possible response |
| --- | --- |
| **Mostly all wrong** | Successes may be too rare under this policy and budget. Consider a curriculum, stronger initial policy or larger budget if the objective permits. Filtering cannot create successful attempts. |
| **Mostly all correct** | The policy may already meet the objective. Otherwise, introduce harder or underrepresented tasks. |
| **Mixed rewards, little progress** | Inspect exploration ([02](debug-entropy.md)) and the update ([04](debug-updates.md)). Reward contrast alone does not guarantee improvement. |

**If informative groups are scarce but obtainable:** [dynamic sampling](https://arxiv.org/html/2503.14476v2#S3.SS2) fills each fixed-size training batch with mixed-reward groups. For binary rewards, retain groups with **at least one correct and one incorrect response**; discard uniform groups. Keep accepted groups while sampling more until the target is met, then update the policy.

Track **group acceptance rate** (accepted groups divided by sampled groups) and collection time. Low acceptance means more generation per retained group; check whether rejections are mainly all-wrong or all-correct. There is no universal cutoff: judge whether the additional learning signal justifies the collection cost.

**Filtering can help learning, but a change in training-batch accuracy is not proof.** Selection changes which outcomes enter that metric, so it no longer estimates success on the original prompt distribution. Compare unfiltered held-out improvement **per update and per elapsed time** against a run without filtering. Training metrics still diagnose reward contrast, gradients and stability.
