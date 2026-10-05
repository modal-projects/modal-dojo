# Training reward improves but evaluation does not

Establish whether the gap is caused by measurement, execution, selection,
scoring, or behavior that fails to transfer. Do not diagnose overfitting from
two aggregate curves. This procedure expands handbook §05 into an investigation
of the underlying datasets and responses.

## 1. Align the comparison

Identify a baseline checkpoint, the first divergence, and a recent checkpoint.
Associate evaluation with the weights actually evaluated, not its completion
timestamp. Check export/loading acknowledgments where available; a run label
does not prove which checkpoint an endpoint served. Confirm completion counts,
missing results, retries, timeouts, and evaluator errors.

Build a compact comparison of training and evaluation protocols:

| Dimension | Evidence to collect on both sides |
| --- | --- |
| Data | Source/revision, split, row IDs, task categories, difficulty proxies, reference answers/tests, duplicates and overlap |
| Inputs | Raw rows and rendered messages, system prompt, chat template, tokenizer, tool schema/observations, answer format |
| Generation | Weight version, temperature, top-p/top-k, number of attempts, token/turn/tool budgets, stop conditions |
| Scoring | Answer extraction, verifier/reward code and version, component rewards, aggregation, error handling |
| Reported metric | Reward versus task success, pass@1 versus pass@k, sample versus prompt weighting, completed versus accepted population |

Mark unknown values explicitly. Document intentional train/eval differences;
the target evaluation protocol remains the standard even when a diagnostic
comparison temporarily matches training conditions.

## 2. Inspect the data and response populations

Use `run trace` for training rollouts. For evaluation, determine whether the
metric came from the framework's internal `TrainConfig.eval_dataset` loop or
an independent offline evaluator. Find that evaluator's saved per-example
results, responses, configuration, and logs; training exports alone cannot
explain evaluation failures. Trace dataset materialization and prompt
formatting from the relevant code when artifacts disagree.

Inventory all available records in the selected comparison intervals. Compute
aggregate checks over all of them where practical, streaming large exports.
Read a stratified set of actual prompts and responses rather than attempting
to fit every trace into context. Include each task family, early and late
checkpoints, high and low rewards, eval successes and failures, length extremes,
and error/termination categories. Expand inspection when a discrepancy appears.
If sampling is necessary, record the selection rule, seed, counts, and coverage;
do not claim an exhaustive audit from a handful of examples.

Preserve provenance for each inspected attempt: source artifact, row/prompt
ID, attempt/group ID, checkpoint/update, raw and rendered prompt, response or
tool transcript, reference, extracted prediction, reward components, verdict,
length, termination, and execution errors when retained. Keep unavailable
fields unknown. Join checkpoints by stable example IDs; use a documented
content hash when IDs are missing. Do not pair unrelated train/eval rows by
position or silently merge repeated attempts.

Check dataset-level discrepancies:

- Task/domain and difficulty coverage; language, modality, prompt length,
  reference format, reasoning/tool requirements, and answerability.
- Duplicate and near-duplicate prompts, split overlap, and answer leakage in
  prompts or tool observations. Distinguish benign shared boilerplate from
  duplicated tasks; report the matching rule and counts.
- Raw versus rendered input: dropped fields, double chat templating, different
  system instructions, truncation, or inconsistent tool/environment setup.
- Submitted, completed, and accepted task mix. Dynamic filtering, timeouts,
  async completion, and stale rejection can favor easier or shorter tasks.
  Missing rejected attempts limit claims about the original population.

## 3. Explain the discrepancies with examples and counts

Inspect high-reward training responses beside failed evaluation responses
from comparable task categories. Also inspect training failures and evaluation
successes to check whether the apparent pattern is specific. For fixed eval
IDs, compare earlier and later responses to locate regressions.

Classify observed discrepancies, retaining sample IDs and supporting excerpts:

- Genuine task failure despite valid generation and scoring.
- Correct result rejected by answer extraction, formatting, or verifier bugs.
- Incorrect result rewarded through a shortcut, leakage, or permissive parsing.
- Token/turn budget exhaustion, missing final answers, repetition, or incoherence.
- Tool/environment failures or evaluator errors represented as model failures.
- A capability absent from training or a regression within a task category.

For each category, report its count/denominator on both sides and across
checkpoints, response-length and truncation/error rates, and reward versus
task-success agreement where an independent success judgment exists. Explain
whether categories overlap. Keep an unknown category rather than forcing an
unsupported judgment. Break aggregate changes down by task type so a changed
mixture cannot conceal regressions.

Re-score saved responses through the exact reward/extraction path when it can
be reproduced. Exercise known correct, incorrect, malformed, and missing-answer
cases. When both scorers apply to the same task, score the same saved responses
with both to isolate verifier differences. Otherwise identify the incompatible
objective and construct appropriate fixtures; do not compare incomparable
scores. Preserve required sandbox/environment semantics for executable answers,
and label stochastic scorer variation or missing environment state.

## 4. Choose the smallest discriminating experiment

Use the evidence to choose a test; do not launch every experiment by default.
New generation or training follows the lifecycle skill and existing scope.

| Evidence | Diagnostic comparison |
| --- | --- |
| Stale/wrong weights or incomplete eval | Re-evaluate the identified checkpoint and reconcile completion/error counts |
| Different prompts, tools, sampling, or budgets | Hold weights and examples fixed; change the suspected protocol difference alone |
| Different verifier decisions | Re-score identical saved responses; fix the failing extraction/verifier case and remeasure |
| Accepted training reward rises as task mix changes | Compare unfiltered metrics if retained, or evaluate a fixed training subset and fixed held-out subset under the same protocol |
| Only fixed training examples improve | Compare both subsets at baseline and later weights; inspect duplication, coverage, and difficulty before calling it memorization |
| Gains in one task family hide regressions elsewhere | Compare per-family outcomes at fixed weights/protocol, then test a targeted mixture or constraint if authorized |

A useful matched comparison evaluates baseline and later weights on both a
fixed training subset and a fixed held-out subset, with one shared protocol.
This separates learning on seen examples from transfer without comparing
changing training batches to a fixed benchmark. Keep the target evaluation
alongside this diagnostic result. Avoid moving held-out examples into training;
if repeated diagnosis shapes development choices, reserve a separate untouched
test set for the final assessment.

## 5. Verify and report

Report the protocol differences, dataset/response coverage, quantified failure
categories, representative paired evidence, and supported cause or unresolved
alternatives. Name the next comparison and what result would support or refute
the hypothesis. Keep example counts and uncertainty visible; repeated attempts
at one prompt are not independent prompts. Use paired per-prompt outcomes or
prompt-level uncertainty estimates when comparing the same evaluation set.

A repaired curve is insufficient: verify corrected scoring/execution and
held-out task success under the fixed target protocol, including categories
that previously regressed. Distinguish rescoring existing responses from
generating new responses or retraining; each supports a different conclusion.
