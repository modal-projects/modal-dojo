# Rollout fault tolerance — design

Status: proposal. This document is a design sketch for review; no
implementation is attached. The intent is to agree on semantics and the
config surface before a PR lands.

## Problem

RL rollout is the least deterministic stage of a training run, and today a
small number of bad conversations or one bad worker can abort an entire
optimizer step — observed failure shapes include:

- A single episode hitting the rollout timeout produces zero samples for
  its group, and the step hard-fails on the first step.
- A handful of aborted conversations exceed a fixed per-group tolerance and
  the batch is rejected mid-step.
- An eval pass where zero conversations score (scorer misconfig, parser
  drift, model refusing format) fails with a generic "no conversation was
  scored" after a full rollout.

Two properties make these expensive: they surface *after* the cluster has
already done minutes-to-hours of rollout work, and the retry unit is the
whole app — not the conversation.

## Goals

- A bounded, configurable retry for a failed/aborted conversation before it
  counts against group tolerance.
- Configurable partial-group admission: proceed with a reduced batch when
  enough groups are complete, rather than aborting the step.
- Fast failure for degenerate steps: detect "zero conversations scored"
  early and attribute the cause (scorer error, parse failure, timeout)
  instead of a bare count.

## Non-goals

- Changing the RL objective. Group-relative methods (GRPO) have batch
  semantics; fault tolerance must stay a thin transport layer, not a new
  estimator. Anything that biases the sample distribution beyond a bounded
  retry should be explicit opt-in.
- Reimplementing rollout engines. Where Miles/Slime already expose a hook
  (per-conversation callbacks, abort categories), use it; Dojo should own
  the policy, the framework owns the mechanics.

## Design

### 1. Conversation retry

Each aborted conversation gets up to `rollout_max_retries` (default 2)
re-dispatches before it counts as failed. Only transient categories retry —
timeout, worker crash, engine restart. `trial_error`/`tool_error` outcomes
that are deterministic (bad prompt, unparseable tool call) retry zero
times: retrying a deterministic failure just burns tokens.

Retries re-enter the same scheduler queue rather than a synchronous loop,
so a wedged engine can't serialize the whole batch behind one prompt.

### 2. Partial-group admission

A group is admitted when `completed >= ceil(n_samples_per_prompt *
min_group_completion)` (default `min_group_completion=1.0` — today's strict
behavior). The step proceeds when the fraction of admitted groups is at
least `min_step_groups` (default 1.0).

Setting either below 1.0 is a training-semantics change and must be
opt-in: a partial group shifts the advantage baseline. The runner should
log the admitted/total counts per step so skew is visible in metrics.

### 3. Degenerate-step fast failure

Before a step trains on rollout output, check a preflight predicate:
`scored > 0` (and more generally `admitted_groups > 0`). When the check
fails, aggregate the per-conversation abort categories into the failure
reason — "0/48 scored: 44 scorer_error, 4 timeout" — instead of a bare
"no conversation was scored".

### 4. Pathological-prompt quarantine

A conversation that fails `rollout_max_retries` times with the same
category gets its prompt hash recorded in run metadata; if the same hash
fails again on a later step, it is skipped without consuming retries. This
keeps one poisoned prompt from repeatedly draining the retry budget across
steps.

## Config surface

On the recipe (framework recipes map these to their rollout args; unknown
or unsupported keys surface a validation warning at recipe construction
rather than a remote failure):

| field | default | meaning |
| --- | --- | --- |
| `rollout_max_retries` | 2 | transient-failure retries per conversation |
| `min_group_completion` | 1.0 | fraction of a group required for admission |
| `min_step_groups` | 1.0 | fraction of groups required to run the step |
| `rollout_quarantine` | false | skip repeat pathological prompts |

Defaults preserve current behavior exactly; only `rollout_max_retries`
changes anything, and it is bounded and logged.

## Risks / open questions

- Distribution skew: partial admission changes which groups contribute to
  the step. Acceptable when opt-in; needs the logged admission rate.
- Retry storms: a wedged inference engine turns N retries into N×waste.
  Mitigate by capping retries per step globally, not just per conversation.
- Upstream vs Dojo: the abort categories and per-conversation scheduling
  live in Miles/Slime. If the frameworks can't expose per-conversation
  retry hooks cleanly, the honest path is a small upstream patch rather
  than a Dojo-side wrapper that re-runs whole groups.
