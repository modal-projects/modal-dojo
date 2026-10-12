# 08 — Async or more GPUs do not help

Check **usable training throughput, policy lag and held-out learning** together. Faster generation helps only if training can use the resulting experience effectively.

## Locate the waiting

Track completed experience waiting for training **alongside worker activity**. Distinguish in-flight trajectories from completed responses; count complete groups when the loss requires them.

**Interpret the queue.** An empty queue matters when the trainer actually
waits for work. A growing queue means usable arrivals exceed consumption over
the interval; check whether the trainer is saturated or blocked.

| Pattern | Check / next action |
| --- | --- |
| **Queue empty; trainer waiting** | Inspect generation, tools, reward computation and rejection counts. Fix or scale the stage limiting usable arrivals. |
| **Responses finish; few groups become ready** | In grouped training such as GRPO, inspect incomplete groups and their slowest members. Refill generation slots independently where supported; this does not remove the group's training requirement. |
| **Queue growing or full** | Check whether the trainer is busy or blocked. Speed up the measured training bottleneck or reduce production; a larger buffer does not increase consumption capacity. |
| **Periodic stalls** | Align pauses with weight publication and evaluation. Inspect export, transfer, loading and cache rebuild time. More frequent publication trades lower lag for more synchronization cost. |

A bounded queue with little idle time can be healthy. Check its sampling point: a size recorded just after draining a batch can be zero even when production is sufficient. [Buffering and publication](https://github.com/radixark/miles/blob/main/docs/user-guide/fully-async.md)

**More GPUs, little gain:** remeasure the rollout/trainer allocation, per-GPU batch size, communication and CPU/tool limits ([07](debug-systems.md)). Retune parallelism for the new allocation; more rollout GPUs cannot accelerate a saturated trainer.

## Locate the policy lag

Record which trainer update produced the rollout weights and when the experience enters training. **Elapsed age, optimizer-update lag and publication count are different quantities.** One publication can span several optimizer updates.

Compare lag distributions for **admitted and discarded** experience separately; check whether logged aggregates include rejected samples. Version lag is a proxy for policy change, not a universal stability threshold.

| Where lag accumulates | Check / next action |
| --- | --- |
| **Before generation** | Compare trainer, published and actually loaded versions across engines. Fix stuck or wrong weights; shorten publication delays where practical. |
| **During generation / reward computation** | Inspect long trajectories, tool waits and verifier latency. Supported pause/resume or partial-rollout scheduling can reduce barriers, but unfinished work still needs correct continuation. |
| **After completion** | Inspect buffer residence time and excess production. Reduce prefetch/concurrency or speed up consumption; test an explicit staleness limit. |

If weights change within a trajectory, preserve original token IDs, prefixes, behavior log-probabilities and version provenance. Resuming under newer weights does not make earlier tokens fresh; rescoring with those weights cannot recover the original behavior probabilities. Check cache and continuation semantics. [Interruptible generation](https://arxiv.org/html/2505.24298v5#S4.SS1) · [Partial rollouts](https://arxiv.org/html/2509.18521v1#S2.SS2)

## If throughput improves but learning deteriorates

1. **Check off-policy updates.** Inspect probability ratios, clipping/rejection fractions and retained training signal by lag. Confirm which policies each ratio compares and that the loss handles stale behavior data. Separate same-version engine disagreement ([03](debug-logprobs.md)) from policy movement.
2. **Test the lag–throughput trade-off.** Reduce publication or queue delays, or tighten the staleness bound. Track lost throughput and discarded work; a tighter bound can starve training. Importance correction does not make arbitrary lag harmless. [Staleness and correction ablations](https://arxiv.org/html/2505.24298v5#S7.SS4)
3. **Check the task mix.** Track submission, completion and admission by task type; compare lengths and rewards for completed and admitted samples. Timeouts, cancellation and stale rejection can underrepresent slow tasks. Count wasted tokens and retries; higher accepted-batch reward may reflect selection.

**Verify:** compare held-out performance per update and elapsed time under the same task and evaluation budgets. Match total resources for sync/async comparisons; report GPU-hours and extra CPU/sandbox/eval capacity. Evaluate fixed weights and label results by their version/update; record completion lag separately. Shared-engine eval can pause production; dedicated eval still has export and resource costs.
