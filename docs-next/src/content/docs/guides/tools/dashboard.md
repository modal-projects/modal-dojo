---
order: 1
---

# Observability dashboard

When doing RL post-training, experiment management and observability are particularly tricky. While all metrics are captured by the [underlying framework](https://miles.radixark.com/docs), the observability dashboard gives you a live view of the most important information; for example, reward curves, score/advantage distributions, traces, and step timing.

Since it's just a [Modal App](https://modal.com/docs/guide/apps), you can get a dedicated dashboard for your workspace with:

```bash
modal-dojo setup
```

Note that you'll have to rerun this command when you bump your `modal-dojo` version.

Once deployed, open the dashboard any time with:

```bash
modal-dojo open
```

By default, the dashboard is deployed without authentication. To put it behind HTTP Basic Auth, set a password:

```bash
modal-dojo set-password
```

## At a glance

The landing page lists every training run in your workspace:

![Training runs list with annotated components](/observability_dashboard_1_annotated.png)

You can easily see:

1. Status of all your runs.
2. Live progress of each run.

## Per run

Click any run to see a detailed view:

![Run summary view with annotated components](/observability_dashboard_2_annotated.png)

Some highlights:

1. Per-step breakdown of wall-clock time for the run; see more below.
2. Mean reward of all rollouts per step; [past returns do not guarantee future results](https://russellinvestments.com/us/blog/past-performance-no-guarantee-future-results).
3. Score distribution and advantage charts; useful for spotting reward collapse (all-equal rewards) long before the mean flatlines.
4. Link to the underlying Modal app for container-level debugging.

The step and substep timeline is the dashboard's built-in profiler. Each step is segmented by:

- Generate rollouts: sampling from the inference engine.
- Offload rollout/train: moving weights between the rollout and training engines.
- Compute log probs: forward passes for the RL objective.
- Train model: the actual gradient steps.
- Optimizer step, checkpoint save, weight sync: bookkeeping that keeps the rollout engine on-policy.

You can scroll to zoom and drag to pan for precise viewing.

Some tips:

- A healthy GRPO step is usually dominated by rollout generation and model training.
- If rollout generation duration grows step over step, look at rollout lengths as responses may be hitting the max length.
- If weight syncing or the offload substeps dominate, the cluster shape is likely misconfigured for the model size.

Custom phases emitted by your code (e.g., a custom reward function) appear as their own markers for easy debugging and tracking.

## Per rollout

You can even inspect each rollout to quickly debug poor performance:

![Rollouts tab with annotated components](/observability_dashboard_3_annotated.png)

You'll see:

1. Same timeline but for each step.
2. Reward distribution across all rollouts. A healthy run is represented as a bimodal distribution for the majority of the run, while an unhealthy one will gravitate towards one end or the other before training has completed. Pictured above is one indicative of the end of a run, as most rollouts have already saturated the reward.
3. Per-rollout trace that shows the full prompt, the system message, the model's thinking, every conversation turn, and the reward.

## Housekeeping

Metadata from old failed or cancelled runs will accumulate in the dashboard over time. See what will be removed with:

```bash
modal-dojo cleanup --older-than-days 7 --dry-run
```

Then execute:

```bash
modal-dojo cleanup --older-than-days 7
```
