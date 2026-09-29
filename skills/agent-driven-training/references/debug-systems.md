# Debug systems performance

Use this reference when startup, rollout, training, synchronization, or
evaluation is slower or less stable than expected.

## Locate the time

```bash
modal-dojo run get <run-id> --verbose
modal-dojo run params <run-id>
modal-dojo run logs <run-id> --tail 200
```

Separate one-time startup from steady-state step time. Identify the dominant
phase:

- model download or conversion,
- rollout-engine initialization,
- rollout generation or evaluation,
- log-prob computation,
- optimizer step,
- weight synchronization or offload, or
- checkpoint save.

Do not call a run slow from one step. Use several comparable steady-state steps
and account for evaluation/checkpoint intervals.

## Form a targeted experiment

Match the setting to the bottleneck:

- Rollout generation: inspect response length, rollout batch/concurrency,
  sampling count, rollout tensor parallelism, and generation errors.
- Log-prob or optimizer work: inspect global/micro batch sizes, token limits,
  gradient accumulation, training parallelism, and CPU/GPU offload.
- Weight sync or offload: inspect model size, sync frequency, topology, and
  whether the phase label is waiting for a following periodic action.
- Checkpoint/evaluation: inspect intervals, sample counts, response limits, and
  storage activity.
- Startup/conversion: distinguish image build, cache miss, download,
  conversion, and capacity scheduling before tuning the training loop.

Change one setting at a time. Keep the model, dataset slice, GPU topology, and
measurement window fixed unless topology is the variable under test. Changing
topology can invalidate converted or shared-cache layouts and introduce a new
startup cost.

## Confirm improvement

For baseline and candidate runs, compare:

- at least several equivalent non-eval steps,
- per-phase and total step time,
- samples/tokens processed,
- reward quality and error rate, and
- startup costs separately from steady state.

Treat differences within normal step-to-step variation as noise. A speedup that
reduces reward quality, changes the workload, or adds instability is not an
improvement.

## Final-step evaluation

When the final step matches `eval_interval`, Slime may spend many minutes in
`evaluate_rollouts` or appear to remain in `weight_sync` while held-out
generation completes. Before stopping it, confirm whether logs show live
SGLang generation and whether the Modal app still has an active task. If both
are true, continue monitoring.
