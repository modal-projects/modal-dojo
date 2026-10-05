# 07 — Steps are slow or timing totals look wrong

## Establish the workload and timing baseline

**Establish a reference.** There is no universal step time for a model size. Architecture, precision, sequence lengths, batch size, hardware, parallelism and environment workload all affect it.

- **With a comparable run or benchmark:** compare steady-state timings at similar token counts and concurrency. Investigate unexplained regressions; a known-working recipe is not necessarily efficient.
- **Without one:** profile a short run after warmup using representative inputs. Record phase times, token counts and throughput across several updates. Use this as the baseline for controlled changes.

Separate startup (image build, capacity placement, model download/conversion,
engine initialization) from steady-state updates. Check cache misses and startup
logs before tuning the training loop. Keep topology fixed unless it is the
variable under test: changing topology may invalidate converted/shared-cache
layouts and introduce a new startup cost.

**Use cost estimates as a sanity check.** Transfer time is roughly **bytes transferred / effective bandwidth**. Full weight synchronization also includes export, resharding, loading and synchronization overhead. [Published transfer estimates](https://modal.com/blog/reinforcement-learning-infrastructure-problem) are useful only with their stated bandwidth and transfer assumptions.

Then find the **critical path**: the work that determines when the next batch or update can finish. Prioritize unexplained delays and avoidable work on this path.

**Interpret phase timings.** Extra step time may be outside generation. When
phases overlap, adding their durations overcounts wall time. Inspect boundaries
and overlap before attributing a slowdown to any one phase.

**Check next, in order.**

1. **Validate timers.** Align boundaries and units; distinguish GPU dispatch from completion. Separate warmup, evaluation and checkpoint-writing costs from ordinary updates. Retrieving buffered async samples does not measure their generation time. [GPU timing](https://docs.pytorch.org/docs/2.8/notes/cuda.html#asynchronous-execution)
2. **Check the workload.** Compare prompt/response tokens, tool calls and generation attempts per update. Longer responses or lower filtering acceptance can increase step time without a system regression.
3. **Compare workers and ranks.** Break down generation, tools, reward computation, training and transfers. Inspect slow workers and what idle workers are waiting for; averages can hide a straggler.

Speeding up already-fast workers cannot finish a batch earlier if it still
waits for a straggler. For the slow worker, compare tool-request, execution-start
and completion timestamps: long tool latency can come from waiting, a slow
sandbox or legitimately expensive work.

## Test the measured bottleneck

| Measured bottleneck | What to inspect / adjust |
| --- | --- |
| **Rollout prefill** | Prompt lengths, prefix-cache reuse and prefill batching/chunking. |
| **Rollout decoding** | Active/queued requests and KV-cache pressure. Tune request concurrency and engine parallelism. Test supported **low-precision rollout** or **speculative decoding**, including conversion/draft costs. |
| **Trainer computation** | Microbatching, packing and work per GPU. Reduce recomputation when memory permits. |
| **Communication / load imbalance** | Token and expert load across ranks; collective time and network topology. Revisit TP/EP/CP/PP and state sharding against their communication costs. [Parallelism guide](https://docs.nvidia.com/nemo/megatron-bridge/0.3.1/parallelisms.html) |
| **Tools / sandbox** | Queueing, startup, execution and retries. Check CPU throttling, RAM pressure and I/O; tune sandbox concurrency and per-sandbox resources. [Resource limits](https://docs.docker.com/engine/containers/resource_constraints/) |
| **Reward / host processing** | Verification, tokenization and serialization. Check blocking calls and worker-pool saturation; batch or parallelize independent work. |
| **Data / weight transfer** | Transferred bytes, duplicate copies and weight-publication pauses. Reduce redundant transfers; overlap work where dependencies allow. |

Test higher concurrency when work is ready but capacity is underused; lower it if contention, cache pressure or retries grow. Measure throughput and tail latency together. [Inference tuning](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hyperparameter_tuning.mdx)

**If a smaller smoke test hangs or fails:** lower concurrency can exercise different engine paths, including combinations of idle, prefilling and decoding ranks. Compare with a known-working configuration and inspect engine errors before interpreting the change as ordinary scaling behavior. [Example engine failure](https://github.com/sgl-project/sglang/pull/34535)

Low-precision support depends on the model, format, hardware and weight-export path; trainer changes may be required. Recheck scoring agreement and learning. [Precision compatibility](https://github.com/radixark/miles/blob/8760515851ec4f76815171d812c836bf8caa0aca/docs/advanced/low-precision.md)

## Check prefix-cache reuse

 miles' **`rollout/prefix_cache_hit_rate`** is summed cached prompt tokens divided by summed recorded prompt tokens. This is a token-weighted fraction, not the fraction of requests with a cache hit. Compare it with prompt lengths and prefill time; missing metadata can produce zero.

**Interpret and act.**

- **Consistently high:** if the run remains slow, inspect decoding, tools and other phases.
- **Persistently low despite shared history:** check exact prefixes, cache eviction and routing/session affinity. Restore reuse where expected, then remeasure prefill and full-loop time.
- **Repeated drops:** align them with weight updates, cache resets, routing and input-length changes. Measure expected refill costs; investigate unexplained drops.

High expected cache reuse assumes a warm workload with shared history. New prefixes or large tool outputs can lower reuse. Cached states must be valid for the active weights. Reuse saves prefill work, not decoding or tool execution. [Prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) · [Routing and load balance](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/policies/cache_aware.rs)

**Verify:** hold task mix, effective batch size and generation/tool budgets fixed. Compare usable trajectories or updates per second, GPU-hours per update and trajectory tail latency; report extra CPU/sandbox resources. Preserve timeout and failure handling. Higher utilization or cache hits alone do not establish a speedup.

## Final-step evaluation

When the final step matches `eval_interval`, Slime may spend many minutes in
`evaluate_rollouts` or appear to remain in `weight_sync` while held-out
generation completes. Before stopping it, confirm whether logs show live
SGLang generation and whether the Modal app still has an active task. If both
are true, continue monitoring.
