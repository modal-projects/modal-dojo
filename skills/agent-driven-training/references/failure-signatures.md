# 06 — The run hangs, crashes or runs out of memory

## Locate the failure

Record status, phase, current/total step, last update, app ID, first actionable
error, and last meaningful log line. Fetch more logs with `--since` or `--search`
when the tail omits the beginning of a traceback.

**Locate the first failure.** Check the last completed operation and earliest relevant error across workers. Collective/HTTP timeouts may follow a peer failure. For hangs, inspect progress in tools, evaluation, collectives and weight publication.

## Check configuration, data, and runtime failures

| Evidence | Likely cause | Next action |
|---|---|---|
| `ValidationError`, `extra_forbidden`, unknown field, or invalid parallelism | Config or recipe does not match the public schema | Inspect `run params` and the current class definition; correct the invalid field or topology locally |
| `KeyError`, `FileNotFoundError`, missing column, parse exception, or zero prepared rows | Dataset preparation or formatting bug | Reproduce with representative local rows; inspect prepared paths and configured input/output keys |
| NCCL timeout, Ray actor death, worker disconnect, or repeated placement retries | Distributed runtime, node, or capacity problem | Check whether tasks are active and whether one rank failed first; retry transient placement failures without changing training semantics |
| App is not live while run status remains `running` | App exited before metadata finalized | Use the final logs as ground truth and report the stale metadata separately |

Do not apply a table fix mechanically. Confirm that the cited evidence matches
the failing component.

## Inspect memory

For **out-of-memory (OOM)** errors, identify the failing **rank, phase and allocation**.

**Read the pattern.** Compare similar workloads at the same phase boundary.

- **A — Stable post-phase memory:** memory returns to a similar baseline after each phase. If an OOM occurs at a phase peak, reduce or distribute that phase's memory demand using the table below.
- **B — Rising post-phase memory:** more memory remains between phases. Inspect retained tensors, buffers and caches; release or bound what should not accumulate. Growth alone does not prove a leak or guarantee an eventual OOM.

## Match the memory source to the knobs

| Memory source | Candidate changes |
| --- | --- |
| **Training activations** | Smaller microbatch / per-device token budget; [**activation recomputation**](https://docs.nvidia.com/nemo/megatron-bridge/0.3.1/training/activation-recomputation.html). |
| **One long training sequence** | [**Context parallelism (CP)**](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) for sequence activations. |
| **Model weights** | Tensor parallelism (TP), pipeline parallelism (PP), or parameter sharding. |
| **Gradients / optimizer states** | State sharding; CPU offload. |
| **MoE expert weights** | Expert parallelism (EP). |
| **Logits / log-probabilities / entropy** | Smaller token chunks; supported **fused computation** that avoids full token-by-vocabulary intermediates. |
| **Rollout prefill** | Smaller prefill chunks; reserve activation/workspace memory outside the cache pool. |
| **Rollout decoding** | Lower request concurrency; inspect key/value (KV) cache capacity and retained session state. |
| **Weight publication / colocated transitions** | Reduce duplicate weights and transfer buffers; release or offload inactive state. |
| **Host RAM** | Bound trajectory queues and tool concurrency; clean up retained processes; provision memory. |

**Parallelism:** CP targets sequence activations. Megatron-style **sequence parallelism (SP)** partitions selected activations and requires TP > 1. Ordinary **data parallelism (DP)** replicates model state; more replicas do not automatically reduce its per-device footprint. Check model/backend compatibility. [Parallelism and sharding](https://docs.nvidia.com/nemo/megatron-bridge/0.3.1/parallelisms.html)

**Rollout memory:** increasing cache reservation leaves less room for activations and CUDA graphs. Prefill OOM and cache exhaustion need different adjustments. [Inference tuning](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hyperparameter_tuning.mdx)

**Memory accounting:** compare equivalent workloads at the same phase boundary. Allocated memory tracks tensor storage; reserved memory also includes unused allocator blocks. High reservation alone does not prove a leak; a process exit alone does not prove OOM. [PyTorch memory accounting](https://docs.pytorch.org/docs/2.8/notes/cuda.html#memory-management)

**Preserve the experiment:** use gradient accumulation to retain the intended effective batch when reducing microbatch size. Preserve loss normalization. Shortening responses or dropping long samples changes the learning problem.

**Verify:** repeat the failing workload, including long sequences and phase transitions; then measure throughput ([07](debug-systems.md)). Bound infrastructure retries. Repeated failure at the same operation needs diagnosis; longer timeouts cannot repair a crashed worker.

For an apparently hung final evaluation, inspect
[phase timing and final-step evaluation](debug-systems.md#final-step-evaluation).
Use [modal-infrastructure](../../modal-infrastructure/SKILL.md) for raw worker,
container, volume, or capacity investigation when CLI evidence is insufficient.
Cleanup and relaunch follow the [training lifecycle](../SKILL.md#stop-or-relaunch).
