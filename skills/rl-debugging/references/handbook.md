# RL Quick Reference Handbook (QRH)

**Before you debug: data, infrastructure and algorithms interact**

**Poor learning may originate in the data.** Task difficulty relative to the initial policy, unreliable rewards, limited coverage and differences between training and evaluation can all limit progress. Check these before assuming an algorithmic failure. Data, infrastructure and algorithms interact: generation budgets and scheduling affect the experience available for training, while task characteristics affect both learning signal and system workload.

**This guide synthesizes published research and practical experience.** Its patterns suggest hypotheses to investigate, not universal diagnoses or guaranteed fixes. A metric trend alone does not establish a cause, and schematic curves are illustrations, not evidence. Inspect actual examples, validate the generation and scoring paths, and use controlled comparisons to check whether a proposed fix helps your task.

**Find the symptom → inspect the evidence → test a fix → verify the result.** Use the numbered sections below to find the relevant checks.

| No. | What you see |
| --- | --- |
| 01 | [Reward is flat or unexpectedly low](#01--reward-is-not-increasing) |
| 02 | [Entropy behaves unexpectedly, or performance collapses](#02--entropy-is-changing) |
| 03 | [Trainer–rollout log-probability differences grow](#03--log-probability-mismatch-is-increasing) |
| 04 | [Gradients are tiny, spike or become nonfinite](#04--updates-are-ineffective-or-unstable) |
| 05 | [Training reward improves but evaluation does not](#05--training-improves-but-evaluation-does-not) |
| 06 | [The run hangs, crashes or runs out of memory](#06--the-run-hangs-crashes-or-runs-out-of-memory) |
| 07 | [Steps are slow or timing totals look wrong](#07--the-run-is-too-slow) |
| 08 | [Async or more GPUs do not help](#08--async-or-more-gpus-do-not-help) |

Adapted from the supplied RL Quick Reference Handbook. Its image assets were not
included in the export; the diagnostic patterns below are described in text.
Published-source links and recorded run examples are retained as background,
not evidence about the run currently being diagnosed. Missing or disabled
metrics are unknown, not zero.

## 01 — Reward is not increasing

**Inspect the attempt → validate the verifier → interpret the reward distribution.**

| Inspect first | What should hold |
| --- | --- |
| **Generation trace** | Confirm the intended prompt and tool observations reached the policy. Distinguish wrong answers, budget exhaustion, repetitive loops and tool failures. Check the termination reason and end-of-sequence (EOS)/stop settings. |
| **Verifier input and decision** | Confirm the verifier received the intended answer or artifact. Inspect extraction and tests using known correct and incorrect examples. Separate verifier errors from valid failure judgments. |
| **Recorded reward** | Confirm the value matches the verifier's decision and the stated objective. Count truncations and execution errors separately. Repair generation or scoring errors before tuning learning. |

**If attempts end too early:** fix unintended stop conditions; increase the budget (`max_length`) only if the task objective permits. If success must fit a fixed budget, keep that constraint explicit in scoring and evaluation. Masking truncated responses removes their direct policy-loss contribution, but their rewards may still affect other group members' advantages. Check the implementation.

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
| **Mixed rewards, little progress** | Inspect exploration (section 02) and the update(section 04). Reward contrast alone does not guarantee improvement. |

**If informative groups are scarce but obtainable:** [dynamic sampling](https://arxiv.org/html/2503.14476v2#S3.SS2) fills each fixed-size training batch with mixed-reward groups. For binary rewards, retain groups with **at least one correct and one incorrect response**; discard uniform groups. Keep accepted groups while sampling more until the target is met, then update the policy.

Track **group acceptance rate** (accepted groups divided by sampled groups) and collection time. Low acceptance means more generation per retained group; check whether rejections are mainly all-wrong or all-correct. There is no universal cutoff: judge whether the additional learning signal justifies the collection cost.

**Filtering can help learning, but a change in training-batch accuracy is not proof.** Selection changes which outcomes enter that metric, so it no longer estimates success on the original prompt distribution. Compare unfiltered held-out improvement **per update and per elapsed time** against a run without filtering. Training metrics still diagnose reward contrast, gradients and stability.

## 02 — Entropy is changing

Token entropy measures the spread of next-token probabilities at a given prefix. Read it alongside task performance and response diversity: its direction alone does not identify a problem.

**First confirm entropy is being computed.** Logging entropy does not require an entropy bonus in the loss; zero or constant values may be logging placeholders.

Average entropy over the token positions included in the policy-gradient loss, using the same token mask. Inspect the per-token entropy distribution early to establish a baseline, then follow the trend with consistent masking and averaging.

**Read the pattern.**

- **A — Performance improves as entropy falls.** The policy may be concentrating on useful behavior. Lower entropy alone is not a reason to intervene.
- **B — Entropy falls sharply and progress stalls.** This can indicate **entropy collapse**: the policy becomes increasingly deterministic, restricting exploration. Inspect repeated attempts at the same prompts and check sampling settings.
- **C — Entropy rises while quality falls.** Inspect repetition, incoherence and degraded reasoning, then compare against a control for the suspected setting. The example below follows this investigation.

**Before choosing a fix:** confirm consistent entropy measurement, inspect responses and held-out performance, then investigate changes in the objective, updates or data. The same entropy pattern can have different causes.

### Observed example: rising entropy with falling accuracy

An entropy bonus is intended to discourage premature concentration and preserve exploration. For a minimized loss, it adds a term $-\beta H$ to the RL loss: increasing entropy improves that term regardless of whether the additional token alternatives help solve the task. It can therefore compete with reward-driven learning. [Entropy regularization experiments (§4.1)](https://arxiv.org/html/2505.22617v1#S4.SS1)

Both runs use synchronous GLM-4.7-Flash training on DAPO-Math-17k with an 8K response limit. Evaluation uses the same 1,000 held-out questions, eight responses each and the same 8K limit. The configurations differ only in entropy coefficient and run names.

**1. Locate the deterioration.** With coefficient 0.001, held-out pass@1 falls from 65.81% at update 70 to 29.85% at update 140 while entropy rises sharply. Without the bonus, entropy stays near its initial level and accuracy improves. This is pattern C in the affected run: performance collapses while entropy rises.

**2. Inspect the responses.** In the [affected run's dashboard](https://modal-labs-nan-dev--training-gym-dashboard-fastapi-app.modal.run/training/gravitational-conduit-e90cee961cff), compare earlier rollouts with rollout 139, used for update 140. Here is the unchanged opening of one response to a problem asking for the smallest integer whose digits multiply to $9!$:

> We should find the minimal integer. A standard way: as short a representation of $9!$ as possible, meaning it wants a rounded minimal count of numbers: ergy as partas that must produce $5$. Use algorithm.StatusOK sum factor and Sam...
>

The response contains incoherent text, received reward 0 and used 3,879 tokens, below the 8K cap. This selected example illustrates degraded generation; it does not measure how often it occurs.

**3. Test the hypothesis against the control.** The otherwise matched [run without an entropy bonus](https://modal-labs-nan-dev--training-gym-dashboard-fastapi-app.modal.run/training/snowy-skin-5ba2d9b61bd5) reaches 72.15% held-out pass@1 at update 140 and avoids the same deterioration. This supports omitting the 0.001 bonus in this recipe. It does not establish a universal coefficient or demonstrate recovery of an already degraded policy.

### Choose an intervention, then verify learning

**Rising entropy with worsening performance does not uniquely diagnose an excessive entropy bonus.** Reducing or removing it is one controlled test when it is enabled. Without a bonus, investigate the rest of the training loop:

| Evidence to check | Next action |
| --- | --- |
| Abrupt policy changes: learning rate, probability ratios, clipping and gradients | If updates are excessive, test a lower learning rate or fewer updates per collected batch (section 04). |
| Trainer–rollout disagreement at matching weights and prefixes | Check scoring conventions, inputs and weight loading; repair the identified mismatch (section 03). |
| Changed task mix, unreliable rewards or incorrect advantages | Inspect traces, verifier decisions and batch construction. Fix the data or signal; recheck on a fixed evaluation set (sections 01 and 04). |

For **falling entropy with stalled learning**, check unintended deterministic sampling first. If updates concentrate behavior too quickly, test their size before adding an entropy bonus.

**Top-p masking is another option to test for entropy preservation and exploration.** Training normalizes probabilities over the token set retained during rollout. For that fixed set, the policy-gradient term has no direct gradient to excluded logits. Shared parameters still change, so entropy preservation is not guaranteed. [Sampling-support implementation](https://github.com/radixark/miles/pull/2596)

**Verify:** compare full-vocabulary entropy, response diversity and held-out performance using consistent scoring and averaging. Entropy after sampling filters is a different quantity. Task mix, prefixes and response lengths can move the mean; judge an intervention by learning and response quality, not by reaching a target entropy.

Reported cases in DAPO: [entropy collapse, with a paired accuracy comparison (§3.1, Figure 2)](https://arxiv.org/html/2503.14476v2#S2.F2), and [excessive entropy and response-length growth (§3.3, Figure 4)](https://arxiv.org/html/2503.14476v2#S3.F4).

## 03 — Log-probability mismatch is increasing

Unintended disagreement between rollout and trainer probabilities can distort policy updates and contribute to training instability. Reducing it can stabilize training—even when both engines use the same weights. The verifier's reward can be correct while the probabilities used in the update are inconsistent.

### Identify the comparison

Record the engine and weight version behind each score. A current-trainer versus older-rollout comparison can combine two effects:

| Comparison | How to isolate it | First suspects |
| --- | --- | --- |
| **Engine disagreement** | Compare rollout and independently computed trainer scores using the **generating weights**. | Tokens, prefixes, scoring conventions, precision and MoE routing. |
| **Policy movement** | Within the trainer, score the same tokens using the **generating and current weights**. | Updates per batch, learning rate and async policy lag. |

Policy movement is expected after updates; investigate excessive movement using the update checks in section 04 and lag checks in 08. Synchronous execution does not eliminate engine disagreement. [Policy and engine distinction](https://arxiv.org/html/2602.15763v1#S3.SS2)

### Read the pattern, then check the update

A useful metric is **mean absolute log-probability difference on sampled tokens**,
not KL divergence. Check which tokens enter the average and how responses are
weighted; changing task mix or length can move it. A nonzero gap alone does not
establish unhealthy training; check updates and held-out behavior.

**Before changing the recipe:**

1. **Validate the scores.** Use the same token IDs and full prefixes; align temperature, vocabulary normalization and sampling filters where applicable. Compare independently computed scores—reusing rollout scores on both sides gives zero by construction.
2. **Check the same interval in training.** Inspect probability ratios, clipping/rejection fractions, gradients and held-out performance. Check what each ratio compares: old-versus-current clipping can be zero while trainer–rollout disagreement remains nonzero.

### Published example: routing mismatch and collapse

The R3 paper compares MoE math training with and without **rollout routing replay**, which reuses rollout expert choices during training.

- **What deteriorated:** without routing replay, the run's disagreement grows sharply and validation performance collapses.
- **What changed:** the comparison run enables routing replay, with TIS disabled in both runs.
- **What improved:** disagreement stays lower and learning continues. This supports routing replay as a stability intervention for the reported setup. These are separate training runs, not recovery of the collapsed run. [R3, Figures 5–6 and §5.2](https://arxiv.org/html/2510.11370v2#S5.SS2)

### Match the finding to a remedy

| Confirmed finding | Next action |
| --- | --- |
| Multi-turn tokens or prefixes differ | Use [**token-in/token-out (TITO)**](https://www.lmsys.org/blog/2026-05-13-no-token-left-behind/) to preserve generated token IDs and actual conditioning history. Check policy-loss masks. |
| Rollout filters the vocabulary, and training intends that restricted policy | Use [**top-p masking / sampling-support replay**](https://github.com/radixark/miles/pull/2596) to normalize over the recorded token set. |
| MoE expert choices differ | Test **R3**, then remeasure disagreement and learning. Other numerical differences can remain. |
| Precision or kernel changes increase disagreement | Compare with the previous precision path; check quantization and weight export. |
| Correctly computed scores still have consequential mismatch | Test [**TIS**](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#true-on-policy-rl), which clips correction weights, or [**IcePop**](https://arxiv.org/html/2510.18855v2#S2.SS3.SSS2), which rejects out-of-range contributions. These bounds can introduce bias; neither repairs incorrect inputs. |
| Rollout loads wrong or stuck weights | Repair publication/loading and verify the acknowledged version. For intentional async lag, check staleness (section 08). |

**Verify:** for an input, routing or numerical fix, rescore fixed prefixes at matching weights and check whether disagreement decreases. For importance correction, inspect correction weights and how much signal is clipped or rejected; the raw gap need not shrink. In both cases, verify stable updates and held-out learning. Async training still needs lag control: fixing engine disagreement does not remove policy staleness.

**Dashboard example:** compare `train/train_rollout_logprob_abs_diff` in the matched [R3-off](https://modal-labs-nan-dev--training-gym-dashboard-fastapi-app.modal.run/training/warm-avocet-25e74cb154d2) and [R3-on](https://modal-labs-nan-dev--training-gym-dashboard-fastapi-app.modal.run/training/snowy-skin-5ba2d9b61bd5) GLM runs, then inspect their held-out curves over the same updates. R3 lowers the measured gap; both runs remain stable. This illustrates the measurement check, not a reproduced collapse-and-rescue.

## 04 — Updates are ineffective or unstable

Follow one batch through **reward → advantage → loss mask → gradient → optimizer → rollout weights**.

| What you observe | Start here |
| --- | --- |
| Updates appear absent | Follow the signal and optimizer checks below. |
| Gradients spike or become nonfinite | Inspect the affected update below. |
| Updates happen normally, but learning stalls | Recheck data/rewards (01), exploration (02) and evaluation (05). An operating optimizer does not guarantee useful learning. |

### Follow the signal into an update

**Check one batch.** Align responses, rewards and prompt groups, then inspect the training tensors.

| Check | What should hold |
| --- | --- |
| **Advantages** | Each value is attached to the correct response or token and computed from the intended rewards and baseline. |
| **Loss mask** | Apply policy-gradient loss only to policy-generated tokens selected by the loss mask. Prompt, tool-output and padding targets are excluded. |
| **Token alignment** | Each sampled token is scored against its actual preceding context. Check the label shift and apply the mask to target-token positions. |
| **Packing boundaries** | Independent examples cannot attend to one another. Cross-example next-token targets are excluded from the loss. |
| **Loss weighting** | Token-level or sample-level averaging matches the objective; padding and gradient accumulation do not change the intended weights. |

Loss masking controls which policy-generated tokens receive policy-gradient signal; it does not remove context. Including tool targets trains on externally supplied text. Excluding all policy-generated tokens removes the reward-driven policy gradient, while other losses or the optimizer may still change weights.

**Check execution.** Inspect gradients, the actual learning rate, skipped optimizer steps and parameter changes; then confirm rollout loads updated weights. Near-zero scalar policy loss can coexist with nonzero gradients.

**With a critic:** inspect value predictions and return targets. Bootstrap across a collection cutoff if the task continues, but not past termination. An explicit task budget may define termination. [Target construction](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

### Inspect a spiking or nonfinite update

A **pre-clipping gradient norm** can exceed the clipping threshold even when clipping works. It does not measure the parameter change. [Gradient clipping](https://docs.pytorch.org/docs/2.8/generated/torch.nn.utils.clip_grad_norm_.html)

Inspect that batch for outliers in advantages, response lengths and probability ratios; then check whether loss normalization lets a few contributions dominate the update. **Gradient clipping** rescales gradients. **Policy-ratio clipping** modifies the surrogate objective. **Rejection** excludes contributions: for example, [IcePop](https://arxiv.org/html/2510.18855v2#S2.SS3.SSS2) masks token-level policy-gradient contributions whose trainer-to-rollout probability ratio falls outside its bounds.

### Match the finding to a remedy

| Confirmed finding | Action |
| --- | --- |
| Wrong masks or sample alignment | Repair them; inspect the corrected batch again. |
| Most contributions rejected, or regularization suppresses learning | Check ratios and rejection bounds (03), or reference KL and its coefficient. Scalar loss magnitudes alone do not establish dominance. |
| Repeated skipped updates, NaNs or infinities | Find why steps were skipped. For nonfinite values, locate the first failing operation and its inputs before changing precision or loss scaling. |
| Excessive policy changes | Test a lower learning rate or fewer updates per batch. |
| Isolated finite spike with stable subsequent behavior | Inspect the batch; a recipe change may be unnecessary. |

**Verify:** intended samples contribute, optimizer steps succeed, rollout receives updated weights, and held-out learning remains stable. A smaller gradient or loss alone is insufficient.

## 05 — Training improves but evaluation does not

For the detailed dataset and response audit, follow
[train–eval investigation](train-eval-gap.md).

Training reward rises while held-out performance stalls or falls. First establish what the gap measures before diagnosing a failure to generalize.

**Check next, in order.**

1. **Is evaluation valid?** Confirm the evaluated weight version, completed samples and execution errors. Align results to that version's training update, not their arrival time.
2. **What do the metrics measure?** Check prompt populations, verifier, sampling settings, token/turn/tool budgets and metric definitions. Pass@1 and pass@k answer different questions. Filtering or async arrival order can change the training task mix, raising batch reward without better performance on a fixed distribution.
3. **If the gap persists, where does learning fail to transfer?** Read high-reward training responses and failed evaluation responses. Check for verifier shortcuts, missing task coverage and regressions within particular task types. Inspect split overlap and distribution differences between train vs. eval datasets.

Training and evaluation settings may differ intentionally. State those differences; use matched conditions when investigating their effect on the gap.

**Match the finding to a remedy.**

| Confirmed finding | Next action |
| --- | --- |
| Wrong weights, incomplete evaluation or execution errors | Repair evaluation and rerun the affected measurement. |
| Different budgets, sampling settings or reward definitions | Run a diagnostic comparison under matched conditions; retain and document intentional differences in the target evaluation. |
| High reward without genuine task success | Strengthen the verifier or reward specification. Test known correct responses and known failure cases. |
| Improvement concentrated on training examples or narrow task types | Evaluate fixed training and held-out subsets under the same protocol. Identify coverage gaps before changing the training mixture. |
| Some capabilities improve while others regress | Track them separately. Test a targeted task mixture or reference-policy constraint, then measure both gains and regressions. |

**Verify:** keep the held-out protocol fixed across comparisons and inspect per-task results. Repeat uncertain measurements before reacting to small changes; differences comparable to measurement variation do not establish improvement or deterioration. [Evaluation uncertainty](https://arxiv.org/abs/2108.13264)

## 06 — The run hangs, crashes or runs out of memory

**Locate the first failure.** Check the last completed operation and earliest relevant error across workers. Collective/HTTP timeouts may follow a peer failure. For hangs, inspect progress in tools, evaluation, collectives and weight publication.

For **out-of-memory (OOM)** errors, identify the failing **rank, phase and allocation**.

**Read the pattern.** Compare similar workloads at the same phase boundary.

- **A — Stable post-phase memory:** memory returns to a similar baseline after each phase. If an OOM occurs at a phase peak, reduce or distribute that phase's memory demand using the table below.
- **B — Rising post-phase memory:** more memory remains between phases. Inspect retained tensors, buffers and caches; release or bound what should not accumulate. Growth alone does not prove a leak or guarantee an eventual OOM.

### Match the memory source to the knobs

| Memory source | Candidate changes |
| --- | --- |
| **Training activations** | Smaller microbatch / per-device token budget; [**activation recomputation**](https://docs.nvidia.com/nemo/megatron-bridge/0.3.1/training/activation-recomputation.html). |
| **One long training sequence** | [**Context parallelism (CP)**](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) for sequence activations. |
| **Model weights** | Tensor parallelism (TP), pipeline parallelism (PP), or parameter sharding. |
| **Gradients / optimizer states** | State sharding; CPU offload. |
| **MoE expert weights** | Expert parallelism (EP). |
| **Logits / log-probabilities / entropy** | Smaller token chunks; supported [**fused computation**](https://github.com/linkedin/Liger-Kernel#fused-scaled-cross-entropy) that avoids full token-by-vocabulary intermediates. |
| **Rollout prefill** | Smaller prefill chunks; reserve activation/workspace memory outside the cache pool. |
| **Rollout decoding** | Lower request concurrency; inspect key/value (KV) cache capacity and retained session state. |
| **Weight publication / colocated transitions** | Reduce duplicate weights and transfer buffers; release or offload inactive state. |
| **Host RAM** | Bound trajectory queues and tool concurrency; clean up retained processes; provision memory. |

**Parallelism:** CP targets sequence activations. Megatron-style **sequence parallelism (SP)** partitions selected activations and requires TP > 1. Ordinary **data parallelism (DP)** replicates model state; more replicas do not automatically reduce its per-device footprint. Check model/backend compatibility. [Parallelism and sharding](https://docs.nvidia.com/nemo/megatron-bridge/0.3.1/parallelisms.html)

**Rollout memory:** increasing cache reservation leaves less room for activations and CUDA graphs. Prefill OOM and cache exhaustion need different adjustments. [Inference tuning](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hyperparameter_tuning.mdx)

**Memory accounting:** compare equivalent workloads at the same phase boundary. Allocated memory tracks tensor storage; reserved memory also includes unused allocator blocks. High reservation alone does not prove a leak; a process exit alone does not prove OOM. [PyTorch memory accounting](https://docs.pytorch.org/docs/2.8/notes/cuda.html#memory-management)

**Preserve the experiment:** use gradient accumulation to retain the intended effective batch when reducing microbatch size. Preserve loss normalization. Shortening responses or dropping long samples changes the learning problem.

**Verify:** repeat the failing workload, including long sequences and phase transitions; then measure throughput (07). Bound infrastructure retries. Repeated failure at the same operation needs diagnosis; longer timeouts cannot repair a crashed worker.

## 07 — The run is too slow

**Establish a reference.** There is no universal step time for a model size. Architecture, precision, sequence lengths, batch size, hardware, parallelism and environment workload all affect it.

- **With a comparable run or benchmark:** compare steady-state timings at similar token counts and concurrency. Investigate unexplained regressions; a known-working recipe is not necessarily efficient.
- **Without one:** profile a short run after warmup using representative inputs. Record phase times, token counts and throughput across several updates. Use this as the baseline for controlled changes.

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

**If confirmed, try the matching optimization.**

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

**Within generation, check prefix-cache reuse.** miles' **`rollout/prefix_cache_hit_rate`** is summed cached prompt tokens divided by summed recorded prompt tokens. This is a token-weighted fraction, not the fraction of requests with a cache hit. Compare it with prompt lengths and prefill time; missing metadata can produce zero.

**Interpret and act.**

- **Consistently high:** if the run remains slow, inspect decoding, tools and other phases.
- **Persistently low despite shared history:** check exact prefixes, cache eviction and routing/session affinity. Restore reuse where expected, then remeasure prefill and full-loop time.
- **Repeated drops:** align them with weight updates, cache resets, routing and input-length changes. Measure expected refill costs; investigate unexplained drops.

High expected cache reuse assumes a warm workload with shared history. New prefixes or large tool outputs can lower reuse. Cached states must be valid for the active weights. Reuse saves prefill work, not decoding or tool execution. [Prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) · [Routing and load balance](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/policies/cache_aware.rs)

**Verify:** hold task mix, effective batch size and generation/tool budgets fixed. Compare usable trajectories or updates per second, GPU-hours per update and trajectory tail latency; report extra CPU/sandbox resources. Preserve timeout and failure handling. Higher utilization or cache hits alone do not establish a speedup.

## 08 — Async or more GPUs do not help

Check **usable training throughput, policy lag and held-out learning** together. Faster generation helps only if training can use the resulting experience effectively.

### Locate the waiting

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

**More GPUs, little gain:** remeasure the rollout/trainer allocation, per-GPU batch size, communication and CPU/tool limits (07). Retune parallelism for the new allocation; more rollout GPUs cannot accelerate a saturated trainer.

### Locate the policy lag

Record which trainer update produced the rollout weights and when the experience enters training. **Elapsed age, optimizer-update lag and publication count are different quantities.** One publication can span several optimizer updates.

Compare lag distributions for **admitted and discarded** experience separately; check whether logged aggregates include rejected samples. Version lag is a proxy for policy change, not a universal stability threshold.

| Where lag accumulates | Check / next action |
| --- | --- |
| **Before generation** | Compare trainer, published and actually loaded versions across engines. Fix stuck or wrong weights; shorten publication delays where practical. |
| **During generation / reward computation** | Inspect long trajectories, tool waits and verifier latency. Supported pause/resume or partial-rollout scheduling can reduce barriers, but unfinished work still needs correct continuation. |
| **After completion** | Inspect buffer residence time and excess production. Reduce prefetch/concurrency or speed up consumption; test an explicit staleness limit. |

If weights change within a trajectory, preserve original token IDs, prefixes, behavior log-probabilities and version provenance. Resuming under newer weights does not make earlier tokens fresh; rescoring with those weights cannot recover the original behavior probabilities. Check cache and continuation semantics. [Interruptible generation](https://arxiv.org/html/2505.24298v5#S4.SS1) · [Partial rollouts](https://arxiv.org/html/2509.18521v1#S2.SS2)

### If throughput improves but learning deteriorates

1. **Check off-policy updates.** Inspect probability ratios, clipping/rejection fractions and retained training signal by lag. Confirm which policies each ratio compares and that the loss handles stale behavior data. Separate same-version engine disagreement (03) from policy movement.
2. **Test the lag–throughput trade-off.** Reduce publication or queue delays, or tighten the staleness bound. Track lost throughput and discarded work; a tighter bound can starve training. Importance correction does not make arbitrary lag harmless. [Staleness and correction ablations](https://arxiv.org/html/2505.24298v5#S7.SS4)
3. **Check the task mix.** Track submission, completion and admission by task type; compare lengths and rewards for completed and admitted samples. Timeouts, cancellation and stale rejection can underrepresent slow tasks. Count wasted tokens and retries; higher accepted-batch reward may reflect selection.

**Verify:** compare held-out performance per update and elapsed time under the same task and evaluation budgets. Match total resources for sync/async comparisons; report GPU-hours and extra CPU/sandbox/eval capacity. Evaluate fixed weights and label results by their version/update; record completion lag separately. Shared-engine eval can pause production; dedicated eval still has export and resource costs.
