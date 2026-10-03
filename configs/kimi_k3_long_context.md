# Kimi-K3 at 64k

`configs/kimi_k3_long_context.py` builds a long-context configuration of the
K3 LoRA recipe on `main`, including the startup and observability fixes from
[#636](https://github.com/modal-projects/modal-dojo/pull/636). It retains the
64-B300 colocated topology, pinned image, adapter-only synchronization, persistent
kernel caches and base checkpoint volume. Training remains synchronous.

```python
from configs.kimi_k3_long_context import build_recipe
from modal_dojo import Kimi_K3, TrainConfig

config = TrainConfig(
    model=Kimi_K3(),
    dataset=your_dataset,
    recipe=build_recipe(num_rollout=2, rm_type="deepscaler"),
)
```

| Setting | Default |
| --- | --- |
| Total inference context, including prompt and response | 65,536 tokens |
| Rollout and evaluation response ceilings | 65,536 tokens, bounded by remaining context |
| Training and logprob microbatch budget per GPU | 32,768 tokens at CP2 |
| Training sequence and position limits | 65,536 tokens |
| Trainer parallelism | TP4 × PP8 × CP2, DP1, EP8 |
| Training / inference precision | BF16 rank-32 LoRA / native MXFP4 |
| Inference engines | Four TP16 engines |
| Active generation requests | One per engine, four cluster-wide |
| Token pool per engine | 131,072 tokens |
| Chunked prefill | 4,096 tokens |
| Update batch | Two prompts, four completions each |
| Colocated handoff | Offload the active model before restoring the other |
| Host RAM per node | 2.5 TiB requested, 3 TiB limit |

The long-context configuration uses `colocate_memory_peak_device="cpu"`.
The earlier full-model capacity run completed its 128k training update and
checkpoint save, then ran out of GPU memory while restoring inference weights.
The GPU-overlap handoff needed approximately 169 GiB of remaining trainer memory
plus 140 GiB of inference weights on a 268-GiB device. Offloading the trainer
first avoids that overlap. A one-B300 allocation probe completed two handoffs
with a 218.9-GiB peak, but does not establish full-model training correctness.

Both models' host backups can coexist during this transition. Their estimated
weight-only footprint is 1,849 GiB per eight-GPU node, before optimizer state,
processes and other buffers. The larger RAM request and limit provide headroom
on the AWS B300 nodes. Full-model peak host memory still needs measurement.
This larger RAM request can constrain scheduling. DP8, QAT and one TP8
inference engine per node are not part of this configuration.

The default text generator bounds each request to the remaining context, in
both training and evaluation. With a 2,048-token prompt, the maximum response
is 63,487 tokens, reserving the pinned scheduler's one-token boundary margin.
It delegates token IDs, logprobs, LoRA selection and weight-version bookkeeping
to the pinned Miles generator. Aborted samples retain their generated prefix;
resuming does not reset the context budget.

`build_recipe(context_length=131072)` explicitly selects 128k instead, with a
65,536-token per-GPU packing budget and a 262,144-token inference pool.
The training budget is for packing microbatches, not a separate context limit
or a memory guarantee.
At CP2, the default 32,768 tokens per GPU permits approximately one 64k sequence
per microbatch. A smaller packing budget cannot shrink a single long sequence.
Full uniform recomputation with `recompute_num_layers=1` already recomputes
every transformer layer. Increasing that number changes the checkpoint segment
size; it does not enable recomputation on more layers.

## H200 candidate

`build_recipe(gpu_type="H200")` selects eight nodes with eight H200s each.
This candidate is limited to a 64k context and one request per TP16 engine.
It uses full uniform recomputation in groups of three layers, 100% optimizer
CPU offload, FlashInfer MLA decoding, and the same offload-first handoff.
Host memory is 1,792 GiB requested with a 1,920-GiB limit: the estimated
simultaneous weight backups total approximately 1,624 GiB per node. The B300
profile's 2.5-TiB request exceeds the 2-TiB RAM capacity of
[AWS H200 hosts](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html).
The reduced request depends on compact inference weights; full-run host peaks
still need measurement.
Training remains TP4 × PP8 × CP2 with EP8; inference remains four TP16 engines.

The H200 image adds a shape-specific Marlin allocation patch: K3's 192-wide
expert shard uses the supported 64-element alignment instead of padding to
256. Other shapes and backends retain their existing behavior. The original
MXFP4 checkpoint is reused; no NVFP4 conversion is needed. In single-H200
kernel tests, the compact layout used 25% less expert-weight storage and gave
identical outputs to the padded layout at 1, 16 and 128 tokens, including
nonzero LoRA hook updates. The NVFP4 kernel also ran, but used approximately
6% more storage than compact MXFP4 in that test.

Across 92 layers and 896 experts, removing this padding saves approximately
28.06 GiB per inference GPU. Subtracting that from the measured 140.15-GiB
B300 weight allocation estimates 112.09 GiB of base weights. Routed-expert
rank-32 adapters with shared outer factors add approximately 2.89 GiB, and
the 131,072-token MLA cache adds 3.375 GiB. Other adapters, KDA state, graph
buffers, communication and temporary allocations are additional. These are
estimates, not a measured full-model H200 peak. The candidate sets SGLang's
static memory fraction to 0.95; the inherited 0.75 budget is below even the
estimated base-weight footprint on the measured 139.8-GiB H200.

Run the local configuration check with:

```bash
uv run -m scripts.validate_kimi_k3_long_context --gpu-type H200
```

The patched allocator and post-load processing also passed a single-H200
check with two in-place reloads, unchanged parameter identities and storage
addresses, and finite outputs at all three batch sizes. This tests the real
MXFP4 method with synthetic tensors, not a complete checkpoint load.

A two-node H200 TP16 server using a four-layer, 64-expert checkpoint also
passed health, 64,512-token prefill plus 1,022-token decoding, and KV-cache /
CUDA-graph release and resume. Cold startup took 605 seconds; the long request
took 44.4 seconds including compilation. This pruned checkpoint produces
incoherent text and is only an execution-path check, not a quality benchmark.
The pinned-runtime preflight verified the H200 arguments, tokenizer, dataset,
reward edge cases and 64k mask assembly; 50 focused local tests passed.

The first full-model H200 run,
[adagio-hull](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/adagio-hull-e43716fddb12),
loaded the compact inference weights at **111.50 GiB per rank**, but failed
before its first rollout. SGLang cloned and retained every incoming adapter
bucket on GPU until the complete adapter arrived. A 168-MiB clone failed with
only 44 MiB free. This was adapter synchronization, before long-context
generation or training.

The H200 image now also patches that staging copy to an independent, blocking
CPU copy. Blocking is necessary because the sender may reuse its CUDA IPC
bucket as soon as the RPC returns. SGLang's existing whole-adapter validation,
checksum checks and abort handling remain in place; its loader slices the
host tensors into the existing TP-sharded GPU adapter pool. The B300 profile
does not apply this patch.

The H200 image also fixes generation health checks with the single adapter
slot. The pinned `/health_generate` otherwise requests the base model, which
would evict the installed adapter; `lora_no_cpu_backup` correctly refuses that
eviction. The health request now uses the sole registered policy, preserving
actual generation health checks without allocating a second GPU adapter pool.
Its request lease is released on completion or timeout. An ambiguous registry
with multiple adapters returns unhealthy instead of selecting an arbitrary one.

A single-H200 regression reproduced the original accumulating-stash OOM,
then staged 1,344 MiB with zero additional GPU allocation using the patch.
Reusing the source bucket did not change staged values. The pinned memory
saver released its 2-GiB CPU backup when weights resumed on GPU. Thus, during
adapter synchronization, host memory holds the trainer backup and adapter
staging, rather than both frozen-model backups plus staging. Approximately
46 GiB of raw routed-expert factors per rank, plus up to 31 GiB for temporary
normalization, adds about 616 GiB per eight-rank node to the roughly 732-GiB
trainer backup. Runtime overhead and actual full-model peaks still need
measurement.

The run used the deployment-compatible launcher at `1e4c5bdd2` with the same
recipe, image patches and flags. Current `main` requires the shared Training
Gym-to-Dojo configuration/volume migration; that migration was not performed
as part of this model fix. In an environment configured for current `main`, run:

```bash
uv run -m scripts.validate_kimi_k3_long_context --gpu-type H200 --rollouts 2 --launch
```

The full-model 64k trainer peak, checkpoint save, model quality and subsequent
rollout with an updated adapter remain unverified.

## Agent trajectories

Pass `custom_generate_function=your_agent` and `custom_rm_function=your_reward`
to replace the default single-turn generator. Your agent must enforce the total
sequence budget across the initial prompt, assistant turns, tool observations
and message delimiters. Use
`remaining_response_tokens(context_length, len(token_ids), response_ceiling)`
before each generation and stop when it returns zero. Bound tool observations
before appending them. Preserve generated token IDs and logprobs across turns,
and mask observations and environment-inserted delimiters out of the loss.
Apply the same contract to evaluation.

`rollout_max_prompt_len=None` disables the initial dataset length filter; it
does not exempt prompts or growing histories from the inference context limit.
The default generator rejects prompts that leave no space for a response.

Increasing `concurrency_per_engine` adjusts the token pool, KDA state slots,
client concurrency and decode graph sizes together. It requires a new memory
capacity check. The base recipe's router connection pool keeps headroom for
health checks alongside generation requests.

## Validation

```bash
# Local configuration check; no GPU allocation:
uv run -m scripts.validate_kimi_k3_long_context

# Two near-64k full-model capacity updates (64 B300s):
uv run -m scripts.validate_kimi_k3_long_context --mode capacity --launch

# Explicit 128k capacity test:
uv run -m scripts.validate_kimi_k3_long_context --context-length 131072 --rollouts 2 --launch

# Response-heavy stress test, forcing generation despite EOS:
uv run -m scripts.validate_kimi_k3_long_context --mode decode --decode-tokens 57344 --rollouts 1 --launch
```

Set `MODAL_ENVIRONMENT` to the environment containing the base K3 checkpoint.
Capacity mode inserts a large masked synthetic observation between two real
generations. Every sample must reach within 1,024 tokens of the context ceiling.
This checks sequence assembly, long prefill and training memory; it does not
establish 60k-token assistant generation or customer-task performance. Decode mode
is a separate stress test, and its forced generation length is not a production
sampling setting. `--mode math` runs ordinary task rollouts with the context
budget enforced; natural short completions do not validate long sequences.

The current base recipe completed the
[10-step K3 validation](https://modal-labs-helena-dev--training-gym-dashboard-fastapi-app.modal.run/training/legacy-mamba-e18bdc58b9cc)
with responses capped at 4k. That run validates the base training cycle, not
64k capacity or reward improvement.

The [two-step 64k validation](https://modal-labs-helena-dev--dojo-dashboard-fastapi-app.modal.run/training/shiny-fall-0bf32fd3aaf0)
was submitted on September 30, 2026 using these settings. It was subsequently
cancelled, with no training steps or rewards recorded. The cached
checkpoint was found and conversion was skipped. This run predates the rebase
onto current `main`; it does not validate changes subsequently merged there.

Before that launch, 29 local configuration and generation-boundary tests passed.
A preflight inside the pinned image verified the 64k argument values, real K3
tokenizer and dataset formatting, token/logprob/mask alignment, and rewards for
correct, incorrect, malformed and missing answers. Mocked generation assembled
a 64,530-token sequence. This establishes harness behavior, not full-model
memory fit or generation quality. The offload-first configuration still needs
a complete full-model two-step run and measured host-memory peaks.

Earlier checks on the same pinned runtime verified argument parsing, token
alignment, masked observations, and a 130,048-token prefill on a four-layer
model. Those checks do not establish full-model training capacity. Historical
full-model attempts used the previous base recipe:
[wise-bridge](https://modal-labs-helena-dev--training-gym-dashboard-fastapi-app.modal.run/training/wise-bridge-7312f2a0aca8)
and [basic-baryon](https://modal-labs-helena-dev--training-gym-dashboard-fastapi-app.modal.run/training/basic-baryon-1d409d89645d).
Basic-baryon trained eight sequences of 130,973–131,070 tokens with finite
gradients and saved its checkpoint, then failed during the inference-weight
restore before its second rollout.

Before claiming validated long-context support, check actual sequence lengths,
finite gradients, correct rewards and loss masks, peak memory, substep timings,
checkpoint saves, and a subsequent rollout using the updated adapter.
