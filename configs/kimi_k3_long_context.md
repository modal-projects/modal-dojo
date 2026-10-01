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
was submitted on September 30, 2026 using these settings. As of October 1 it
remains queued for B300 workers, with no training steps completed. The cached
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
