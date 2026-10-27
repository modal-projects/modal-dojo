---
title: Reflection Beam 1.0 LoRA 256k
description: Experimental 256k Beam LoRA recipe and measured throughput validation.
---

Beam support is experimental, based on [Miles PR 4149](https://github.com/radixark/miles/pull/4149).
The maintained recipe is `Beam_1_0_501B_A23B_LoRA_256k_Recipe`. Short-context
and full-parameter recipes are not exposed. Full-window generation and a trainer-only optimizer-step replay have passed;
fresh end-to-end confirmation is in progress. Rollout concurrency eight is the measured
default; end-to-end throughput remains unvalidated.

```python
from modal_dojo import Beam_1_0_501B_A23B_LoRA, Beam_1_0_501B_A23B_LoRA_256k_Recipe, TrainConfig
from modal_dojo.common.dataset import HuggingFaceDataset

config = TrainConfig(
    model=Beam_1_0_501B_A23B_LoRA(),
    dataset=HuggingFaceDataset(
        "zhuzilin/dapo-math-17k", hf_split="train[:32]",
        input_column="prompt", output_column="label", input_format="messages",
    ),
    recipe=Beam_1_0_501B_A23B_LoRA_256k_Recipe(
        num_rollout=1, save=None, save_interval=None, max_retries=0,
    ),
)
# Allocates 32 H200 GPUs:
# run = config.launch()
```

The recipe uses 229,376 prompt tokens plus up to 32,768 response tokens, for a
262,144-token total window. Training uses TP8/CP2/EP16/expert-TP1 across 16 H200s,
with a separate 16-H200 rollout engine. Rank-16 LoRA, the upstream optimizer,
and 32 prompts × eight responses (global batch 256) are preserved. The
capacity proof uses only one prompt × eight responses and is not a measurement
of the maintained batch-256 recipe.

FlashAttention avoids quadratic score storage. Fixed `bshd` microbatches are
required; dynamic packing is disabled. Context parallelism uses `all_gather`,
because the pinned Transformer Engine rejects sliding-window attention with
`p2p`. A recipe-scoped image patch propagates `cp_comm_type` through the Miles
bridge provider, which otherwise silently drops this flag. It normalizes the CLI
singleton list to a scalar: Megatron interprets lists as per-layer settings and
requires 52 entries for Beam. The patch is guarded
against upstream source changes and has pinned source fixtures.

The recipe pins Miles `819cfd23`, SGLang `9ac1c65b`, the v0.1.2 image digest,
and the HF checkpoint revision. Dependency upgrades are disabled when installing
these forks to preserve the image's Torch and compiled extensions. Dojo's async
substep timing remains enabled.

## Validation and throughput

Both capacity proofs independently generated eight responses with exactly
229,376 input tokens and 32,768 output tokens each (262,144 generated tokens).

| Rollout concurrency | Run | Rollout time | Output tokens/sec |
| --- | --- | --- | --- |
| 4 | `staff-hatchback-b91f94bef820` | 3,271.43 s | 80.13 |
| 8 | `rectilinear-bullet-6180c878f4a7` | 1,842.32 s | 142.29 |

Concurrency eight improved this workload's rollout throughput by **1.78×**.
The latter time also includes saving the generated batch for trainer debugging.
These are rollout-only measurements, not end-to-end training throughput.
The shared, synthetically padded prompt produced a high prefix-cache hit rate;
distinct repository contexts may perform differently. This is not SWE-bench
validation or an agentic software-engineering workload.

Full-model training remains experimental. Early proofs exposed missing
context-parallel configuration and excessive expert activation buffers. Recipe-scoped patches reuse
expert LoRA buffers and bound expert activation temporaries while preserving
Beam's unfused BF16 arithmetic. The allocator uses expandable segments.
The LoRA changes passed H200 output/gradient and checkpoint-recomputation checks.
For the activation change, H200 tests matched outputs and input gradients
exactly; routing-weight gradients matched at `rtol=1e-6, atol=1e-5`.
A representative activation forward/backward test reduced peak memory from
13.00 GiB to 5.41 GiB. Trainer replay `constant-take-9b17c555b6e0` then completed a valid optimizer
step on all eight saved full-window samples and finished `SUCCEEDED`: 136.28 s
for log probabilities, 394.16 s for training, finite loss -0.00057016 and gradient
norm 0.0160357. GPU 7 emitted recoverable allocator-pressure warnings, so memory
headroom is tight. This is a trainer-only replay, not a fresh end-to-end run.
A two-batch confirmation with fresh generation and adapter transfer is pending.

Attention-only preflight on two H200s passed for both sliding-window and full
attention in the pinned runtime. At 1,024 tokens, FlashAttention CP1 and CP2
forward/backward matched unfused CP1 within BF16 tolerances (output absolute
0.005/relative 0.03; gradient absolute 0.001/relative 0.05). CP2 also completed
32,768- and 262,144-token forward/backward with finite outputs and gradients.
The probe uses one TP-local attention slice (six query heads, one KV head), not
all model layers, adapters, optimizer states or SGLang. Its memory and timing
are therefore not full-model throughput measurements. Probe app:
[beam-256k-attention-proof](https://modal.com/apps/modal-labs/main/ap-3PQBGYfT2nzHql2dszAtGp).


Rollout validation must consume the entire batch (32 prompts × eight responses),
letting every sample reach EOS or the 32,768-token response limit. Record natural
termination versus length/context truncation and actual generated token counts.
Do not use shortened generation, partial-rollout training, copied responses or
an early-exit flag as evidence of full-window throughput. The response limit is
a hard cap, not a promise that every sample reaches EOS before it.

For an explicit full-window throughput stress test, enable both hooks:

```python
from modal_dojo.frameworks.miles.benchmarks.beam import (
    audit_full_batch,
    force_full_window,
)

recipe = Beam_1_0_501B_A23B_LoRA_256k_Recipe(
    num_rollout=1, save=None, save_interval=None, max_retries=0,
    custom_generate_function=force_full_window,
    custom_rollout_log_function=audit_full_batch,
)
```

This inserts tokenizer-verified synthetic padding inside each user message to
reach exactly 229,376 input tokens. Generation suppresses EOS and stop sequences,
requests exactly 32,768 new tokens, then asserts actual response-token and
log-probability counts. The batch audit rejects missing or repeated sample indices
and checks every response, not just dashboard previews. With the unchanged batch,
all 256 responses must be independently generated (8,388,608 output tokens).
Any short or aborted sample fails the benchmark. Forced continuations and padded
prompts are synthetic workload measurements, not evidence of reasoning quality or
learning. Normal presets and evaluation retain their usual EOS behavior.


The engine's internal context allocation is 262,146 slots: the pinned SGLang
worker and scheduler each reserve one. Logical input+output and training stay at
262,144 tokens. Without those two extra internal slots, SGLang silently clips a
full-window response by two tokens; the benchmark's observed-length assertion
would reject it.
