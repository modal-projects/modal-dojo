---
title: Reflection Beam 1.0
description: Experimental Miles presets and throughput validation for Beam.
---

Beam support is experimental, based on [Miles PR 4149](https://github.com/radixark/miles/pull/4149).
The presets have passed local configuration tests, but have not been GPU-validated
or throughput-benchmarked on Modal. The upstream PR remains a draft.

`Beam_1_0_501B_A23B_Recipe` uses 96 H200 GPUs: 64 trainers
(TP8, EP8, expert TP8) and two 16-GPU rollout engines. The separate
`Beam_1_0_501B_A23B_LoRA_Recipe` uses 32 H200 GPUs: 16 trainers
(TP8, EP16, expert TP1) and one 16-GPU rollout engine. LoRA changes which
parameters learn and is not an equivalent throughput comparison.

```python
from modal_dojo import Beam_1_0_501B_A23B, Beam_1_0_501B_A23B_Recipe, TrainConfig
from modal_dojo.common.dataset import HuggingFaceDataset

config = TrainConfig(
    model=Beam_1_0_501B_A23B(),
    dataset=HuggingFaceDataset(
        "zhuzilin/dapo-math-17k",
        hf_split="train[:32]",
        input_column="prompt",
        output_column="label",
        input_format="messages",
    ),
    recipe=Beam_1_0_501B_A23B_Recipe(
        num_rollout=1, save=None, save_interval=None, max_retries=0,
    ),
)
# Allocates 96 H200 GPUs; run only with an approved cluster budget:
# run = config.train()
```

Both presets preserve upstream async GRPO with 32 prompts, eight responses per
prompt, a batch of 256, and a 5,120-token window (1,024 prompt + 4,096 response).
The reward retains upstream boxed-integer grading and the soft length penalty.
AIME evaluation prompts need the same answer-format instruction as DAPO.

The recipes pin Miles `819cfd23`, SGLang `9ac1c65b`, the upstream v0.1.2 image
digest and HF revision. SGLang is installed without dependency upgrades to keep
Torch and the image's prebuilt extensions compatible. Bridge loads the BF16 HF
checkpoint directly. Full training streams optimizer state to node-local storage
with a 1 TiB reservation per training worker; LoRA keeps its optimizer resident.

The baseline presets retain the upstream unfused `bshd` attention path. The
experimental 256k preset below uses FlashAttention through the same Beam attention
wrapper. Dynamic token packing remains disabled. Expert LoRA requires expert TP1.
Dojo's timing hooks are preserved, including async executor timing and rollout
wait time; generation and training overlap, so their durations are not additive.

For throughput tuning, first prove one healthy step and inspect responses, reward
and per-stage timing. Sweep rollout concurrency (64, 128, 256) one value at a time
with all training semantics fixed. Confirm the best candidate against the baseline
using two three-step replicas per recipe before promoting a throughput winner.
Then validate a longer smoke run for output quality and learning. There is no
measured speedup or promoted throughput winner yet.


## Experimental 256k LoRA

`Beam_1_0_501B_A23B_LoRA_256k_Recipe` keeps the 32-H200 layout and upstream
optimizer and sample count, with the requested 32,768-token response budget.
It allows up to 229,376 prompt tokens, for 262,144 total. It is explicitly selected rather than replacing
the shorter default:

```python
from modal_dojo import Beam_1_0_501B_A23B_LoRA_256k_Recipe

recipe = Beam_1_0_501B_A23B_LoRA_256k_Recipe(
    num_rollout=1, save=None, save_interval=None, max_retries=0,
)
```

The training layout is TP8/CP2/EP16/expert-TP1 on 16 H200s, with a separate
16-GPU rollout engine. FlashAttention avoids materializing quadratic attention
scores. Context parallelism uses `all_gather`: the pinned Transformer Engine
rejects sliding-window attention with `p2p`, and all-to-all cannot shard the one
KV head per TP rank across CP2. Rollout concurrency is four and prefill chunks
are 4,096 tokens, limiting the memory consumed by long requests without changing
the training batch.

The checkpoint advertises 1,048,576 positions, so 256k does not require extending
its configured positional range. That is not evidence of training capacity or
quality at 256k. Full-model long-context training, rollout memory and routing replay
still require GPU validation. A short DAPO prompt with a larger configured limit
does not establish 256k throughput. The original short-context proof was cancelled
during download and produced no training measurements.


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
