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

Do not enable dynamic token packing or replace the unfused `bshd` attention path:
Beam requires its custom attention semantics. Expert LoRA requires expert TP1.
Dojo's timing hooks are preserved, including async executor timing and rollout
wait time; generation and training overlap, so their durations are not additive.

For throughput tuning, first prove one healthy step and inspect responses, reward
and per-stage timing. Sweep rollout concurrency (64, 128, 256) one value at a time
with all training semantics fixed. Confirm the best candidate against the baseline
using two three-step replicas per recipe before promoting a throughput winner.
Then validate a longer smoke run for output quality and learning. There is no
measured speedup or promoted throughput winner yet.
