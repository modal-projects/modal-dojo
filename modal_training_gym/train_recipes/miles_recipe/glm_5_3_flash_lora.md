# GLM-5.3-Flash bridge LoRA on Miles

`GLM_5_3_Flash_LoRA` selects `GLM_5_3_Flash_LoRA_Recipe`, an experimental
text GRPO preset based on [Miles PR #3098](https://github.com/radixark/miles/pull/3098).
The original `GLM_5_3_Flash_Recipe` remains the 32-GPU full-parameter preset.

The LoRA preset uses **3 nodes × 8 B300 GPUs**, TP8/EP24/PP1/ETP1, with one
colocated TP8/EP8 rollout engine per node. This preserves the PR's 24-H200
parallel layout; full-model training on Modal has not been validated.
Its DAPO settings follow the PR's reported full-model experiment: 15 prompts
× 8 samples, global batch 120, response limit 7168, temperature 1, routing
replay, and learning rate `1e-5`. Fixed microbatches and full recompute match
the upstream launcher. The default is one rollout.

Rank-16, alpha-32 adapters cover KDA attention and gate projections, DSA
projections, dense MLPs, shared experts, and routed experts. Each routed
expert has its own adapter factors. The sparse indexer, KDA convolutions,
norms, and mHC parameters remain frozen, following upstream's target list.
Changing rank also requires setting `sglang_max_lora_rank` to the same value.

Bridge mode loads the public block-FP8 checkpoint directly, dequantizing
trainer weights to BF16. It skips the FFT preset's HF-to-torch_dist conversion.
SGLang serves the original FP8 base and receives adapter updates. CPU base
backup and colocated offload follow upstream; LoRA still needs the full
frozen base model and substantial host memory. No pretrained adapter is
required. Loading arbitrary external GLM adapters and exporting a portable
PEFT adapter have not been validated by this recipe.

The dependencies are pinned independently of FFT:

| Component | Pin |
|---|---|
| Base image | `radixark/miles:glm53next@sha256:66725f740a6013b00d27e21fdfd480a24b0d3b5c61840342e9405bf1e09e5162` |
| Miles PR #3098 | `5a5353d36c9133958f9ddb62c93be463bc849f88` |
| [Megatron-Bridge PR #35](https://github.com/radixark/Megatron-Bridge/pull/35) | `6527b18e8bb0db994a267e6dfd4db7dafc669df9` |
| [SGLang PR #37753](https://github.com/sgl-project/sglang/pull/37753) | `94c97e3a7d7ab1dcbba82f4c09046d2c209b1aa7` |
| `sglang-kernel` | `0.4.6.post1` |

The recipe installs Bridge without replacing the image's PyTorch, CUDA, or
Megatron stack. Its KDA kernel patch is shared with FFT. A separate timing
adapter covers this PR's older synchronous driver, actor logprobs,
forward/backward, and optimizer step. Those three Miles files are restored
from the pinned checkout before instrumentation, and golden fixtures enforce
the source contract. No common launcher code changes are required.

This preset uses open companion PRs. Local tests do not establish numerical
correctness or successful adapter updates on the full model. Before merging,
run a one-step proof and a fresh ten-step smoke, verify readable responses,
reward improvement, train/rollout logprob agreement, adapter equality, saved
checkpoint behavior, and step/substep dashboard timings. Dispatch the standard
model validation workflow after GPU budget and capacity are confirmed.

Preflight on Modal verified the image build, SGLang's declared LoRA targets,
Miles' bidirectional target mapping, all six upstream Bridge helper tests
(FP8 dequantization, KDA convolution mapping, and mHC scale mapping), and the
exact recipe CLI constructing a 45-layer `Glm5NextModelProvider`. One B300
was needed because Transformer Engine imports require the CUDA driver. The
FP8 helper test moves its CUDA result to CPU for comparison with the upstream
CPU reference; its numeric tolerances are unchanged. No full model weights
were loaded, and this does not validate training, serving, or distributed sync.

```python
from modal_training_gym import (
    GLM_5_3_Flash_LoRA, GLM_5_3_Flash_LoRA_Recipe,
    HuggingFaceDataset, TrainConfig,
)

config = TrainConfig(
    model=GLM_5_3_Flash_LoRA(),
    recipe=GLM_5_3_Flash_LoRA_Recipe(num_rollout=2),
    dataset=HuggingFaceDataset(
        "zhuzilin/dapo-math-17k", hf_split="train[:30]",
        input_column="prompt", output_column="label", input_format="messages",
    ),
)
# config.train() allocates 24 B300 GPUs.
```

The registry name is `GLM-5.3-Flash-LoRA`:

```bash
uv run scripts/validate_model_configs.py check -m GLM-5.3-Flash-LoRA -n 2
```
