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

Upstream reports two separate full-model validations: an
[eight-step GSM8K run](https://github.com/radixark/miles/pull/3098#issuecomment-5523089411)
with an 8192-token response cap, and a later
[20-step DAPO run](https://github.com/radixark/miles/pull/3098#issuecomment-5531105191)
with a 7168-token cap. This preset follows the latter workload. Both reports
use H200 GPUs; using the same flags on B300 does not reproduce that environment.
The PR describes Triton 3.6 validation, while our image probe reports Triton
3.7.1. A complete upstream validation image/version manifest was not supplied
in those reports.

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

The September 29, 2026 two-step validation passed 246 local tests and the
isolated GPU preflight, then exercised the full 24-B300 topology. Run
`compact-twill-3ac0d3dafb0e` loaded the base through Bridge, synchronized the
initial adapters to all three rollout engines, and generated 120 responses
in 284 seconds (mean math reward 0.45; 54.2% reached the response limit).
It stalled before the first optimizer update: repeated trainer stacks on
multiple ranks were blocked in Transformer Engine's MoE chunk-sort kernel
loading, with the native stack in `cuModuleLoadData`. The run was stopped.
Neither the two optimizer steps nor checkpoint saving is validated. An
earlier attempt stalled during one rollout engine's CUDA graph capture;
an identical retry passed that stage. These observations do not establish
the underlying runtime cause. The FFT recipe and shared launcher were not
changed for these validation attempts.

A two-B300 diagnostic passed fused MoE sorting and gradient comparisons
after NCCL all-to-all under both lazy and eager CUDA loading. It did not
reproduce the full-cluster stall. Eager initialization took 320 seconds.
A validation-only retry (`merry-mean-5ae4ab814954`) with
`CUDA_MODULE_LOADING=EAGER` failed earlier, at Miles' fixed 120-second
router-readiness timeout. Eager loading has not been adopted by the recipe.
Further work should isolate kernel preloading to trainer processes and prove
the distributed training path before another full validation attempt.

Set `recipe.environment["TRAINING_GYM_GLM53_LORA_DEBUG"] = "1"` to enable
recipe-scoped diagnostics. Each trainer logs phase boundaries and Triton
module-load boundaries, with bounded MoE-sort progress logging. A separate CPU
observer captures native stacks with `py-spy` and GPU counters if diagnostic
progress stops for 120 seconds. `GLM53_DEBUG_STALL_SECONDS` adjusts it.
There is no in-process traceback timer: the first instrumented H200 run lost
rank 0 during such a dump, before rollout, so that timer was removed. All three
serving engines had completed loading and graph capture; training is still
unvalidated.
An observer report indicates a long operation, not proof of deadlock. The
observer does not synchronize CUDA, execute collectives, or dump tensor
contents, local variables, or environment credentials. Diagnostics are off
by default and are installed only in this LoRA recipe's image.
Set `GLM53_DEBUG_ARCHIVE_DIR=/checkpoints/glm53-debug` to also retain phase
markers, original Python exceptions, and observer captures in per-run,
per-rank JSONL files. Kernel and MoE progress stays in console logs to avoid
frequent writes to the volume. Archive failures preserve the training exception.

The H200 retry after removing the timer completed initialization and initial
adapter synchronization, then generated 120 GSM8K samples with mean reward
0.8917, mean response length 206 tokens, and no truncation (144 seconds).
Inspected responses were coherent. It entered log-probability computation but
exited with code 1 before any verified optimizer step. GPU XID warnings were
recorded, but the exact exception was obscured by NCCL collective logging and
Modal's output-rate limit; the root cause remains unresolved. For diagnostics,
use `NCCL_DEBUG_SUBSYS=INIT,NET` rather than per-collective `COLL` output. The
two-step LoRA validation has not passed.

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
