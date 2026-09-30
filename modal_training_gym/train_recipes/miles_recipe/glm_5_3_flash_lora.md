# GLM-5.3-Flash bridge LoRA on Miles

`GLM_5_3_Flash_LoRA` selects `GLM_5_3_Flash_LoRA_Recipe`, an experimental
text GRPO preset based on [Miles PR #3098](https://github.com/radixark/miles/pull/3098).
The original `GLM_5_3_Flash_Recipe` remains the 32-GPU full-parameter preset.

The LoRA preset uses **3 nodes × 8 B300 GPUs**, TP8/EP24/PP1/ETP1, with one
colocated TP8/EP8 rollout engine per node. This preserves the PR's 24-H200
parallel layout. A two-step GSM8K validation passed on 24 H200s on
September 30, 2026; the default B300/DAPO combination remains unvalidated.
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
Megatron stack. Its KDA forward kernel patch is shared with FFT. A LoRA-only
backward patch restricts the FLA 0.4.2 fused KDA backward autotuner on Hopper
to BK32/BV32, four warps, one stage. The original search reproducibly accesses
invalid GPU memory on H200 with Triton 3.7.1; the conservative configuration
passes an isolated output/gradient comparison against a PyTorch reference.
Other architectures retain the original search. This is a configuration
workaround, not a diagnosis of the underlying compiler/kernel defect.
A separate timing
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

The September 30, 2026 H200 run
[`rectilinear-cabana-878d31aad74c`](https://modal.com/apps/modal-labs/helena-dev/ap-Lnc7ew9GqmGmV52jRsisZ0)
completed both optimizer steps and both adapter synchronizations. Ray reported
`SUCCEEDED`, and Training Gym recorded `completed`. It used commit `ded8f9e2`,
the full `zhuzilin/gsm8k` train split, 15 prompts × 8 responses per rollout,
8192 response tokens, 16384 training tokens, and the upstream TP8/EP24 topology.
Checkpoint saving was disabled, matching the upstream validation command.

| Metric | Step 1 | Step 2 |
|---|---:|---:|
| Correct responses / samples | 113 / 120 | 116 / 120 |
| Mean raw reward | 0.9417 | 0.9667 |
| Truncation | 0% | 0% |
| Gradient norm | 0.06937 | 0.11576 |
| Train/rollout logprob absolute difference | 0.01180 | 0.00846 |
| Train/rollout KL | 0.000743 | 0.000507 |
| Generation | 151.8 s | 30.2 s |
| Log probabilities | 301.8 s | 83.9 s |
| Forward/backward | 587.2 s | 210.1 s |
| Optimizer update | 3.07 s | 3.14 s |
| Adapter synchronization | 42.6 s | 35.7 s |

Downloaded traces from both rollouts had coherent prompts and responses.
The second rollout reported adapter weight version 2 with no mixed versions;
upstream's adapter equality checks completed without error. Dashboard timing
records cover startup, rollout, log probabilities, forward/backward, optimizer,
offload, and synchronization. The first step includes kernel compilation.
The two batches differ, so the higher reward is not evidence of generalization
or a statistically established improvement. This proves the requested two-step
H200 path, not a ten-step smoke test, B300 support, or checkpoint export.

Before the full run, 27 focused local tests passed. A one-H200 diagnostic
reproduced the stock FLA backward crash with synchronous CUDA launches. The
patched image then passed packed and strided Q/K/V tests with FP32 gates and
beta, lengths up to 16384, finite gradients, and a short-sequence reference
comparison for output plus all five gradients (under 0.4% relative RMS error).

Earlier B300 attempts stopped in CUDA graph capture or Transformer Engine's
MoE-sort module loading. A two-GPU sort/gradient reproducer did not reproduce
the latter. Eager CUDA loading caused a router-readiness timeout and was not
adopted. The actionable H200 failure and its workaround are described below.
FFT and shared launcher code were unchanged during these LoRA retries.

Set `recipe.environment["TRAINING_GYM_GLM53_LORA_DEBUG"] = "1"` to enable
recipe-scoped diagnostics. Each trainer logs phase boundaries and Triton
module-load boundaries, with bounded MoE-sort progress logging. A separate CPU
observer captures native stacks with `py-spy` and GPU counters if diagnostic
progress stops for 120 seconds. `GLM53_DEBUG_STALL_SECONDS` adjusts it.
There is no in-process traceback timer: the first instrumented H200 run lost
rank 0 during such a dump, before rollout, so that timer was removed. All three
serving engines had completed loading and graph capture. The timer-free
diagnostics were retained for the successful run above.
An observer report indicates a long operation, not proof of deadlock. The
observer does not synchronize CUDA, execute collectives, or dump tensor
contents, local variables, or environment credentials. Diagnostics are off
by default and are installed only in this LoRA recipe's image.
Set `GLM53_DEBUG_ARCHIVE_DIR=/checkpoints/glm53-debug` to also retain phase
markers, original Python exceptions, and observer captures in per-run,
per-rank JSONL files. Kernel and MoE progress stays in console logs to avoid
frequent writes to the volume. Archive failures preserve the training exception.

H200 run `bright-buck-93464065c328` exposed the original exception:
after rollout and log probabilities, the first backward pass failed in
FLA's `chunk_kda_bwd_kernel_wy_dqkg_fused` autotuning. A single-H200 reproducer
with `CUDA_LAUNCH_BLOCKING=1` failed during the stock configuration search
at BK32/BV64/four warps/two stages. BK32/BV32/four warps/one stage passed.
The upstream FLA int64 indexing fix was already present in this image;
reapplying it would not address this failure. The LoRA-only Hopper workaround
above avoids the failing search without changing the training algorithm.

Use `NCCL_DEBUG_SUBSYS=INIT,NET` instead of per-collective `COLL` output when
diagnosing. Even startup communication logs can exceed Modal's console rate
limit: the successful run's optimizer metrics were recovered from raw Ray
worker logs, while phase completion was retained in the JSONL archive and
dashboard timing records. An absent console line is not evidence of a skipped
optimizer update.

```python
from modal_training_gym import (
    GLM_5_3_Flash_LoRA, GLM_5_3_Flash_LoRA_Recipe,
    HuggingFaceDataset, TrainConfig,
)

config = TrainConfig(
    model=GLM_5_3_Flash_LoRA(),
    recipe=GLM_5_3_Flash_LoRA_Recipe(
        gpu_type="H200", num_rollout=2, rollout_max_response_len=8192,
        save=None, save_interval=None,
        extra_config={"num_steps_per_rollout": 1, "num_gpus_per_node": 8},
    ),
    dataset=HuggingFaceDataset(
        "zhuzilin/gsm8k", hf_split="train",
        input_column="messages", output_column="label", input_format="messages",
    ),
)
# config.train() allocates 24 H200 GPUs.
```

The registry name is `GLM-5.3-Flash-LoRA`; registry validation uses the default
B300/DAPO preset, rather than the H200/GSM8K override above:

```bash
uv run scripts/validate_model_configs.py check -m GLM-5.3-Flash-LoRA -n 2
```
