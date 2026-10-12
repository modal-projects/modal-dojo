# 04 — Gradients are tiny, spike or become nonfinite

Follow one batch through **reward → advantage → loss mask → gradient → optimizer → rollout weights**.

| What you observe | Start here |
| --- | --- |
| Updates appear absent | Follow the signal and optimizer checks below. |
| Gradients spike or become nonfinite | Inspect the affected update below. |
| Updates happen normally, but learning stalls | Recheck data/rewards ([01](debug-reward.md)), exploration ([02](debug-entropy.md)) and evaluation ([05](debug-evaluation.md)). An operating optimizer does not guarantee useful learning. |

## Follow the signal into an update

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

## Inspect a spiking or nonfinite update

A **pre-clipping gradient norm** can exceed the clipping threshold even when clipping works. It does not measure the parameter change. [Gradient clipping](https://docs.pytorch.org/docs/2.8/generated/torch.nn.utils.clip_grad_norm_.html)

Inspect that batch for outliers in advantages, response lengths and probability ratios; then check whether loss normalization lets a few contributions dominate the update. **Gradient clipping** rescales gradients. **Policy-ratio clipping** modifies the surrogate objective. **Rejection** excludes contributions: for example, [IcePop](https://arxiv.org/html/2510.18855v2#S2.SS3.SSS2) masks token-level policy-gradient contributions whose trainer-to-rollout probability ratio falls outside its bounds.

## Match the finding to a remedy

| Confirmed finding | Action |
| --- | --- |
| Wrong masks or sample alignment | Repair them; inspect the corrected batch again. |
| Most contributions rejected, or regularization suppresses learning | Check ratios and rejection bounds ([03](debug-logprobs.md)), or reference KL and its coefficient. Scalar loss magnitudes alone do not establish dominance. |
| Repeated skipped updates, NaNs or infinities | Find why steps were skipped. For nonfinite values, locate the first failing operation and its inputs before changing precision or loss scaling. |
| Excessive policy changes | Test a lower learning rate or fewer updates per batch. |
| Isolated finite spike with stable subsequent behavior | Inspect the batch; a recipe change may be unnecessary. |

**Verify:** intended samples contribute, optimizer steps succeed, rollout receives updated weights, and held-out learning remains stable. A smaller gradient or loss alone is insufficient.
