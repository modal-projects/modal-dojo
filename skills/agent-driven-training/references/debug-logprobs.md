# 03 — Trainer–rollout log-probability differences grow

Unintended disagreement between rollout and trainer probabilities can distort policy updates and contribute to training instability. Reducing it can stabilize training—even when both engines use the same weights. The verifier's reward can be correct while the probabilities used in the update are inconsistent.

## Identify the comparison

Record the engine and weight version behind each score. A current-trainer versus older-rollout comparison can combine two effects:

| Comparison | How to isolate it | First suspects |
| --- | --- | --- |
| **Engine disagreement** | Compare rollout and independently computed trainer scores using the **generating weights**. | Tokens, prefixes, scoring conventions, precision and MoE routing. |
| **Policy movement** | Within the trainer, score the same tokens using the **generating and current weights**. | Updates per batch, learning rate and async policy lag. |

Policy movement is expected after updates; investigate excessive movement using the update checks in [04](debug-updates.md) and lag checks in [08](debug-async.md). Synchronous execution does not eliminate engine disagreement. [Policy and engine distinction](https://arxiv.org/html/2602.15763v1#S3.SS2)

## Read the pattern, then check the update

A useful metric is **mean absolute log-probability difference on sampled tokens**,
not KL divergence. Check which tokens enter the average and how responses are
weighted; changing task mix or length can move it. A nonzero gap alone does not
establish unhealthy training; check updates and held-out behavior.

**Before changing the recipe:**

1. **Validate the scores.** Use the same token IDs and full prefixes; align temperature, vocabulary normalization and sampling filters where applicable. Compare independently computed scores—reusing rollout scores on both sides gives zero by construction.
2. **Check the same interval in training.** Inspect probability ratios, clipping/rejection fractions, gradients and held-out performance. Check what each ratio compares: old-versus-current clipping can be zero while trainer–rollout disagreement remains nonzero.

## Published example: routing mismatch and collapse

The R3 paper compares MoE math training with and without **rollout routing replay**, which reuses rollout expert choices during training.

- **What deteriorated:** without routing replay, the run's disagreement grows sharply and validation performance collapses.
- **What changed:** the comparison run enables routing replay, with TIS disabled in both runs.
- **What improved:** disagreement stays lower and learning continues. This supports routing replay as a stability intervention for the reported setup. These are separate training runs, not recovery of the collapsed run. [R3, Figures 5–6 and §5.2](https://arxiv.org/html/2510.11370v2#S5.SS2)

## Match the finding to a remedy

| Confirmed finding | Next action |
| --- | --- |
| Multi-turn tokens or prefixes differ | Use [**token-in/token-out (TITO)**](https://www.lmsys.org/blog/2026-05-13-no-token-left-behind/) to preserve generated token IDs and actual conditioning history. Check policy-loss masks. |
| Rollout filters the vocabulary, and training intends that restricted policy | Use [**top-p masking / sampling-support replay**](https://github.com/radixark/miles/pull/2596) to normalize over the recorded token set. |
| MoE expert choices differ | Test **R3**, then remeasure disagreement and learning. Other numerical differences can remain. |
| Precision or kernel changes increase disagreement | Compare with the previous precision path; check quantization and weight export. |
| Correctly computed scores still have consequential mismatch | Test [**TIS**](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#true-on-policy-rl), which clips correction weights, or [**IcePop**](https://arxiv.org/html/2510.18855v2#S2.SS3.SSS2), which rejects out-of-range contributions. These bounds can introduce bias; neither repairs incorrect inputs. |
| Rollout loads wrong or stuck weights | Repair publication/loading and verify the acknowledged version. For intentional async lag, check staleness ([08](debug-async.md)). |

**Verify:** for an input, routing or numerical fix, rescore fixed prefixes at matching weights and check whether disagreement decreases. For importance correction, inspect correction weights and how much signal is clipped or rejected; the raw gap need not shrink. In both cases, verify stable updates and held-out learning. Async training still needs lag control: fixing engine disagreement does not remove policy staleness.

**Observed example:** in matched R3-off and R3-on GLM runs, `train/train_rollout_logprob_abs_diff` was lower with R3 enabled; both runs remained stable. This illustrates the measurement check, not a reproduced collapse-and-rescue.
