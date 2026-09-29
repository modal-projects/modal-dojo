# GLM-5.3-Flash on Miles

`GLM_5_3_Flash` and `GLM_5_3_Flash_Recipe` port the text GRPO configuration from
[Miles #2786](https://github.com/radixark/miles/pull/2786) and its
[model guide](https://miles.radixark.com/docs/models/glm/glm5-3-flash).
The recipe is registered with `MilesRecipe.get_base_recipe` and the
`GLM-5.3-Flash` validation target. Live training on Modal is pending.

The preset uses **4 nodes × 8 B300 GPUs**, TP8/PP4/EP8/ETP1, with pipeline
stages of 11/11/11/12 layers. This maps upstream's 32-GPU layout onto Modal's
eight-GPU workers. Rollouts are colocated, with one TP8/EP8 engine per node.
The default is one rollout: four DAPO prompts, eight samples each, at most
4096 response tokens. Adam uses `lr=1e-6`; training uses fixed microbatches,
full recompute, and disk offload. Routing replay is optional.

The custom KDA/DSA/mHC model specification comes from Miles' Python
`glm5.3-flash` model-argument provider. The `glm5_next` raw weight converter
handles the mHC weight mapping. Vision and MTP are not trained; this preset
does not add multimodal or LoRA support.

The image is pinned to the multi-architecture `glm53next` digest, with Miles
overlaid at merge commit `cc76e23915b2132ecfff65b97331fd83b02637e6`. The image
supplies SGLang `9a26e749` and Megatron `e8f57451`. The Miles overlay includes
the final SwiGLU clamp and architecture corrections missing from the original
image. The shared image and other presets are unchanged.

Two recipe-scoped adapters are necessary. The KDA patch backports the PR's
`next_power_of_2` hoist to the image's FLA 0.4.2 kernel; Triton 3.7 rejects that
Python call inside a JIT function. The timing adapter instruments the merged
Miles synchronous driver, logprob computation, forward/backward, and optimizer
step because their source no longer matches the shared timing patch. Both
adapters fail on unexpected source changes and have golden snapshot tests.
The timing adapter covers the synchronous preset; asynchronous training has
not been validated with this image.

The public checkpoint is approximately 328 GB of block-FP8 tensors. The image's
inherited DeepSeek mbridge reader dequantizes `weight_scale_inv` tensors to
BF16 during conversion; no additional checkpoint rewrite is needed.

Conversion uses TP8/PP1/EP8/ETP1 on one node. Its pipeline layout is independent
of training, so the 11/12 layer split must not be passed to conversion. The
conversion job reserves 1 TiB of local disk; training reserves 2 TiB per node
for offloaded state and checkpoint staging. Saves contain parameters only;
resuming restarts optimizer moments and RNG state.

The preset retains the standard Training Gym dashboard hooks and substep
instrumentation. Before merging, obtain GPU budget and capacity approval, run
a one-step proof followed by a fresh ten-step smoke, and dispatch the model
validation workflow. Check readable responses, nonzero then increasing
reward, train/rollout logprob agreement, and both step and substep timings.

```bash
uv run scripts/validate_model_configs.py check -m GLM-5.3-Flash -n 1
uv run scripts/validate_model_configs.py check -m GLM-5.3-Flash -n 10
```
