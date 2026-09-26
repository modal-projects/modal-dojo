---
order: 2
---

# Cooking with a recipe

Lastly, before we start training, we need a recipe.

While the model and dataset dictate what will be trained, the recipe dictates how training will occur by specifying parameters for:

- Training type
- Hardware and parallelism
- Total rollout size
- Environment
- Etc.

```python
from modal_training_gym import Qwen3_5_4B_Recipe

recipe = Qwen3_5_4B_Recipe()
```

We provide optimized recipes for all supported models in the Training Gym, but note that they are easily extensible to fit whatever use case you may have. Under the hood, each recipe is backed by one of two backend frameworks: [Miles](https://github.com/radixark/miles) or [Slime](https://github.com/THUDM/slime). All recipes allow you to specify framework-native parameters using the corresponding recipe fields.

This guide will focus on the most important ones. However, you can see the full lists for each of the base classes (i.e., [MilesRecipe](https://gym.modal.dev/reference/milesrecipe) and [SlimeRecipe](https://gym.modal.dev/reference/slimerecipe)).

See [this guide](https://gym.modal.dev/guides/metric) for more details on logging integrations.

## Training type

For supervised fine-tuning (SFT), set `loss_type="sft_loss"`. At the moment, passing `eval_dataset` is not supported.

```python
from modal_training_gym import HuggingFaceDataset, Qwen3_0_6B, Qwen3_0_6B_Recipe, TrainConfig

conversations = HuggingFaceDataset(
    "HuggingFaceH4/no_robots",
    input_column="messages",
    input_format="messages",
)

TrainConfig(
    model=Qwen3_0_6B(),
    dataset=conversations,
    recipe=Qwen3_0_6B_Recipe(loss_type="sft_loss", num_epoch=3),
).train()
```

Prompt and answer datasets also work too:

```python
pairs = HuggingFaceDataset("statworx/haiku", input_column="keywords", output_column="text")
```

## Hardware and parallelism

```python
recipe = Qwen3_5_4B_Recipe(
    # ...
    gpu_type="B300",
    actor_num_nodes=1,
    actor_num_gpus_per_node=8,
    colocate=True,
)
```

The Gym runs your training workloads across one or more nodes on Modal, each with one or more GPUs. Smaller models (i.e., tens of billions of parameters) can be trained on a single node, while larger models may require a [multi-node cluster](https://modal.com/docs/guide/multi-node-training).

Each recipe automatically provisions the smallest cluster shape that will work, but you may want to increase this to maximize throughput. You can choose any type from [Modal’s supported GPUs](https://modal.com/docs/guide/gpu#picking-a-gpu). Also, note that you may not actually need (or even want!) multiple nodes: we suggest setting `actor_num_gpus_per_node` to the [maximum amount](https://modal.com/docs/guide/gpu#specifying-gpu-count) to minimize unnecessary communication between nodes.

Actor parameters pertain to your training cluster, and rollout parameters your rollout cluster. When `colocate` is set to `True`, these are one and the same. When set to `False`, this will create a separate cluster for inference (i.e., disaggregated, async RL), so be sure you have the budget for it!

```python
Qwen3_5_4B_Recipe(
    # ...
    gpu_type="B300",
    colocate=False,
    actor_num_nodes=1,
    actor_num_gpus_per_node=8,
    rollout_num_gpus=8,
    rollout_num_gpus_per_engine=8,
)
```

You can also tune how model computations are parallelized and sharded across multiple GPUs. These parameters can be difficult to determine and may differ for each model, so we provide defaults in each model’s recipe. However, if you’re experiencing out-of-memory errors or want complete control over how your GPUs are utilized, you can manually set these yourself:

```python
from modal_training_gym import Qwen3_5_4B_Miles_Recipe

Qwen3_5_4B_Miles_Recipe(
    # ...
    tensor_model_parallel_size=2,
    sequence_parallel=True,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=1,
)
```

## Rollouts

Each step of training involves the model generating rollouts to calculate rewards. More specifically, a random subset is taken from our dataset to prompt the model, and the model generates one or more completions for each prompt.

The four most important parameters to specify are:

- `num_rollout`: the number of steps.
- `rollout_batch_size`: the number of prompts taken from the dataset for each step.
- `n_samples_per_prompt`: the number of rollouts sampled for each prompt.
- `global_batch_size`: the number of samples per optimizer step.

Note that the latter three are [related](https://miles.radixark.com/docs/user-guide/concepts#the-four-knob-invariant).

```python
Qwen3_5_4B_Recipe(
    # ...
    num_rollout=10,
    rollout_batch_size=8,
    n_samples_per_prompt=4,
    global_batch_size=16,
)
```

You'll want to start with low values to verify training works (e.g., 1, 2, 2, respectively), then increase them until the rewards curves are no longer noisy.

The effect of these parameters on run length and cost is multiplicative; the parameters above imply a total of 10 × 8 × 4 = 320 samples taken over the course of a run.

## Environment

An environment specifies how the model acts and how its responses are rewarded. The underlying frameworks are [environment-agnostic](https://miles.radixark.com/docs/user-guide/environments), so you have full control over the environment.

An example of a simple reward function is shown below. Take note of the `async` keyword and the argument list, as these are particulars of the underlying frameworks.

```python
async def my_custom_rm(args, sample, **kwargs) -> float:
    response = model.parse_response(sample.response)
    sample.metadata["em_dashes_used"] = response.content.count("—")
    return 1 if response.content == sample.label else 0


recipe = Qwen3_5_4B_Recipe(
    # ...
    custom_rm_function=my_custom_rm,
)
```

The simplest reward functions (like the above) return binary scores for correct or incorrect responses. Likely, though, you'll want to provide the model with more granular information for better training performance; for example, giving an exact distance between its response and the expected answer. And to enable use cases like code generation and gameplay, you'll want to incorporate external components such as a [Modal Sandbox](https://modal.com/docs/guide/sandboxes).

For logging purposes, you can attach metadata to each sample for more observability in the [dashboard](https://gym.modal.dev/guides/dashboard/).

When your task requires something beyond a single-turn interaction, all it takes is implementing a [custom generate](https://miles.radixark.com/docs/user-guide/generate-endpoint) function.

```python
async def my_custom_generate(args, sample, sampling_params):
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    # step 1

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    tokenizer = GenerateState(args).tokenizer
    
    # step 2

    output = await post(
        url,
        {
            "text": sample.prompt,
            "sampling_params": sampling_params,
        },
    )
    finish_type = output["meta_info"]["finish_reason"]["type"]
    if finish_type == "abort":
        sample.status = Sample.Status.ABORTED
        return sample
    response = output["text"]
        
    # step 3

    prompt_token_ids = tokenizer(
        sample.prompt,
        add_special_tokens=False
    )["input_ids"]
    response_token_ids = tokenizer(
        response,
        add_special_tokens=False
    )["input_ids"]
    loss_mask = [1] * len(response_token_ids)
    
    # step 4

    sample.tokens = prompt_token_ids + response_token_ids
    sample.response = response
    sample.response_length = len(response_token_ids)
    sample.loss_mask = loss_mask
    if finish_type == "length":
        sample.status = Sample.Status.TRUNCATED
    else:
        sample.status = Sample.Status.COMPLETED
    sample.metadata = {}
    return sample


recipe = Qwen3_5_4B_Recipe(
    custom_generate_function=my_custom_generate,
)
```

Typically, your generation function will:

1. Determine the URL of the internal SGLang server and set up a tokenizer for later use.
2. Make one or multiple calls against the SGLang endpoint to generate responses from the model.
3. Tokenize the prompt and response with a corresponding loss mask.
4. Set response fields on the sample.

If the function generates the prompts (i.e., no initial dataset), you must use the [OnlineRollout](https://gym.modal.dev/reference/onlinerollout) class.

Since the containers use [Modal Images](https://modal.com/docs/guide/images) under the hood, you can easily use external packages by extending the base image:

```python
recipe = Qwen3_5_4B_Recipe(
    # ...
    image_overlay=lambda image: image.pip_install("syllables"),
)
```
