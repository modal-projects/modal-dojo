# slime — Modal launcher for slime RL training

Thin Modal launcher that runs [slime](https://github.com/THUDM/slime) GRPO training on GPU clusters.

## Quick start

Define a `TrainConfig` with your model, dataset, and a `SlimeRecipe`, then call
`train()`:

```python
from modal_dojo import SlimeRecipe, TrainConfig, WandbConfig

config = TrainConfig(
    model=my_model,
    dataset=my_dataset,
    recipe=SlimeRecipe(
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        gpu_type="H100",
        metrics=WandbConfig(project="my-project"),
    ),
)


def train():
    return config.train()
```

Then run:

```bash
uv run modal run my_tutorial.py::train
```

`config.train()` handles model download and dataset prep automatically — if
either isn't cached yet, it runs those steps before training.

## SlimeRecipe

`SlimeRecipe` is a Pydantic dataclass that holds all configuration for a slime training run:
launcher instructions, cluster topology, RL hyperparameters, and checkpointing.

Every field on `SlimeRecipe` (except internal fields like `environment`, `async_mode`,
`metrics`, `image_overlay`, etc.) is forwarded to slime as a CLI argument:
`field_name` → `--field-name`.

See `modal_dojo/train_recipes/slime_recipe/recipe.py` for the full field list.

### Key field groups

- **Cluster**: `gpu_type`, `actor_num_nodes`, `actor_num_gpus_per_node`, `colocate`, `tensor_model_parallel_size`
- **RL algorithm**: `advantage_estimator`, `n_samples_per_prompt`, `eps_clip`, `kl_loss_coef`
- **Rollout**: `rollout_batch_size`, `rollout_max_response_len`, `rollout_temperature`
- **Training**: `global_batch_size`, `lr`, `lr_decay_style`, `weight_decay`, `optimizer`
- **Checkpointing**: `save`, `save_interval`, `megatron_to_hf_mode`
- **Eval**: `eval_interval`, `eval_config`
- **Reward**: `rm_type`, `custom_rm_path`, `custom_rm_function`

## YAML config fields

`eval_config`, `extra_config`, and `sglang_config` map to slime options that
normally take file paths (`--eval-config`, `--custom-config-path`,
`--sglang-config`). In `SlimeRecipe` you can write them as inline dicts — the
launcher materializes them to temp YAML files automatically:

```python
recipe = SlimeRecipe(
    eval_config={
        "eval": {
            "defaults": {"max_response_len": 16384},
            "datasets": [
                {"name": "aime", "path": "/data/aime.jsonl", "rm_type": "deepscaler"},
            ],
        }
    },
    extra_config={
        "max_turns": 3,
        "rollout_interaction_env_path": "examples.my_env.rollout",
    },
)
```

`extra_config` is also the escape hatch for any slime/sglang option that has no
`SlimeRecipe` field: keys become attributes on slime's parsed args and always
override same-named recipe fields. Alternatively, declare the flag as a field on
a recipe subclass — every non-launcher field is emitted as
`--<field-name-with-dashes>`, so new slime flags (and sglang server args via the
`sglang_*` prefix) work without editing `SlimeRecipe` itself. See the
`SlimeRecipe` docstring for the full field reference.

## Image overlay

To customize the container image (e.g. install extra packages), pass `image_overlay`:

```python
recipe = SlimeRecipe(
    image_overlay=lambda img: img.pip_install("my-package"),
)
```

## Dev overlay

To test a local slime checkout without rebuilding the base image, set
`local_slime`. The overlay is used as-is; build-time patches are not reapplied,
so include any required changes in the checkout. This also means the
substep-timing patch is not applied automatically, and local runs will not
produce substep timing unless the checkout includes the equivalent changes:

```python
recipe = SlimeRecipe(
    local_slime="/path/to/your/slime",
)
```
