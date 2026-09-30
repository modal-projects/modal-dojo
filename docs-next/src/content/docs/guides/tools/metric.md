---
order: 2
---

# Logging metrics

The [observability dashboard](https://dojo.modal.dev/guides/dashboard) captures the most important plots and metadata you'd care about during training, and its **Metrics** tab charts every scalar the [underlying framework](https://miles.radixark.com/docs) logs through `wandb.log` — with any metric provider. Use [Weights & Biases](https://wandb.ai) or [Trackio](#trackio) when you also want those numbers in an external tracker.

## Dashboard only

This is the default: `SlimeRecipe` and `MilesRecipe` start with `metrics=DashboardMetricConfig()` (a few model recipes override it, e.g. `Qwen3_6_27B_Recipe_Agentic` ships with Trackio), which sends the framework's metrics to the dashboard and nowhere else — no account, API key, or extra server. Set it explicitly to name the project or group:

```python
from modal_dojo import DashboardMetricConfig, Qwen3_5_4B, Qwen3_5_4B_Recipe, TrainConfig

config = TrainConfig(
    model=Qwen3_5_4B(),
    dataset=my_dataset,
    recipe=Qwen3_5_4B_Recipe(
        # ...
        metrics=DashboardMetricConfig(project="my-rl-project"),  # optional; this is the default provider
    ),
)

run = config.launch()
```

Open the run in the dashboard and switch to the Metrics tab: keys are grouped by prefix like W&B panels (`train/`, `rollout/`, `perf/`, ...), the search box filters every group, and charts refresh while the run trains. Only finite scalars are kept (nested dicts flatten to `a/b`; images, tables, and strings are dropped).

Pass `metrics=None` to turn metric logging off entirely — the framework runs without `--use-wandb` and nothing is stored.

## Weights & Biases

When you need everything W&B offers (media, tables, cross-project reports), pass a `WandbConfig`. The framework logs to W&B as usual and the same scalars also appear in the dashboard's Metrics tab.

First, you'll need to create a [Modal Secret](https://modal.com/docs/guide/secrets) with your API key:

```bash
modal secret create wandb-secret WANDB_API_KEY=<your-api-key>
```

Then, just pass it in your [training recipe](https://dojo.modal.dev/guides/recipe):

```python
from modal_dojo import Qwen3_5_4B, Qwen3_5_4B_Recipe, TrainConfig, WandbConfig

config = TrainConfig(
    model=Qwen3_5_4B(),
    dataset=my_dataset,
    recipe=Qwen3_5_4B_Recipe(
        # ...
        metrics=WandbConfig(
            project="my-rl-project",
            group="lr-sweep",  # optional: organize related runs
        ),
    ),
)

run = config.launch()
```

See the [reference page](https://dojo.modal.dev/reference/wandbconfig) for the full list of parameters.

When launching a [hyperparameter sweep](https://dojo.modal.dev/tutorials/param_sweep), the `group` parameter is especially useful to overlay multiple runs' reward curves.

## Trackio

[Trackio](https://huggingface.co/docs/trackio) is an open-source W&B alternative which you can deploy on Modal or on a Hugging Face Space. You can even self-host! Modal Dojo installs it in the training image and routes the framework's metric calls to it whenever a recipe uses `TrackioConfig`; scalars also appear in the dashboard's Metrics tab.

### Deploy on Modal

You can host a Trackio server on Modal:

```python
from modal_dojo import TrackioConfig

metrics = TrackioConfig.deploy_to_modal(project="my-rl-project")
```

Just like the [main dashboard](https://dojo.modal.dev/guides/dashboard), the Trackio dashboard is unauthenticated unless you set a password:

```bash
modal-dojo set-password
```

Note that unlike the main dashboard, this will not redeploy the Trackio dashboard. I.e., you'll have to rerun `deploy_to_modal()` after changing it.

### Deploy on a Hugging Face Space

Simply specify a `space_id` and optionally a `bucket_id`:

```python
from modal_dojo import Qwen3_5_4B, Qwen3_5_4B_Recipe, TrainConfig, TrackioConfig

config = TrainConfig(
    model=Qwen3_5_4B(),
    dataset=my_dataset,
    recipe=Qwen3_5_4B_Recipe(
        # ...
        metrics=TrackioConfig(
            project="my-rl-project",
            space_id="my-org/training-metrics",
            bucket_id="my-org/training-metrics",  # optional
        ),
    ),
)
```

### Self-hosted

You'll need to create a [Modal Secret](https://modal.com/docs/guide/secrets) with your API key:

```bash
modal secret create trackio-write-token TRACKIO_WRITE_TOKEN=<your-api-key>
```

Then, it's as easy as:

```python
metrics = TrackioConfig(
    project="my-rl-project",
    server_url="https://trackio.example.com",
    modal_secret_name="trackio-write-token",
)
```

See the [reference page](https://dojo.modal.dev/reference/trackioconfig) for all parameters.
