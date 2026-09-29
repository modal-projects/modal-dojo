---
order: 3
---

# Training in the Modal Dojo

Once you have your model, dataset, and training recipe, you're ready to start training:

```python
config = TrainConfig(
    model=model,
    dataset=dataset,
    recipe=recipe,
)
run = config.launch()

print(f"Run ID: {run.training_run_id}")
print(f"Modal App: {run.modal_app_id}")
print(f"Modal App URL: {run.modal_app_url}")
```

The training run will continue in the background entirely on Modal-managed infrastructure, so sit back and relax! At any time, you can easily observe its progress on the [Modal Dojo dashboard](https://gym.modal.dev/guides/dashboard) or with the [CLI](https://gym.modal.dev/reference/cli).

## Handling checkpoints

After the run has completed, you'll likely want to get its saved checkpoints (which are stored on a [Modal Volume](https://modal.com/docs/guide/volumes)) to either deploy them as an [Endpoint](https://modal.com/docs/guide/endpoints) or continue training with a new configuration.

Note that you can always access runs by their ID:

```python
run = TrainingRun.from_id("bristled-pine-a7c3e91d4b2")
```

While the run is in-progress, you can poll for the latest checkpoint:

```python
import time

with config.launch() as run:
    checkpoint = None
    while True:
        done = run.done()
        latest = run.latest_checkpoint()
        if latest is not None and latest != checkpoint:
            checkpoint = latest
        if done:
            break
        time.sleep(30)
```

Now, we could do offline evals:

```python
deployment = Endpoint.launch(
    model, checkpoint, unauthenticated=True, recreate_if_existing=True
)
deployment.wait_until_ready()
print(f"trained model deployed to {deployment.url}")


def score(response, label):
    return 1 if response==label else 0


def run_eval(deployment, max_concurrency: int = 2) -> float:
    from concurrent.futures import ThreadPoolExecutor

    def _score_one(example):
        prompt = example["prompt"][0]["content"]
        msg = deployment.chat(
            [{"role": "user", "content": prompt}],
        )
        response = msg.get("content") or msg.get("reasoning_content") or ""
        return score(response, example["label"])

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        scores = list(executor.map(_score_one, eval_dataset.rows()))
    percent_correct = (
        len([s for s in scores if s == 1]) / len(scores) if scores else float("nan")
    )
    return percent_correct

print("running trained model evaluation...")
correct = run_eval(deployment)
print(f"percent correct: {correct:.1%}")
```

Or train the model further with the existing checkpoint as a starting point:

```python
config = TrainConfig(
    model=model,
    dataset=dataset,
    resume_from_checkpoint=checkpoint,
    recipe=recipe,
)
```

Why might you want to continue training the model? As an example, you can implement curriculum learning by increasing the difficulty of the data and reward function over time. The new run starts from the trained weights with an untrained optimizer:

```python
simple_config = TrainConfig(
    model=model,
    dataset=simple_dataset,
    recipe=simple_recipe,
)

with simple_config.launch() as simple_run:
    simple_checkpoint = None
    while True:
        done = simple_run.done()
        latest = simple_run.latest_checkpoint()
        if latest is not None and latest != simple_checkpoint:
            simple_checkpoint = latest
        if done:
            break
        time.sleep(30)

complex_config = TrainConfig(
    model=model,
    dataset=complex_dataset,
    resume_from_checkpoint=simple_checkpoint,
    recipe=complex_recipe,
)

with complex_config.launch() as complex_run:
    complex_checkpoint = None
    while True:
        done = complex_run.done()
        latest = complex_run.latest_checkpoint()
        if latest is not None and latest != complex_checkpoint:
            complex_checkpoint = latest
        if done:
            break
        time.sleep(30)
```
