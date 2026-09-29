---
order: 0
---

# Migrating to the new dataset API

New versions of the Training Gym feature a reworked `DatasetConfig` API that is simpler and easier to customize. This guide breaks down the breaking changes so you can migrate your existing datasets to the new API.

## Training and evaluation datasets are now separate

Previously, a `DatasetConfig` instance was responsible for both a train and eval split. In the new API, each `DatasetConfig` instance now maintains only one split. To have a separate evaluation dataset, you must instantiate an additional `DatasetConfig` instance.

For example, with `HarborDataset`:

```python
# Before
dataset = HarborDataset(dataset_name="harbor/hello-world")

# After
train_dataset = HarborDataset(dataset_name="harbor/hello-world", split="train")
eval_dataset = HarborDataset(dataset_name="harbor/hello-world", split="eval")
```

If you use a recipe with a defined `eval_interval`, you must pass in an `eval_dataset` to your `TrainConfig` separately. This will supply an additional dataset so Slime or Miles can use it internally for evaluations:

```python
from modal_dojo import TrainConfig

config = TrainConfig(
    model=model,
    recipe=recipe,
    dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```

Training and evaluation datasets passed to `TrainConfig` must have the same shape; in particular, these must match:

* Input and label keys (`input_key()` and `label_key()`)
* Whether the chat template should be applied (`apply_chat_template()`)
* Modalities, for multimodal datasets

## Adapting your custom DatasetConfig subclasses

Previously, you defined custom datasets by setting `input_key` and `label_key`, then implementing `prepare()` and `load()`. In the new API, you instead provide implementations of `input_key()`, `label_key()`, and `rows()`.

`input_key()` and `label_key()` return the keys your dataset uses for prompts and labels, respectively. They are identical to the old `input_key` and `label_key` fields, but are now methods to make it easier for subclasses to determine these dynamically.

`rows()` returns an iterable collection of rows, where each row is a dictionary. It replaces the old `prepare()` and `load()` methods, as the base `DatasetConfig` class now handles writing these rows to disk for you.

Generally, to migrate a dataset, you convert `input_key` and `label_key` to their method counterparts, then you move your logic from `prepare()` and `load()` into `rows()`:

```python
# Before
class ExampleDataset(DatasetConfig):
    input_key = "prompt"
    label_key = "label"

    def load(self, split = "all") -> list[DatasetRow]:
        rows: list[DatasetRow] = []
        for i in range(5):
            rows.append({
                self.input_key: f"What is {i} + {i}?",
                self.label_key: str(i + i),
            })
        return rows

    def prepare(self, path, eval_paths) -> None:
        rows = self.load()
        for p in [path, *(eval_paths or {}).values()]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")


# After
class ExampleDataset(DatasetConfig):
    def input_key(self) -> str:
        return "prompt"

    def label_key(self) -> str:
        return "label"

    def rows(self) -> Iterable[DatasetRow]:
        for i in range(5):
            yield {
                self.input_key(): f"What is {i} + {i}?",
                self.label_key(): str(i + i),
            }
```

You then implement these additional methods to migrate other aspects of your dataset's behavior:

* `apply_chat_template()` replaces the `apply_chat_template` field. It returns a boolean that controls whether the Gym should tokenize your prompts using your model's chat template. The default implementation returns `True`, but you can override this and return `False` if you are passing in a raw prompt for a custom generation function.
* `write(path)` allows you to override how datasets are written to disk, which is useful if your existing dataset had custom logic in `prepare()`. If you implement `write(path)`, you should also implement `output_format()` to return either `jsonl` or `parquet`, then write your dataset in that format to the given `path`.
* `cache_key()` lets you customize where your dataset is cached for future runs. Return a stable string for the Gym to cache your dataset, or return `None` to always regenerate your dataset for each training run.

These fields have been removed:

* `always_prepare` has been removed on the base `DatasetConfig` class. For `HuggingFaceDataset` and `HarborDataset`, this functionality is now provided by the `always_download` property. For custom subclasses, returning `None` from `cache_key()` is equivalent to setting `always_prepare` to `True`.
* `writes_eval_paths` has been removed, as evaluation datasets are now completely separate from training datasets.
* `dataset_id` and `name` have been removed altogether.

## Using the new HuggingFaceDataset API

Previously, you subclassed `HuggingFaceDataset` to specify its parameters. In the new API, you instead instantiate `HuggingFaceDataset` directly and pass those parameters into the initializer:

```python
# Before
class HaikuDataset(HuggingFaceDataset):
    hf_repo = "statworx/haiku"
    input_column = "keywords"
    output_column = "text"
    prompt_template = "Write a haiku about {input}."
    output_format = "jsonl"
    always_prepare = True

train_dataset = HaikuDataset(hf_split="train[:10]")


# After
haiku_dataset = HuggingFaceDataset(
    "statworx/haiku",
    hf_split="train[:10]",
    input_column="keywords",
    output_column="text",
    input_format="text",
    prompt_template="Write a haiku about {input}.",
    always_download=True,
)
```

You also no longer set `input_key`, `label_key`, or `apply_chat_template` directly. Instead, you configure these through `input_column`, `output_column`, and a new `input_format` parameter. The value you use for `input_format` depends on the shape of your dataset:

* If your dataset's input consists of plain text that must be interpolated into a prompt, pass in `input_format="text"`. `HuggingFaceDataset` will convert each row's `input_column` into a list of OpenAI-format chat messages, including an optional `system_message` and a user message based on an optional `prompt_template`.
* If your dataset's input is already a list of messages, pass in `input_format="messages"` and set your `input_column` to the column containing those messages.
* If you were previously using `apply_chat_template=False`, pass in `input_format="raw"` and set `input_column` to the column containing your raw inputs.

When running your own eval loop, you previously used the `load()` method to iterate over the dataset's rows. In the new API, you call `rows()`, which now formats rows identically to how they appear in your reward function:

```python
# Before
for row in eval_dataset.load():
    prompt = eval_dataset.prompt_template.format(input=row["keywords"])
    response = deployment.chat([{"role": "user", "content": prompt}])


# After
for row in eval_dataset.rows():
    response = deployment.chat(row[eval_dataset.input_key()])
    label = row[eval_dataset.label_key()]
```

If your `input_format` is set to `text`, you will need to update your evaluation logic to read from each row's `input_key()` instead, regardless of whether you are iterating through the dataset yourself or using `EvalConfig`. Prompts in your dataset will be given as a list of formatted messages.

Finally, some fields and methods have been replaced or renamed:

* `always_prepare` has been renamed to `always_download`.
* `n_rows` has been removed. You should migrate to [Hugging Face's native slicing syntax instead.](https://gym.modal.dev/guides/dataset#hugging-face)

## Using the new HarborDataset API

Previously, you defined one `HarborDataset` instance for both training and evaluation. In the new API, you define two instances and vary the `split` parameter, as well as the number of `train_repeats` or `eval_repeats`:

```python
# Before
dataset = HarborDataset(
    path="/path/to/tasks",
    train_size=80,
    eval_size=20,
    shuffle_tasks=True,
    shuffle_seed=42,
    label_metadata_path="task.toml",
    train_repeats=4,
    eval_repeats=1,
)


# After
options = dict(
    path="/path/to/tasks",
    train_size=80,
    eval_size=20,
    shuffle_tasks=True,
    shuffle_seed=42,
    label_metadata_path="task.toml",
)
train_dataset = HarborDataset(split="train", train_repeats=4, **options)
eval_dataset = HarborDataset(split="eval", eval_repeats=1, **options)
```

If you were previously using `EvalConfig` with your `HarborDataset`, your evaluation function will now receive preformatted rows instead of task records, identical to what your generate and reward functions receive. Each row will contain:
* A `messages` list containing messages for the model
* A JSON-encoded `label` containing metadata for the task

## Adapting your MultimodalDataset instances

If you are constructing `MultimodalDataset` directly, you can continue to pass in a `modality` and a list of `rows`, each containing a `prompt`, `media`, and `label`, to the initializer.

If you subclass `MultimodalDataset` to provide a dynamic list of rows, you will need to:
* Rename your custom `rows()` method to `source_rows()`
* Specify modality by calling into the superclass's constructor

For example:

```python
# Before
class ImageQuestions(MultimodalDataset):
    def __init__(self, examples):
        self.examples = list(examples)
        super().__init__(modality="image")

    def rows(self):
        return [
            {
                "prompt": "Describe this image.",
                "media": image_path,
                "label": answer,
            }
            for image_path, answer in self.examples
        ]


# After
class ImageQuestions(MultimodalDataset):
    def __init__(self, examples):
        self.examples = list(examples)
        super().__init__(modality="image")

    def source_rows(self):
        return [
            {
                "prompt": "Describe this image.",
                "media": image_path,
                "label": answer,
            }
            for image_path, answer in self.examples
        ]
```

If you previously set `apply_chat_template` to `False`, you will instead need to override the `apply_chat_template()` method in a subclass:

```python
# Before
apply_chat_template = False

# After
def apply_chat_template(self) -> bool:
    return False
```

## Removing arbitrary keyword arguments from initializers

Previously, many `DatasetConfig` initializers allowed you to pass in arbitrary keyword arguments that would then be set on the instance. This is no longer the case, and many implicitly set fields have been moved to explicit keyword arguments. To migrate your code, you should audit calls to `DatasetConfig` initializers and check for any arguments that are not explicitly declared.

## Final checks

To verify your dataset before training, inspect a row from it using `rows()` and check that your prompt and label columns match what you expect.

For Hugging Face and Harbor datasets, confirm that your evaluation function accepts the same row structure as your reward function.
