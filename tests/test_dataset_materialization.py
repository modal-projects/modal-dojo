import json
import pickle

import pytest

from modal_training_gym.common.dataset import DatasetConfig, HarborDataset, _SftDataset
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.launcher_helpers import (
    write_dataset_if_needed,
    write_datasets,
)
from modal_training_gym.train_recipes.base import BaseTrainRecipe


class RowsDataset(DatasetConfig):
    def __init__(self, key: str | None, value: str = "row") -> None:
        self.key = key
        self.value = value
        self.write_count = 0

    def cache_key(self) -> str | None:
        return self.key

    def input_key(self) -> str:
        return "prompt"

    def label_key(self) -> str:
        return "label"

    def rows(self):
        yield {"prompt": self.value, "label": self.value}

    def write(self, path: str) -> None:
        self.write_count += 1
        super().write(path)


def test_resolve_data_paths_uses_cache_key():
    dataset = RowsDataset("org/data-train")
    path = BaseTrainRecipe._resolve_data_paths(dataset)
    assert path.startswith("/data/org-data-train-")
    assert path.endswith(".jsonl")
    assert "%2F" not in path
    assert path != BaseTrainRecipe._resolve_data_paths(RowsDataset("org_data-train"))


def test_resolve_data_paths_generates_fresh_random_id():
    dataset = RowsDataset(None)
    first = BaseTrainRecipe._resolve_data_paths(dataset)
    assert BaseTrainRecipe._resolve_data_paths(dataset) != first
    assert BaseTrainRecipe._resolve_data_paths(RowsDataset(None)) != first
    assert not hasattr(dataset, "_materialization_id")


def test_dataset_fields_use_discrete_eval_dataset():
    train_dataset = RowsDataset("train")
    eval_dataset = RowsDataset("eval")
    fields = BaseTrainRecipe._dataset_to_fields(
        train_dataset,
        eval_dataset,
        dataset_path="/data/train.jsonl",
        eval_dataset_path="/data/eval.jsonl",
    )
    assert fields["prompt_data"] == "/data/train.jsonl"
    assert fields["eval_prompt_data"] == ["eval", "/data/eval.jsonl"]
    assert fields["input_key"] == "prompt"
    assert fields["label_key"] == "label"


def test_write_dataset_if_needed_reuses_valid_file(tmp_path):
    dataset = RowsDataset("train")
    path = str(tmp_path / "train.jsonl")
    assert write_dataset_if_needed(dataset, path)
    assert not write_dataset_if_needed(dataset, path)
    assert dataset.write_count == 1
    assert json.loads((tmp_path / "train.jsonl").read_text()) == {
        "prompt": "row",
        "label": "row",
    }


def test_write_caller_creates_parent_directory(tmp_path):
    dataset = RowsDataset("train")
    path = str(tmp_path / "nested" / "train.jsonl")
    with pytest.raises(FileNotFoundError):
        dataset.write(path)
    assert write_dataset_if_needed(dataset, path)


def test_write_datasets_writes_train_and_eval(tmp_path):
    train_dataset = RowsDataset("train", "training")
    eval_dataset = RowsDataset("eval", "evaluation")

    assert write_datasets(
        train_dataset,
        eval_dataset,
        str(tmp_path / "train.jsonl"),
        str(tmp_path / "eval.jsonl"),
    )
    assert train_dataset.write_count == 1
    assert eval_dataset.write_count == 1


def test_eval_dataset_fields_must_match_training_dataset():
    class OtherInputDataset(RowsDataset):
        def input_key(self) -> str:
            return "messages"

    class OtherLabelDataset(RowsDataset):
        def label_key(self) -> str:
            return "answer"

    class OtherChatTemplateDataset(RowsDataset):
        def apply_chat_template(self) -> bool:
            return False

    with pytest.raises(TrainingGymConfigError, match="same input_key"):
        BaseTrainRecipe._validate_datasets(
            RowsDataset("train"), OtherInputDataset("eval")
        )
    with pytest.raises(TrainingGymConfigError, match="same label_key"):
        BaseTrainRecipe._validate_datasets(
            RowsDataset("train"), OtherLabelDataset("eval")
        )
    with pytest.raises(TrainingGymConfigError, match="same apply_chat_template"):
        BaseTrainRecipe._validate_datasets(
            RowsDataset("train"), OtherChatTemplateDataset("eval")
        )


def test_harbor_instances_select_discrete_splits(tmp_path):
    for name in ("one", "two", "three"):
        task = tmp_path / name
        task.mkdir()
        (task / "instruction.md").write_text(f"solve {name}")

    train = HarborDataset(
        task_root=str(tmp_path),
        split="train",
        train_size=2,
        eval_size=1,
        train_repeats=2,
    )
    evaluation = HarborDataset(
        task_root=str(tmp_path),
        split="eval",
        train_size=2,
        eval_size=1,
    )

    train_rows = list(train.rows())
    eval_rows = list(evaluation.rows())
    assert len(train_rows) == 4
    assert len(eval_rows) == 1
    train_prompts = {row["messages"][0]["content"] for row in train_rows}
    assert eval_rows[0]["messages"][0]["content"] not in train_prompts
    assert train.cache_key() != evaluation.cache_key()


def test_harbor_always_download_disables_materialization_reuse():
    dataset = HarborDataset(
        dataset_name="harbor/example",
        always_download=True,
    )

    assert dataset.cache_key() is None
    assert BaseTrainRecipe._resolve_data_paths(
        dataset
    ) != BaseTrainRecipe._resolve_data_paths(dataset)


class PairDataset(DatasetConfig):
    def __init__(self, row: dict) -> None:
        self.row = row

    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> str:
        return "label"

    def rows(self):
        yield self.row


USER = {"role": "user", "content": "hi"}
ASSISTANT = {"role": "assistant", "content": "hello"}


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ("hi", [USER, ASSISTANT]),
        ([USER], [USER, ASSISTANT]),
        ([ASSISTANT], [ASSISTANT]),
    ],
)
def test_sft_dataset_formats_rows(messages, expected):
    dataset = pickle.loads(
        pickle.dumps(_SftDataset(PairDataset({"messages": messages, "label": "hello"})))
    )
    assert [row["messages"] for row in dataset.rows()] == [expected]


@pytest.mark.parametrize("row", [{"messages": "hi", "label": ""}, {"label": "hello"}])
def test_sft_dataset_rejects_incomplete_rows(row):
    with pytest.raises(TrainingGymConfigError, match="SFT row"):
        list(_SftDataset(PairDataset(row)).rows())


def test_sft_rejects_eval_dataset():
    with pytest.raises(TrainingGymConfigError, match="eval_dataset"):
        BaseTrainRecipe._validate_datasets(
            RowsDataset("a"), RowsDataset("b"), loss_type="sft_loss"
        )
