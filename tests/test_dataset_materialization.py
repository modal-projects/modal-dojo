import json
import pickle

import pytest

from modal_dojo.common.dataset import DatasetConfig, HarborDataset, _SftDataset
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.launcher_helpers import (
    write_dataset_if_needed,
    write_datasets,
)
from modal_dojo.train_recipes.base import BaseTrainRecipe


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

    with pytest.raises(DojoConfigError, match="same input_key"):
        BaseTrainRecipe._validate_datasets(
            RowsDataset("train"), OtherInputDataset("eval")
        )
    with pytest.raises(DojoConfigError, match="same label_key"):
        BaseTrainRecipe._validate_datasets(
            RowsDataset("train"), OtherLabelDataset("eval")
        )
    with pytest.raises(DojoConfigError, match="same apply_chat_template"):
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
    with pytest.raises(DojoConfigError, match="SFT row"):
        list(_SftDataset(PairDataset(row)).rows())


def test_miles_pretokenized_rows_materialize_and_replay_without_dummy_text(tmp_path):
    from types import SimpleNamespace
    from modal_dojo.frameworks.miles.sft_rollout import generate_rollout

    metadata = {"input_ids": [5, 6, 7, 8], "loss_mask": [0, 1, 0, 1]}
    dataset = _SftDataset(
        PairDataset({"metadata": metadata}),
        allow_pretokenized=True,
        require_pretokenized=True,
    )
    path = tmp_path / "train.jsonl"
    dataset.write(str(path))
    dataset.validate_written(str(path))
    row = json.loads(path.read_text())
    assert row == {"messages": [], "metadata": metadata}
    assert dataset.label_key() is None
    sample = SimpleNamespace(prompt=row["messages"], metadata=row["metadata"])
    buffer = SimpleNamespace(get_samples=lambda n: [[sample]])
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=1)
    ((out,),) = generate_rollout(args, 0, buffer)
    assert out.tokens == metadata["input_ids"]
    assert out.response_length == 3
    assert out.loss_mask == [1, 0, 1]


def test_miles_pretokenized_rows_are_validated_during_materialization():
    dataset = _SftDataset(
        PairDataset({"metadata": {"input_ids": [1, 2], "loss_mask": [0, 0]}}),
        allow_pretokenized=True,
    )
    with pytest.raises(DojoConfigError, match=r"SFT row 0.*trains no token"):
        list(dataset.rows())


def test_required_pretokenized_mode_rejects_conversation_rows():
    dataset = _SftDataset(
        PairDataset({"messages": [USER, ASSISTANT]}),
        allow_pretokenized=True,
        require_pretokenized=True,
    )
    with pytest.raises(DojoConfigError, match=r"needs metadata\.input_ids"):
        list(dataset.rows())


def test_pretokenized_custom_metadata_column_and_cache_isolation():
    class CachedPairDataset(PairDataset):
        def cache_key(self):
            return "same-data"

    inner = CachedPairDataset({"encoded": {"input_ids": [1, 2], "loss_mask": [0, 1]}})
    dataset = _SftDataset(inner, allow_pretokenized=True, metadata_key="encoded")
    assert list(dataset.rows())[0]["messages"] == []
    assert (
        len(
            {
                dataset.cache_key(),
                _SftDataset(inner).cache_key(),
                _SftDataset(inner, allow_pretokenized=True).cache_key(),
                _SftDataset(
                    inner, allow_pretokenized=True, require_pretokenized=True
                ).cache_key(),
            }
        )
        == 4
    )


def test_slime_does_not_accept_token_only_rows_without_a_supported_rollout():
    with pytest.raises(DojoConfigError, match="missing a prompt"):
        list(
            _SftDataset(
                PairDataset({"metadata": {"input_ids": [1, 2], "loss_mask": [0, 1]}})
            ).rows()
        )


def test_sft_rejects_eval_dataset():
    with pytest.raises(DojoConfigError, match="eval_dataset"):
        BaseTrainRecipe._validate_datasets(
            RowsDataset("a"), RowsDataset("b"), loss_type="sft_loss"
        )


def test_failed_write_cleanup_preserves_committed_destination(tmp_path):
    path = str(tmp_path / "train.jsonl")

    class RaceDataset(RowsDataset):
        def write(self, dest: str) -> None:
            self.write_count += 1
            RowsDataset("peer", "peer").write(path)
            raise DojoConfigError("boom")

    with pytest.raises(DojoConfigError, match="boom"):
        write_dataset_if_needed(RaceDataset("train"), path)
    assert json.loads((tmp_path / "train.jsonl").read_text()) == {
        "prompt": "peer",
        "label": "peer",
    }
    assert not any(p.name.endswith(".tmp.jsonl") for p in tmp_path.iterdir())
