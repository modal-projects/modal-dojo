from types import SimpleNamespace

from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.models import Qwen3_4B
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.common.training_rollout import TrainingRolloutResult
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe


def _dataset(*, input_format: str, input_column: str) -> HuggingFaceDataset:
    return HuggingFaceDataset(
        hf_repo="some/dataset",
        input_column=input_column,
        output_column="answer",
        input_format=input_format,
    )


def test_evaluate_forces_eval_only_recipe_on_given_dataset(monkeypatch) -> None:
    train_dataset = _dataset(input_format="messages", input_column="prompt")
    eval_dataset = _dataset(input_format="text", input_column="question")
    assert train_dataset.input_key() != eval_dataset.input_key()

    config = TrainConfig(
        dataset=train_dataset,
        model=Qwen3_4B(),
        recipe=SlimeRecipe(
            num_epoch=3,
            num_rollout=10,
            eval_interval=50,
            n_samples_per_eval_prompt=2,
            extra_config={
                "num_rollout": 99,
                "eval_interval": 7,
                "qkv_format": "bshd",
            },
        ),
    )

    captured: list[TrainConfig] = []

    def fake_train(self, *, show_output: bool = True):
        captured.append(self)
        return SimpleNamespace(training_run_id="eval-run")

    monkeypatch.setattr(TrainConfig, "train", fake_train)
    monkeypatch.setattr(
        "modal_training_gym.common.train.vol_get",
        lambda *_args, **_kwargs: TrainingRolloutResult(
            training_run_id="eval-run",
            rollout_id=0,
            samples=[],
        ).model_dump(),
    )

    n_samples = 5
    assert config.evaluate(eval_dataset, n_samples) == []
    assert len(captured) == 1
    eval_config = captured[0]
    assert eval_config.dataset is eval_dataset
    assert eval_config.eval_dataset is eval_dataset

    fields = eval_config.recipe._fields(
        dataset=eval_config.dataset,
        eval_dataset=eval_config.eval_dataset,
    )
    assert fields["num_rollout"] == 0
    assert fields.get("num_epoch") is None
    assert fields["eval_interval"] == 1
    assert fields["n_samples_per_eval_prompt"] == n_samples
    assert fields["input_key"] == eval_dataset.input_key()
