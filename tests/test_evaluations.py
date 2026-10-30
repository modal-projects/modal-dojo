import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from modal_dojo.common import evaluations
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.evaluations import ModalEvalConfig, due_checkpoints


def test_missing_sdk_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "modal_evals", None)
    with pytest.raises(DojoConfigError, match="modal-evals"):
        evaluations.run_evaluation("example", object())  # type: ignore[arg-type]


def test_modal_eval_config_needs_a_positive_interval() -> None:
    assert ModalEvalConfig(every_n_checkpoints=3).reward_function is None
    with pytest.raises(ValidationError):
        ModalEvalConfig(every_n_checkpoints=0)
    with pytest.raises(ValidationError):
        ModalEvalConfig(every_n_checkpoints=1, scorer="exact_match")  # type: ignore[call-arg]


def test_train_config_takes_eval() -> None:
    from modal_dojo.common.train import TrainConfig

    assert TrainConfig.__dataclass_fields__["eval"].default is None


@pytest.mark.parametrize(
    ("names", "evaluated", "finished", "due"),
    [
        (["i1", "i2", "i3", "i4", "i5"], set(), False, ["i2", "i4"]),
        (["i1", "i2", "i3", "i4", "i5"], {"i2"}, False, ["i4"]),
        (["i1", "i2", "i3", "i4", "i5"], {"i2", "i4"}, True, ["i5"]),
        (["i1", "i2"], {"i2"}, True, []),
        ([], set(), True, []),
    ],
)
def test_due_checkpoints(
    names: list[str], evaluated: set[str], finished: bool, due: list[str]
) -> None:
    config = ModalEvalConfig(every_n_checkpoints=2)
    assert due_checkpoints(names, evaluated, config, finished=finished) == due


def test_due_checkpoints_skips_final_when_disabled() -> None:
    config = ModalEvalConfig(every_n_checkpoints=2, evaluate_final=False)
    assert due_checkpoints(["i1", "i2", "i3"], {"i2"}, config, finished=True) == []


def test_eval_rows_default_to_rows() -> None:
    from modal_dojo.common.dataset import DatasetConfig, HuggingFaceDataset

    class Rows(DatasetConfig):
        def input_key(self) -> str:
            return "q"

        def label_key(self) -> str:
            return "a"

        def rows(self):
            return [{"q": "1", "a": "1"}]

    assert list(Rows().eval_rows() or ()) == [{"q": "1", "a": "1"}]
    dataset = HuggingFaceDataset("org/data", input_column="q", eval_split="test")
    assert dataset.eval_split == "test"
    assert HuggingFaceDataset("org/data", input_column="q").eval_split is None


def test_training_run_evaluations_reads_metadata() -> None:
    from modal_dojo.common.run import TrainingRun

    run = TrainingRun.model_construct(
        metadata={"modal_evals": [{"iteration": "iter_0000002", "score": 0.5}]}
    )
    assert run.evaluations() == [{"iteration": "iter_0000002", "score": 0.5}]
    assert TrainingRun.model_construct(metadata={}).evaluations() == []


def test_modal_eval_requires_modal_labs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluations, "_require_modal_evals", lambda: None)
    monkeypatch.setattr(evaluations, "_workspace_name", lambda: "acme")
    train_config = SimpleNamespace(eval=ModalEvalConfig(every_n_checkpoints=1))
    with pytest.raises(DojoConfigError, match="modal-labs"):
        evaluations.prepare_eval(train_config)  # type: ignore[arg-type]


async def _reward(args, sample, **kwargs) -> float:
    return float(sample.response == sample.label)


def _train_config(recipe_reward=None, **eval):
    return SimpleNamespace(
        eval=ModalEvalConfig(every_n_checkpoints=1, max_rows=10, **eval),
        recipe=SimpleNamespace(custom_rm_function=recipe_reward),
        dataset="train",
        eval_dataset="held-out",
    )


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> dict:
    published: dict = {}
    monkeypatch.setattr(evaluations, "_require_modal_evals", lambda: None)
    monkeypatch.setattr(evaluations, "_workspace_name", lambda: "modal-labs")
    monkeypatch.setattr(
        evaluations,
        "publish_dataset_evaluation",
        lambda dataset, **kwargs: published.update(dataset=dataset, **kwargs),
    )
    return published


def test_modal_eval_scores_with_the_recipe_reward(published: dict) -> None:
    evaluations.prepare_eval(_train_config(_reward))  # type: ignore[arg-type]
    assert published == {
        "dataset": "held-out",
        "reward_function": _reward,
        "limit": 10,
    }


def test_modal_eval_reward_function_overrides_the_recipe(published: dict) -> None:
    def other(args, sample):
        return 0.0

    evaluations.prepare_eval(_train_config(_reward, reward_function=other))  # type: ignore[arg-type]
    assert published["reward_function"] is other


def test_modal_eval_needs_a_reward_function(published: dict) -> None:
    with pytest.raises(DojoConfigError, match="custom_rm_function"):
        evaluations.prepare_eval(_train_config(None))  # type: ignore[arg-type]


def test_evaluate_checkpoint_records_and_stops_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from modal_dojo.common import endpoint as endpoint_module

    events = []

    class FakeEndpoint:
        @classmethod
        def launch(cls, model, checkpoint, *, environment):
            events.append(("launch", checkpoint.name))
            return cls()

        def wait_until_ready(self):
            events.append(("ready",))

        def stop(self):
            events.append(("stop",))

    class FakeJob:
        job_id = "ej-1"

        def wait(self):
            return SimpleNamespace(
                job=SimpleNamespace(status="succeeded"),
                output_volume=SimpleNamespace(url="https://modal.com/id/vo"),
            )

        def mean(self, field, results):
            return 0.75

    recorded = []
    monkeypatch.setattr(endpoint_module, "Endpoint", FakeEndpoint)
    monkeypatch.setattr(
        evaluations,
        "run_evaluation",
        lambda release_id, target, **kwargs: (
            events.append(("run", kwargs["tags"])) or FakeJob()
        ),
    )
    monkeypatch.setattr(
        evaluations,
        "_record_evaluation",
        lambda run_id, record: recorded.append(record),
    )
    record = evaluations.evaluate_checkpoint(
        training_run_id="run-1",
        checkpoint=SimpleNamespace(name="iter_0000004"),
        release_id="er-1",
        config=ModalEvalConfig(every_n_checkpoints=2),
        model=SimpleNamespace(),  # type: ignore[arg-type]
        environment=None,
    )
    assert record == {
        "iteration": "iter_0000004",
        "job_id": "ej-1",
        "status": "succeeded",
        "score": 0.75,
        "url": "https://modal.com/id/vo",
    }
    assert recorded == [record]
    assert events[-1] == ("stop",)
    assert ("run", {"training_run_id": "run-1", "checkpoint": "iter_0000004"}) in events


def test_mean_averages_finished_trials() -> None:
    def trial(status: str, result: dict | None) -> SimpleNamespace:
        return SimpleNamespace(status=status, result=result)

    results = SimpleNamespace(
        tasks=[
            SimpleNamespace(
                trials=[
                    trial("succeeded", {"score": 1.0}),
                    trial("succeeded", {"score": 0.0}),
                ]
            ),
            SimpleNamespace(
                trials=[trial("succeeded", {"score": 1}), trial("failed", None)]
            ),
        ]
    )
    job = evaluations.EvaluationJob("ej-1")
    assert job.mean("score", results) == pytest.approx(2 / 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="latency"):
        job.mean("latency", results)  # type: ignore[arg-type]
