import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from modal_dojo.common import evaluations
from modal_dojo.common.errors import DojoConfigError

modal_evals = pytest.importorskip("modal_evals")


def _deployment(*, unauthenticated: bool):
    from modal_dojo.common.deployment import CustomDeployment

    return CustomDeployment.model_construct(
        deployment_id="dep",
        app_name="app",
        served_model_name="my-model",
        unauthenticated=unauthenticated,
        url="https://example--serve.modal.run/",
    )


def _target(target, **kwargs):
    options = {"deployment_id": None, "single_replica": False, "environment": None}
    return evaluations._LOOP.run(
        evaluations._target_proto(target, **(options | kwargs))
    )


def test_custom_deployment_becomes_openai_compatible_target() -> None:
    proto = _target(_deployment(unauthenticated=True))
    assert proto.custom.base_url == "https://example--serve.modal.run/v1"
    assert proto.custom.model == "my-model"
    assert not proto.custom.HasField("api_key")


def test_proxy_auth_custom_deployment_is_refused() -> None:
    with pytest.raises(DojoConfigError, match="unauthenticated=True"):
        _target(_deployment(unauthenticated=False))


def test_endpoint_becomes_flash_endpoint_target() -> None:
    from modal_dojo.common.endpoint import Endpoint

    endpoint = Endpoint(
        "https://example.modal.run",
        endpoint_id="ep-abc",
        model_name="m",
        requires_proxy_auth=True,
    )
    proto = _target(endpoint, deployment_id="ed-def", single_replica=True)
    assert proto.flash_endpoint.endpoint_id == "ep-abc"
    assert proto.flash_endpoint.deployment_id == "ed-def"
    assert proto.flash_endpoint.single_replica


@pytest.mark.parametrize(
    ("scorer", "response", "label", "score"),
    [
        ("exact_match", "  Paris ", "paris", 1.0),
        ("exact_match", "Paris, France", "paris", 0.0),
        ("contains", "The capital is Paris.", "paris", 1.0),
        ("boxed_math", "so the total is \\boxed{1,000}", "1000", 1.0),
        ("boxed_math", "#### 42", "The answer is 42", 1.0),
        ("boxed_math", "I think 41", "42", 0.0),
    ],
)
def test_scorers(scorer: str, response: str, label: str, score: float) -> None:
    from modal_dojo.evaluation_runner.dataset import SCORERS

    assert SCORERS[scorer](response, label) == score


def test_task_from_row_wraps_prompts_as_messages() -> None:
    from modal_dojo.evaluation_runner.dataset import task_from_row

    task = task_from_row(
        3, {"q": "1+1?", "a": 2}, input_key="q", label_key="a", scorer="boxed_math"
    )
    assert task.key == "row-000003"
    assert json.loads(task.data.messages) == [{"role": "user", "content": "1+1?"}]
    assert task.data.label == "2"
    messages = [{"role": "system", "content": "Be brief."}]
    task = task_from_row(
        0,
        {"id": "q-7", "q": messages, "a": "x"},
        input_key="q",
        label_key="a",
        scorer="exact_match",
        key_column="id",
    )
    assert task.key == "q-7"
    assert json.loads(task.data.messages) == messages


class _Dataset(BaseModel):
    count: int = 5

    def input_key(self) -> str:
        return "question"

    def label_key(self) -> str:
        return "answer"

    def eval_rows(self):
        for index in range(self.count):
            yield {"question": f"1+{index}?", "answer": str(1 + index)}


def test_publish_dataset_evaluation_builds_private_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = []

    async def publish(evaluation, environment, server):
        definition = evaluation.definition()
        published.append((definition, await definition.materialize_tasks()))
        return evaluations.EvaluationRelease("er-1", evaluation.key, task_count=2)

    monkeypatch.setattr(evaluations, "_publish", publish)
    evaluations.publish_dataset_evaluation(_Dataset(), scorer="boxed_math", limit=2)  # type: ignore[arg-type]
    definition, tasks = published[-1]
    assert definition.key.startswith("dojo-")
    assert definition.runner.entrypoint == "modal_dojo.evaluation_runner.dataset:run"
    assert definition.runner.package == "modal_dojo.evaluation_runner"
    assert [task.key for task in tasks] == ["row-000000", "row-000001"]

    evaluations.publish_dataset_evaluation(_Dataset(), scorer="boxed_math", limit=2)  # type: ignore[arg-type]
    assert published[-1][0].key == definition.key
    evaluations.publish_dataset_evaluation(_Dataset(), scorer="contains", limit=2)  # type: ignore[arg-type]
    assert published[-1][0].key != definition.key


def test_publish_requires_a_label() -> None:
    class Unlabeled(BaseModel):
        def input_key(self) -> str:
            return "q"

        def label_key(self) -> None:
            return None

    with pytest.raises(ValueError, match="label_key"):
        evaluations.publish_dataset_evaluation(Unlabeled())  # type: ignore[arg-type]


class _FakeClient:
    submitted: list = []

    async def environment_id(self, environment):
        return "en-1"

    async def list_releases(self, request):
        release = SimpleNamespace(evaluation_key="example", release_id="er-1")
        return SimpleNamespace(releases=[release])

    async def submit_job(self, request):
        self.submitted.append(request)
        return SimpleNamespace(job_id="ej-1")

    async def close(self):
        pass


def test_run_evaluation_submits_latest_release(monkeypatch: pytest.MonkeyPatch) -> None:
    async def client(server):
        return _FakeClient()

    monkeypatch.setattr(evaluations, "_client", client)
    job = evaluations.run_evaluation(
        "example",
        _deployment(unauthenticated=True),
        params={"temperature": 0},
        tags={"training_run_id": "run-1"},
        trial_concurrency=4,
    )
    assert job.job_id == "ej-1"
    request = _FakeClient.submitted[-1]
    assert request.release_id == "er-1"
    assert request.params["temperature"] == 0
    assert dict(request.tags) == {"training_run_id": "run-1"}
    assert request.trial_concurrency == 4
    assert request.target.custom.model == "my-model"
