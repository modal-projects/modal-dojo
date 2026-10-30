import sys
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from modal_dojo.common import evaluations
from modal_dojo.common.errors import DojoConfigError


def test_missing_sdk_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "modal_evals", None)
    with pytest.raises(DojoConfigError, match="modal-evals"):
        evaluations.run_evaluation("example", object())  # type: ignore[arg-type]


def test_mean_averages_finished_trials() -> None:
    def trial(status: str, result: dict | None) -> SimpleNamespace:
        return SimpleNamespace(status=status, result=result)

    results = SimpleNamespace(
        tasks=[
            SimpleNamespace(
                trials=[
                    trial("succeeded", {"correct": True}),
                    trial("succeeded", {"correct": False}),
                ]
            ),
            SimpleNamespace(
                trials=[trial("succeeded", {"correct": 1.0}), trial("failed", None)]
            ),
        ]
    )
    job = evaluations.EvaluationJob("ej-1")
    assert job.mean("correct", results) == pytest.approx(2 / 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="latency"):
        job.mean("latency", results)  # type: ignore[arg-type]


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


def test_dataset_tasks_maps_rows_to_task_data() -> None:
    class Row(modal_evals.TaskData):
        prompt: str
        answer: str

    class Dataset(BaseModel):
        def rows(self):
            yield {"question": "1+1", "answer": "2", "extra": 0}
            yield {"question": "2+2", "answer": "4", "extra": 1}

    tasks = evaluations.dataset_tasks(
        Dataset(),  # type: ignore[arg-type]
        Row,
        columns={"prompt": "question"},
        limit=1,
    )
    assert [task.key for task in tasks] == ["row-000000"]
    assert tasks[0].data == Row(prompt="1+1", answer="2")


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
    assert request.environment_id == "en-1"
    assert request.params["temperature"] == 0
    assert dict(request.tags) == {"training_run_id": "run-1"}
    assert request.trial_concurrency == 4
    assert request.target.custom.model == "my-model"
