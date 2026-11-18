import json
from pathlib import Path
from types import SimpleNamespace

import httpx
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
        evaluations._resolve_target(target, **(options | kwargs))
    )


def test_custom_deployment_becomes_openai_compatible_target() -> None:
    target = _target(_deployment(unauthenticated=True))
    assert isinstance(target, modal_evals.CustomTarget)
    assert target.base_url == "https://example--serve.modal.run/v1"
    assert target.model == "my-model"
    assert target.api_key is None


def test_proxy_auth_custom_deployment_is_refused() -> None:
    with pytest.raises(DojoConfigError, match="unauthenticated=True"):
        _target(_deployment(unauthenticated=False))


def test_endpoint_becomes_modal_endpoint_target() -> None:
    from modal_dojo.common.endpoint import Endpoint

    endpoint = Endpoint(
        "https://example.modal.run",
        endpoint_id="ep-abc",
        model_name="m",
        requires_proxy_auth=True,
    )
    target = _target(endpoint, deployment_id="ed-def", single_replica=True)
    assert isinstance(target, modal_evals.ModalEndpointTarget)
    assert target.endpoint_id == "ep-abc"
    assert target.deployment_id == "ed-def"
    assert target.single_replica


async def _reward(args, sample, **kwargs) -> float:
    return float(sample.response.strip() == sample.label)


def test_score_calls_rewards_as_training_does() -> None:
    from modal_dojo.evaluation_runner.dataset import (
        pickle_reward,
        score,
        _reward as load,
    )

    reward = load(pickle_reward(_reward))
    assert evaluations._LOOP.run(score(reward, [], " 2 ", "2")) == 1.0
    assert evaluations._LOOP.run(score(lambda a, s: {"reward": 0.5}, [], "", "")) == 0.5
    assert evaluations._LOOP.run(score(lambda a, s: True, [], "", "")) == 1.0


def test_task_from_row_wraps_prompts_as_messages() -> None:
    from modal_dojo.evaluation_runner.dataset import task_from_row

    task = task_from_row(
        3, {"q": "1+1?", "a": 2}, input_key="q", label_key="a", reward="r"
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
        reward="r",
        key_column="id",
    )
    assert task.key == "q-7"
    assert json.loads(task.data.messages) == messages


def test_dataset_trial_uses_bound_inference_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from modal_evals.local import test_evaluation as run_local_evaluation
    from modal_evals.protocol import TrialSucceeded

    from modal_dojo.evaluation_runner import dataset

    def complete(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.test/v1/chat/completions"
        assert json.loads(request.content)["model"] == "test-model"
        return httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})

    monkeypatch.setattr(
        dataset,
        "httpx",
        SimpleNamespace(
            AsyncClient=lambda **kwargs: httpx.AsyncClient(
                transport=httpx.MockTransport(complete), **kwargs
            )
        ),
    )
    task = dataset.task_from_row(
        0,
        {"q": "1+1?", "a": "2"},
        input_key="q",
        label_key="a",
        reward=dataset.pickle_reward(_reward),
    )
    evaluation = dataset.build_evaluation(
        "dojo-example",
        [task],
        display_name="Arithmetic",
        description="Score an arithmetic answer.",
        max_concurrency=1,
    )
    target = modal_evals.CustomTarget(
        base_url="https://example.test/v1", model="test-model"
    )
    run = evaluations._LOOP.run(
        run_local_evaluation(
            evaluation.definition(),
            task_keys=(),
            all_tasks=False,
            params={"inference_target": target.model_dump(mode="json")},
            output_root=tmp_path,
            seed=0,
        )
    )
    outcome = run.outcomes[task.key]
    assert isinstance(outcome, TrialSucceeded), outcome
    assert outcome.result["score"] == 1.0


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
    evaluations.publish_dataset_evaluation(_Dataset(), reward_function=_reward, limit=2)  # type: ignore[arg-type]
    definition, tasks = published[-1]
    assert definition.key.startswith("dojo-")
    assert definition.runner.entrypoint == "modal_dojo.evaluation_runner.dataset:run"
    assert definition.runner.package == "modal_dojo"
    assert [task.key for task in tasks] == ["row-000000", "row-000001"]

    evaluations.publish_dataset_evaluation(_Dataset(), reward_function=_reward, limit=2)  # type: ignore[arg-type]
    assert published[-1][0].key == definition.key
    evaluations.publish_dataset_evaluation(_Dataset(), reward_function=_reward, limit=3)  # type: ignore[arg-type]
    assert published[-1][0].key != definition.key


def test_publish_requires_a_label() -> None:
    class Unlabeled(BaseModel):
        def input_key(self) -> str:
            return "q"

        def label_key(self) -> None:
            return None

    with pytest.raises(ValueError, match="label_key"):
        evaluations.publish_dataset_evaluation(Unlabeled(), reward_function=_reward)  # type: ignore[arg-type]


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
    assert request.params["inference_target"]["kind"] == "custom"
    assert request.params["inference_target"]["model"] == "my-model"


def test_finished_job_scores_trials_from_task_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from modal_evals.proto import flash_eval_api_pb2 as api

    class ResultsClient(_FakeClient):
        async def get_job(self, request):
            return api.GetEvaluationJobResponse(
                job=api.EvaluationJob(
                    job_id=request.job_id, status=api.EVALUATION_STATUS_SUCCEEDED
                )
            )

        async def list_tasks(self, request):
            assert request.job_id == "ej-1"
            return [
                api.EvaluationTask(
                    task_id="task-1",
                    task_key="row-000000",
                    trials=[
                        api.EvaluationTrial(
                            trial_id="trial-1",
                            status=api.EVALUATION_STATUS_SUCCEEDED,
                            result={"score": 0.75},
                        )
                    ],
                )
            ]

        async def get_output_volume(self, response):
            return SimpleNamespace(object_id="vo-1")

    async def client(server):
        return ResultsClient()

    monkeypatch.setattr(evaluations, "_client", client)
    job = evaluations.EvaluationJob("ej-1")
    results = job.wait()
    assert job.mean("score", results) == 0.75
    assert results.output_volume.url == "https://modal.com/id/vo-1"
