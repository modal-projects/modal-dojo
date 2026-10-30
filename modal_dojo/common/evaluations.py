"""Run [Modal Evals](https://modal.com/evaluations) evaluations against Dojo
deployments.

Modal Evals publishes evaluations (typed tasks, an async trial function, and a
report) and runs them as jobs against a target: a Modal Endpoint, or an
OpenAI-compatible URL. This module turns Dojo deployment handles into targets,
submits jobs, and reads their results.

The integration is internal-only while the ``modal-evals`` SDK is unpublished.
Install it from a modal-labs/modal checkout::

    uv pip install --no-sources <modal>/flash_eval/modal_evals

Example::

    import modal_dojo as dojo

    endpoint = dojo.Endpoint.launch(...)
    job = dojo.run_evaluation("terminal-bench-2-1", endpoint, wait=True)
    print(job.mean("reward"))
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Coroutine, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.errors import DojoConfigError

if TYPE_CHECKING:
    from modal_evals import Task, TaskData
    from modal_evals.client import FlashEvalClient
    from modal_evals.proto import flash_eval_api_pb2
    from modal_evals.results import JobResults
    from modal_evals.targets import CustomTarget, FlashEndpointTarget, RecipeTarget

    from modal_dojo.common.dataset import DatasetConfig
    from modal_dojo.common.deployment import CustomDeployment
    from modal_dojo.common.endpoint import Endpoint

    EvaluationTargetLike = (
        Endpoint | CustomDeployment | FlashEndpointTarget | CustomTarget | RecipeTarget
    )

T = TypeVar("T")

INSTALL_HINT = (
    "Modal Evals is not installed. It is internal-only until modal-evals is "
    "published: install it from a modal-labs/modal checkout with "
    "`uv pip install --no-sources <modal>/flash_eval/modal_evals`."
)

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
_POLL_SECONDS = 5.0
_MAX_POLL_SECONDS = 30.0


def _require_modal_evals() -> None:
    try:
        import modal_evals  # noqa: F401
    except ImportError as exc:
        raise DojoConfigError(INSTALL_HINT) from exc


class _BackgroundLoop:
    """One event loop for every call, since the Modal client it opens is bound
    to the loop that created it."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                threading.Thread(
                    target=self._loop.run_forever,
                    name="modal-dojo-evaluations",
                    daemon=True,
                ).start()
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()


_LOOP = _BackgroundLoop()


async def _client(server: str | None) -> FlashEvalClient:
    from modal_evals.client import FlashEvalClient

    return await FlashEvalClient.from_env(server)


async def _active_deployment_id(endpoint_id: str, environment: str | None) -> str:
    """The deployment an Endpoint serves, as `modal endpoint list` reads it."""
    from modal_evals.modal_client import _Client
    from modal_proto import api_pb2

    client = await _Client.from_env()
    created_before = time.time()
    while True:
        response = await client.stub.EndpointList(
            api_pb2.EndpointListRequest(
                environment_name=environment or "",
                pagination=api_pb2.ListPagination(
                    max_objects=100, created_before=created_before
                ),
            )
        )
        for item in response.items:
            if item.endpoint_id == endpoint_id:
                if not item.active_deployment_id:
                    raise DojoConfigError(
                        f"Endpoint {endpoint_id} has no active deployment to evaluate."
                    )
                return item.active_deployment_id
        if len(response.items) < 100:
            raise DojoConfigError(
                f"Endpoint {endpoint_id} is not in environment "
                f"{environment or 'default'!r}; pass environment= or deployment_id=."
            )
        created_before = response.items[-1].metadata.creation_info.created_at


def _custom_target(deployment: CustomDeployment) -> CustomTarget:
    from modal_evals.targets import CustomTarget

    if not deployment.unauthenticated:
        raise DojoConfigError(
            "Modal Evals reaches custom targets with a bearer token only, so it "
            "cannot pass Modal proxy auth. Launch the deployment with "
            "CustomDeployment.launch(..., unauthenticated=True) to evaluate it."
        )
    return CustomTarget(
        base_url=f"{deployment.url.rstrip('/')}/v1",
        model=deployment.served_model_name,
    )


async def _target_proto(
    target: EvaluationTargetLike,
    *,
    deployment_id: str | None,
    single_replica: bool,
    environment: str | None,
) -> flash_eval_api_pb2.EvaluationTarget:
    from modal_evals.publish import evaluation_target_to_proto
    from modal_evals.targets import CustomTarget, FlashEndpointTarget, RecipeTarget

    from modal_dojo.common.deployment import CustomDeployment
    from modal_dojo.common.endpoint import Endpoint

    resolved: FlashEndpointTarget | CustomTarget | RecipeTarget
    if isinstance(target, Endpoint):
        resolved = FlashEndpointTarget(
            endpoint_id=target.endpoint_id,
            deployment_id=deployment_id
            or await _active_deployment_id(target.endpoint_id, environment),
            single_replica=single_replica,
        )
    elif isinstance(target, CustomDeployment):
        if single_replica:
            raise ValueError("single_replica applies only to Modal Endpoints.")
        resolved = _custom_target(target)
    elif isinstance(target, (FlashEndpointTarget, CustomTarget, RecipeTarget)):
        resolved = target
    else:
        raise TypeError(
            "target must be an Endpoint, a CustomDeployment, or a modal_evals "
            f"target, not {type(target).__name__}"
        )
    return evaluation_target_to_proto(resolved)


async def _release_id(
    client: FlashEvalClient, environment_id: str, evaluation: str
) -> str:
    from modal_evals.proto import flash_eval_api_pb2

    if evaluation.startswith("er-"):
        return evaluation
    response = await client.list_releases(
        flash_eval_api_pb2.ListEvaluationReleasesRequest(environment_id=environment_id)
    )
    matches = [r for r in response.releases if r.evaluation_key == evaluation]
    if not matches:
        raise ValueError(
            f"No evaluation {evaluation!r} can run in this environment: none is "
            "published here or globally."
        )
    if len(matches) > 1:
        choices = ", ".join(r.release_id for r in matches)
        raise ValueError(
            f"Several evaluations are named {evaluation!r}; pass a release ID: {choices}"
        )
    return matches[0].release_id


@exclude_from_api_reference
@dataclass(frozen=True)
class EvaluationJob:
    """A Modal Evals job.

    Attributes:
        job_id: Modal Evals job ID.
        environment: Modal environment the job runs in, or the default one.
        server: Modal Evals API URL, or the default one.
    """

    job_id: str
    environment: str | None = None
    server: str | None = None

    async def _get(self) -> tuple[Any, str | None]:
        from modal_evals.proto import flash_eval_api_pb2

        client = await _client(self.server)
        try:
            response = await client.get_job(
                flash_eval_api_pb2.GetEvaluationJobRequest(
                    environment_id=await client.environment_id(self.environment),
                    job_id=self.job_id,
                )
            )
            volume = await client.get_output_volume(response)
        finally:
            await client.close()
        return response, volume.object_id if volume is not None else None

    def results(self) -> JobResults:
        """The job's results so far: every task, trial, and attempt, with each
        trial's typed result values."""
        _require_modal_evals()
        from modal_evals.results import JobResults

        response, volume_id = _LOOP.run(self._get())
        return JobResults.from_response(response, output_volume_id=volume_id)

    def status(self) -> str:
        """``pending``, ``running``, ``succeeded``, ``failed``, ``cancelling``,
        or ``cancelled``."""
        return self.results().job.status

    def wait(self, timeout: float | None = None) -> JobResults:
        """Wait for the job to finish and return its results.

        Raises:
            TimeoutError: The job is still running after ``timeout`` seconds.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        delay = _POLL_SECONDS
        while True:
            results = self.results()
            if results.job.status in _TERMINAL_STATUSES:
                return results
            if deadline is not None and time.monotonic() + delay > deadline:
                raise TimeoutError(
                    f"Evaluation job {self.job_id} is still {results.job.status}."
                )
            time.sleep(delay)
            delay = min(delay * 2, _MAX_POLL_SECONDS)

    def cancel(self) -> None:
        """Cancel the job's unfinished trials."""
        _require_modal_evals()
        from modal_evals.proto import flash_eval_api_pb2

        async def cancel() -> None:
            client = await _client(self.server)
            try:
                await client.cancel_job(
                    flash_eval_api_pb2.GetEvaluationJobRequest(
                        environment_id=await client.environment_id(self.environment),
                        job_id=self.job_id,
                    )
                )
            finally:
                await client.close()

        _LOOP.run(cancel())

    def mean(self, field: str, results: JobResults | None = None) -> float:
        """Mean of a numeric or boolean result field over finished trials.

        Raises:
            ValueError: No finished trial reports ``field``.
        """
        results = results or self.results()
        values: list[float] = []
        for task in results.tasks:
            for trial in task.trials:
                if trial.status != "succeeded" or trial.result is None:
                    continue
                value = trial.result.get(field)
                if isinstance(value, (bool, int, float)):
                    values.append(float(value))
        if not values:
            raise ValueError(
                f"No finished trial of job {self.job_id} reports {field!r}."
            )
        return sum(values) / len(values)


@exclude_from_api_reference
def run_evaluation(
    evaluation: str,
    target: EvaluationTargetLike,
    *,
    params: Mapping[str, Any] | None = None,
    task_keys: Iterable[str] = (),
    trials_per_task: int | None = None,
    trial_concurrency: int | None = None,
    timeout_multiplier: float | None = None,
    secrets: Iterable[str] = (),
    name: str = "",
    tags: Mapping[str, str] | None = None,
    deployment_id: str | None = None,
    single_replica: bool = False,
    environment: str | None = None,
    server: str | None = None,
    wait: bool = False,
    timeout: float | None = None,
) -> EvaluationJob:
    """Run a published Modal Evals evaluation against a Dojo deployment.

    Args:
        evaluation:
            Evaluation key, such as ``terminal-bench-2-1``, whose latest
            release runs, or a release ID (``er-...``).
        target:
            An ``Endpoint``, an unauthenticated ``CustomDeployment``, or a
            ``modal_evals`` target.
        params:
            Evaluation parameters, as the evaluation's ``Params`` declares.
        task_keys:
            Tasks to run; empty runs every task.
        trials_per_task:
            Trials of each task; unset uses the release's default.
        trial_concurrency:
            Trials at once; unset uses the release's ``max_concurrency``.
        timeout_multiplier:
            Factor applied to each task's time limit.
        secrets:
            Modal Secrets of the environment to attach to trials.
        name:
            Job name shown in the Modal dashboard.
        tags:
            Labels for the job, such as a training run ID.
        deployment_id:
            Endpoint deployment (``ed-...``) to evaluate; defaults to the
            Endpoint's active deployment.
        single_replica:
            Evaluate a one-container replica of the Endpoint's deployment.
        environment:
            Modal environment to run in; defaults to the configured one.
        server:
            Modal Evals API URL; defaults to the configured one.
        wait:
            Wait for the job to finish before returning.
        timeout:
            With ``wait``, the most seconds to wait.

    Returns:
        The submitted job.
    """
    _require_modal_evals()
    from google.protobuf.json_format import ParseDict
    from google.protobuf.struct_pb2 import Struct
    from modal_evals.proto import flash_eval_api_pb2

    async def submit() -> str:
        client = await _client(server)
        try:
            environment_id = await client.environment_id(environment)
            request = flash_eval_api_pb2.SubmitEvaluationJobRequest(
                environment_id=environment_id,
                idempotency_key=f"modal-dojo-{uuid.uuid4()}",
                release_id=await _release_id(client, environment_id, evaluation),
                target=await _target_proto(
                    target,
                    deployment_id=deployment_id,
                    single_replica=single_replica,
                    environment=environment,
                ),
                params=ParseDict(dict(params or {}), Struct()),
                task_keys=list(task_keys),
                trials_per_task=trials_per_task,
                trial_concurrency=trial_concurrency,
                timeout_multiplier=timeout_multiplier,
                secret_names=list(secrets),
                name=name,
                tags=dict(tags or {}),
            )
            return (await client.submit_job(request)).job_id
        finally:
            await client.close()

    job = EvaluationJob(_LOOP.run(submit()), environment=environment, server=server)
    if wait:
        job.wait(timeout)
    return job


@exclude_from_api_reference
def dataset_tasks(
    dataset: DatasetConfig,
    task_data: type[TaskData],
    *,
    columns: Mapping[str, str] | None = None,
    key_column: str | None = None,
    limit: int | None = None,
) -> list[Task[Any]]:
    """Modal Evals tasks from a Dojo dataset's rows, for an evaluation's
    ``@evaluation.tasks()`` function.

    Args:
        dataset:
            Dataset whose ``rows()`` become tasks.
        task_data:
            The evaluation's ``TaskData`` subclass. Each field reads the row
            column of the same name unless ``columns`` maps it.
        columns:
            Field name to row column, such as
            ``{"prompt": dataset.input_key()}``.
        key_column:
            Column holding each task's key; defaults to ``row-<index>``.
        limit:
            Most rows to read.
    """
    _require_modal_evals()
    from modal_evals import Task

    columns = dict(columns or {})
    tasks: list[Task[Any]] = []
    for index, row in enumerate(dataset.rows()):
        if limit is not None and index >= limit:
            break
        values = {
            field: row[columns.get(field, field)]
            for field in task_data.model_fields
            if columns.get(field, field) in row
        }
        key = str(row[key_column]) if key_column else f"row-{index:06d}"
        tasks.append(Task(key, task_data(**values)))
    return tasks
