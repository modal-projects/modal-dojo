"""Score training checkpoints with [Modal Evals](https://modal.com/evaluations).

``TrainConfig(..., modal_evals=ModalEvalsConfig(checkpoint_interval=N))``
publishes the dataset's ``eval_rows()`` (or ``eval_dataset.rows()``) as a
private Modal Evals evaluation when the run launches, then evaluates every Nth
checkpoint: a watcher serves it on a Modal Endpoint, runs the evaluation
against it, and stops the Endpoint.

Modal Evals is internal-only while the ``modal-evals`` SDK is unpublished, so
this needs the modal-labs workspace and the SDK installed from a
modal-labs/modal checkout::

    uv pip install --no-sources <modal>/flash_eval/modal_evals
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
import uuid
from collections.abc import Coroutine, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.errors import DojoConfigError

if TYPE_CHECKING:
    from modal_evals.client import FlashEvalClient
    from modal_evals.proto import flash_eval_api_pb2
    from modal_evals.results import JobResults
    from modal_evals.targets import CustomTarget, FlashEndpointTarget, RecipeTarget

    from modal_dojo.common.dataset import DatasetConfig
    from modal_dojo.common.deployment import CustomDeployment
    from modal_dojo.common.endpoint import Endpoint
    from modal_dojo.common.models import ModelConfig
    from modal_dojo.common.train import TrainConfig

    EvaluationTargetLike = (
        Endpoint | CustomDeployment | FlashEndpointTarget | CustomTarget | RecipeTarget
    )

T = TypeVar("T")

Scorer = Literal["exact_match", "contains", "boxed_math"]
MODAL_EVALS_WORKSPACE = "modal-labs"

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


@dataclass(frozen=True)
class EvaluationRelease:
    """A published Modal Evals release.

    Attributes:
        release_id: Release ID (``er-...``).
        evaluation_key: Evaluation the release belongs to.
        task_count: Tasks the release published.
        environment: Modal environment it was published to, or the default one.
        server: Modal Evals API URL, or the default one.
    """

    release_id: str
    evaluation_key: str
    task_count: int
    environment: str | None = None
    server: str | None = None

    def run(self, target: EvaluationTargetLike, **kwargs: Any) -> EvaluationJob:
        """Run this release against ``target``; takes the keyword arguments of
        :func:`run_evaluation`."""
        kwargs.setdefault("environment", self.environment)
        kwargs.setdefault("server", self.server)
        return run_evaluation(self.release_id, target, **kwargs)


def publish_dataset_evaluation(
    dataset: DatasetConfig,
    key: str | None = None,
    *,
    scorer: Scorer = "exact_match",
    display_name: str | None = None,
    description: str | None = None,
    key_column: str | None = None,
    limit: int | None = None,
    max_concurrency: int = 16,
    environment: str | None = None,
    server: str | None = None,
) -> EvaluationRelease:
    """Publish a dataset's ``eval_rows()`` as a private Modal Evals evaluation.

    Each row becomes a task holding its ``input_key()`` column (a prompt or a
    list of chat messages) and ``label_key()`` column. A trial sends the
    messages to the target as one chat completion and scores the reply:
    ``exact_match`` and ``contains`` compare text, ``boxed_math`` compares the
    last ``\\boxed{}`` or number. A new evaluation is private to ``environment``;
    publishing again under the same key adds a release.

    Args:
        dataset: Dataset whose ``eval_rows()`` become tasks.
        key: Evaluation key, lowercase words joined by hyphens; defaults to
            ``dojo-<hash>`` of the tasks and scorer, so the same rows reuse
            one evaluation.
        scorer: How a reply is scored against the row's label.
        display_name: Name shown in the Modal dashboard; defaults to ``key``.
        description: Description shown in the Modal dashboard; defaults to
            one naming the dataset.
        key_column: Column holding each task's key; defaults to ``row-<index>``.
        limit: Most rows to publish.
        max_concurrency: Trials at once by default.
        environment: Modal environment to publish to.
        server: Modal Evals API URL.

    Raises:
        ValueError: The dataset has no label column or no rows.
    """
    _require_modal_evals()
    from modal_dojo.evaluation_runner.dataset import build_evaluation, task_from_row

    label_key = dataset.label_key()
    if label_key is None:
        raise ValueError(
            f"{type(dataset).__name__} has no label_key() to score against."
        )
    input_key = dataset.input_key()
    tasks = []
    for index, row in enumerate(dataset.eval_rows() or ()):
        if limit is not None and index >= limit:
            break
        tasks.append(
            task_from_row(
                index,
                row,
                input_key=input_key,
                label_key=label_key,
                scorer=scorer,
                key_column=key_column,
            )
        )
    if not tasks:
        raise ValueError(f"{type(dataset).__name__} has no eval rows to publish.")
    key = key or _content_key(tasks, scorer)
    evaluation = build_evaluation(
        key,
        tasks,
        display_name=display_name or key,
        description=description
        or f"{len(tasks)} rows of {type(dataset).__name__}, scored by {scorer}.",
        max_concurrency=max_concurrency,
    )
    return _LOOP.run(_publish(evaluation, environment, server))


async def _publish(
    evaluation: Any, environment: str | None, server: str | None
) -> EvaluationRelease:
    from modal_evals.images import build_image, runner_image
    from modal_evals.params_contract import validate_params_contract
    from modal_evals.proto import flash_eval_api_pb2
    from modal_evals.publish import evaluation_definition_to_proto
    from modal_evals.task_snapshots import build_task_snapshot

    definition = evaluation.definition()
    validate_params_contract(definition.params_schema, definition.params_defaults)
    built = await build_task_snapshot(await definition.materialize_tasks())
    client = await _client(server)
    try:
        environment_id = await client.environment_id(environment)
        build = await client.begin_release(environment_id, environment=environment)
        image_id = await build_image(
            runner_image(
                definition.runner.entrypoint,
                package=definition.runner.package,
                files=definition.runner.files,
                base_image=definition.runner.base_image,
                global_namespace=build.global_image,
            ),
            app=build.app,
        )
        request = flash_eval_api_pb2.PublishEvaluationReleaseRequest(
            environment_id=environment_id,
            definition=evaluation_definition_to_proto(
                definition.published_definition(image_id)
            ),
            task_snapshot=flash_eval_api_pb2.TaskSnapshot(
                digest=built.snapshot.digest, task_count=built.snapshot.task_count
            ),
        )
        await client.upload_task_snapshot(built, environment=environment)
        response = await client.publish_release(request)
    finally:
        await client.close()
    return EvaluationRelease(
        release_id=response.release_id,
        evaluation_key=response.evaluation_key,
        task_count=built.snapshot.task_count,
        environment=environment,
        server=server,
    )


def _content_key(tasks: list[Any], scorer: str) -> str:
    payload = json.dumps(
        [scorer, [(task.key, task.data.model_dump(mode="json")) for task in tasks]],
        sort_keys=True,
    )
    return f"dojo-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


@exclude_from_api_reference
@pydantic_dataclass(config=ConfigDict(extra="forbid"))
class ModalEvalConfig:
    """Score every Nth checkpoint of a training run with Modal Evals.

    The run's ``eval_dataset`` (or ``dataset``) ``eval_rows()`` are published
    as a private evaluation at launch. Internal-only: launching needs the
    modal-labs workspace and the ``modal-evals`` SDK.

    Args:
        every_n_checkpoints:
            Evaluate each committed checkpoint whose 1-based position is a
            multiple of this.
        scorer:
            ``exact_match`` or ``contains`` compare the reply with the row's
            label; ``boxed_math`` compares the last ``\\boxed{}`` or number.
        max_rows:
            Most eval rows to publish.
        temperature:
            Sampling temperature of each chat completion.
        max_tokens:
            Most tokens in each reply.
        evaluate_final:
            Also evaluate the last checkpoint once training ends.
    """

    every_n_checkpoints: int = Field(ge=1)
    scorer: Scorer = "exact_match"
    max_rows: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)
    evaluate_final: bool = True


def _workspace_name() -> str:
    async def lookup() -> str:
        from google.protobuf.empty_pb2 import Empty
        from modal.client import _Client

        client = await _Client.from_env()
        return (await client.stub.WorkspaceNameLookup(Empty())).workspace_name

    return _LOOP.run(lookup())


def prepare_modal_eval(train_config: TrainConfig) -> EvaluationRelease:
    """Check that the run may use Modal Evals and publish its evaluation."""
    config = train_config.modal_eval
    assert config is not None
    _require_modal_evals()
    workspace = _workspace_name()
    if workspace != MODAL_EVALS_WORKSPACE:
        raise DojoConfigError(
            f"TrainConfig.modal_eval needs the {MODAL_EVALS_WORKSPACE} workspace, "
            f"not {workspace!r}: Modal Evals is internal-only for now."
        )
    dataset = train_config.eval_dataset or train_config.dataset
    return publish_dataset_evaluation(
        dataset, scorer=config.scorer, limit=config.max_rows
    )


def _watcher_image() -> Any:
    import modal

    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "modal==1.6.1",
            "click>=8.2,<9",
            "cloudpickle",
            "datasets",
            "httpx",
            "packaging",
            "pydantic>=2,<3",
            "fastapi",
            "randomname",
            "rich",
            "requests",
            "grpcio==1.84.0",
            "jsonschema~=4.24",
            "protobuf~=5.29",
            "referencing~=0.37",
            "typing-extensions>=4.4",
        )
        .add_local_python_source("modal_dojo", "modal_evals", copy=True)
    )


def start_modal_eval(
    train_config: TrainConfig, training_run_id: str, release: EvaluationRelease
) -> None:
    """Spawn the detached Function that evaluates the run's checkpoints."""
    import modal

    config = train_config.modal_eval
    assert config is not None
    app = modal.App(f"modal-dojo-evals-{training_run_id}")
    watch = app.function(image=_watcher_image(), timeout=24 * 60 * 60, serialized=True)(
        watch_checkpoints
    )
    with app.run(detach=True):
        watch.spawn(
            training_run_id=training_run_id,
            release_id=release.release_id,
            config=config,
            model=train_config.model,
            environment=release.environment,
        )


def due_checkpoints(
    names: list[str],
    evaluated: set[str],
    config: ModalEvalConfig,
    *,
    finished: bool,
) -> list[str]:
    """The checkpoints, by name, to evaluate now, oldest first."""
    due = [
        name
        for position, name in enumerate(names, 1)
        if position % config.every_n_checkpoints == 0
    ]
    if finished and config.evaluate_final and names and names[-1] not in due:
        due.append(names[-1])
    return [name for name in due if name not in evaluated]


def watch_checkpoints(
    *,
    training_run_id: str,
    release_id: str,
    config: ModalEvalConfig,
    model: ModelConfig,
    environment: str | None,
    poll_seconds: float = 60.0,
) -> None:
    """Evaluate a run's due checkpoints until it ends."""
    from modal_dojo.common.run import TrainingRun

    while True:
        run = TrainingRun.from_id(training_run_id)
        finished = run.done()
        checkpoints = {checkpoint.name: checkpoint for checkpoint in run.checkpoints()}
        evaluated = {str(record.get("iteration")) for record in run.evaluations()}
        for name in due_checkpoints(
            list(checkpoints), evaluated, config, finished=finished
        ):
            evaluate_checkpoint(
                training_run_id=training_run_id,
                checkpoint=checkpoints[name],
                release_id=release_id,
                config=config,
                model=model,
                environment=environment,
            )
        if finished:
            return
        time.sleep(poll_seconds)


def evaluate_checkpoint(
    *,
    training_run_id: str,
    checkpoint: Any,
    release_id: str,
    config: ModalEvalConfig,
    model: ModelConfig,
    environment: str | None,
) -> dict[str, Any]:
    """Serve ``checkpoint``, run the evaluation against it, record the outcome
    in the run's ``metadata["modal_evals"]``, and stop the Endpoint."""
    from modal_dojo.common.endpoint import Endpoint

    record: dict[str, Any] = {
        "iteration": checkpoint.name,
        "job_id": "",
        "status": "failed",
        "score": None,
        "url": None,
    }
    endpoint = None
    try:
        endpoint = Endpoint.launch(model, checkpoint, environment=environment)
        endpoint.wait_until_ready()
        job = run_evaluation(
            release_id,
            endpoint,
            params={"temperature": config.temperature, "max_tokens": config.max_tokens},
            name=f"{training_run_id} {checkpoint.name}",
            tags={"training_run_id": training_run_id, "checkpoint": checkpoint.name},
            environment=environment,
        )
        record["job_id"] = job.job_id
        results = job.wait()
        record["status"] = results.job.status
        record["url"] = results.output_volume.url
        try:
            record["score"] = job.mean("score", results)
        except ValueError:
            pass
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        if endpoint is not None:
            endpoint.stop()
    _record_evaluation(training_run_id, record)
    return record


def _record_evaluation(training_run_id: str, record: dict[str, Any]) -> None:
    from modal_dojo.common.run import TrainingRun

    run = TrainingRun.from_id(training_run_id)
    metadata = dict(run.metadata or {})
    metadata["modal_evals"] = [*run.evaluations(), record]
    run.metadata = metadata
    run.save()
