"""Framework-agnostic launcher helpers shared by the slime and miles apps.

The two launchers are structurally identical: they ship user callables into the
image, resolve checkpoint volumes, tag the Modal app, run the download/prepare
phases, initialize/finalize the ``TrainingRun`` record, and persist
completed-run artifacts. That shared machinery lives here; each framework passes its own
status enum / recipe hooks so the behavior stays framework-specific where it
must.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from contextlib import asynccontextmanager, contextmanager
import inspect
import os
import secrets as _secrets
import tempfile
import textwrap
import time
from typing import Any, Callable

import cloudpickle
from modal import Image, Retries, Volume

from modal_dojo.common import COMMON_DOJO_TAGS, modal_tag_value
from modal_dojo.common.framework import (
    Framework,
    resolve_caller_module,
)
from modal_dojo.common.modal_refs import register_modal_cloudpickle_reducers
from modal_dojo.common.modal_urls import modal_app_dashboard_url
from modal_dojo.common.run import (
    CHECKPOINT_LOCATION_METADATA_KEY,
    TrainingRun,
    TrainingRunStatus,
    mark_training_attempt_finished,
    mark_training_attempt_started,
    metric_run_id_for_attempt,
    record_metric_attempt,
    run_scoped_save_root,
    set_checkpoint_location,
)
from modal_dojo.common.checkpoint import require_within_volume_mount
from modal_dojo.common.launcher_utils import (
    drop_materialized_config_key,
    serialize_recipe_params,
)
from modal_dojo.common.metrics import MetricConfig, metric_metadata
from modal_dojo.utils.metadata import MetadataStore, vol_put
from modal_dojo.common.train_result import (
    save_train_result_blob,
    train_result_payload,
)
from modal_dojo.train_recipes.base import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
)


def resolve_caller_context() -> tuple[Any, str | None]:
    """Register the caller module for cloudpickle-by-value and return
    ``(caller_module, caller_script)``. The script path lets ``ship_callable``
    tell a user's inline callable apart from one imported from a shipped file."""
    caller_module = resolve_caller_module()
    if caller_module is not None and caller_module.__name__ != "__main__":
        cloudpickle.register_pickle_by_value(caller_module)
    register_modal_cloudpickle_reducers()

    caller_script = None
    if caller_module is not None:
        mod_file = getattr(caller_module, "__file__", None)
        if mod_file and os.path.isfile(mod_file):
            caller_script = os.path.abspath(mod_file)
    return caller_module, caller_script


def mount_caller_source(image: "Image", caller_script: str | None) -> "Image":
    """Copy the caller script onto the image at ``/root/<name>.py``."""
    if caller_script is None:
        return image
    name = os.path.splitext(os.path.basename(caller_script))[0]
    return image.add_local_file(
        caller_script,
        remote_path=f"/root/{name}.py",
        copy=True,
    )


def ship_callable(
    image: "Image",
    fn: Any,
    *,
    caller_script: str | None,
    fallback_name: str,
) -> tuple[Image, str]:
    fn_mod = getattr(fn, "__module__", None) or ""
    if fn_mod.startswith("modal_dojo"):
        return image, f"{fn_mod}.{getattr(fn, '__name__', fallback_name)}"
    try:
        fn_file = os.path.abspath(inspect.getfile(fn))
    except (TypeError, OSError):
        fn_file = None
    if fn_file and os.path.isfile(fn_file) and fn_file != caller_script:
        fn_module_name = os.path.splitext(os.path.basename(fn_file))[0]
        image = image.add_local_file(
            fn_file,
            remote_path=f"/root/{fn_module_name}.py",
            copy=True,
        )
        return image, f"{fn_module_name}.{getattr(fn, '__name__', fallback_name)}"
    fn_name = getattr(fn, "__name__", fallback_name)
    try:
        payload = base64.b64encode(cloudpickle.dumps(fn)).decode("ascii")
    except Exception:
        src = textwrap.dedent(inspect.getsource(fn))
        module_src = src
    else:
        module_src = textwrap.dedent(
            f"""
            import base64
            import cloudpickle

            {fn_name} = cloudpickle.loads(base64.b64decode({payload!r}))
            """
        ).lstrip()
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix=f"notebook_{fallback_name}_",
        delete=False,
    ) as tmp:
        tmp.write(module_src)
        tmp_path = tmp.name
    mod_name = os.path.splitext(os.path.basename(tmp_path))[0]
    image = image.add_local_file(
        tmp_path,
        remote_path=f"/root/{mod_name}.py",
        copy=True,
    )
    return image, f"{mod_name}.{fn_name}"


def create_training_volumes(
    checkpoint: Any,
    *,
    volume_prefix: str,
    data_volume_name: str | None = None,
    mount_metadata: bool = False,
) -> tuple[str, str, dict[str, Volume]]:
    def volume(volume_name: str) -> Volume:
        return Volume.from_name(volume_name, create_if_missing=True)

    name = (
        getattr(checkpoint, "checkpoints_volume_name", None)
        or f"{volume_prefix}-checkpoints"
    )
    mount = (
        getattr(checkpoint, "checkpoints_mount_path", None) or str(CHECKPOINTS_PATH)
    ).rstrip("/") or "/"
    volumes = {
        str(HF_CACHE_PATH): volume("huggingface-cache"),
        str(DATA_PATH): volume(data_volume_name or f"{volume_prefix}-data"),
        mount: volume(name),
    }
    if mount_metadata:
        volumes["/metadata"] = volume("training-gym-metadata")
    return name, mount, volumes


def experimental_options(recipe: Any) -> dict[str, Any]:
    return dict((recipe.train_function_kwargs or {}).get("experimental_options") or {})


def _user_secrets(recipe: Any) -> list[Any]:
    secrets = (recipe.train_function_kwargs or {}).get("secrets") or []
    return list(secrets) if isinstance(secrets, (list, tuple)) else [secrets]


def training_function_options(
    recipe: Any,
    *,
    framework: str,
    secrets: list[Any],
    experimental_options: dict[str, Any],
) -> dict[str, Any]:
    overrides = dict(recipe.train_function_kwargs or {})
    overrides.pop("secrets", None)
    if extra := sorted(set(overrides) - {"experimental_options", "ephemeral_disk"}):
        raise TypeError(
            f"Unsupported {framework}.train_function_kwargs keys: {', '.join(extra)}"
        )
    return {
        "gpu": f"{recipe.gpu_type}:{recipe.gpu_allocation.gpus_per_node}",
        "memory": recipe.memory,
        "cpu": recipe.cpu,
        "cloud": recipe.cloud,
        "region": recipe.region,
        "secrets": [*secrets, *_user_secrets(recipe)],
        "ephemeral_disk": overrides.get("ephemeral_disk"),
        "timeout": 24 * 60 * 60,
        "retries": Retries(max_retries=recipe.max_retries, initial_delay=0.0),
        "single_use_containers": True,
        "experimental_options": {
            **experimental_options,
            **(overrides.get("experimental_options") or {}),
        },
        "serialized": True,
        "name": "train",
    }


async def start_training_cluster(
    recipe: Any,
    volumes: "tuple[Volume, ...]",
    *host_ip_vars: str,
    modal_app_id: str,
    modal_app_url: str,
    framework_status_url: str,
    framework_status_token: str,
) -> "tuple[Any, str, str]":
    from modal_dojo.common.ray_cluster import ModalRayCluster

    modal_app_id = modal_app_id or os.environ.get("MODAL_APP_ID", "")
    if framework_status_url:
        os.environ["MODAL_DOJO_FRAMEWORK_STATUS_URL"] = framework_status_url
    if framework_status_token:
        os.environ["MODAL_DOJO_FRAMEWORK_STATUS_TOKEN"] = framework_status_token
    await asyncio.gather(*(volume.reload.aio() for volume in volumes))

    cluster = ModalRayCluster()
    cluster.discover_cluster(recipe.total_nodes)
    os.environ.update(dict.fromkeys([*host_ip_vars, "HOST_IP"], cluster.node_ip))
    return cluster, modal_app_id, modal_app_url or modal_app_dashboard_url(modal_app_id)


def build_app_tags(
    *,
    framework: str,
    model: Any,
    recipe_app_tags: dict[str, str],
    metrics: "MetricConfig | None",
) -> dict[str, str]:
    """Build the Modal app tag dict for dashboard auto-discovery."""
    tags = {
        **COMMON_DOJO_TAGS,
        "_modal_framework": framework,
        "_modal_model_name": modal_tag_value(model.model_name),
        **recipe_app_tags,
    }
    if metrics is not None:
        tags["_modal_metric_provider"] = metrics.provider
        tags["_modal_metric_project"] = modal_tag_value(metrics.project)
        if metrics.group:
            tags["_modal_metric_group"] = modal_tag_value(metrics.group)
        # Provider-neutral tags already identify W&B. Duplicating the project
        # and group as legacy W&B tags exceeds Modal's eight-tag limit.
    return tags


def report_phase(
    training_run_id: str, phase: str, url: str = "", token: str = ""
) -> None:
    from modal_dojo.common.status_reporter import enqueue_framework_status

    if training_run_id:
        enqueue_framework_status(
            training_run_id, phase, url=url or None, token=token or None, is_active=True
        )


def register_recipe_functions(
    app: Any,
    image: Image,
    recipe: Any,
    *,
    hf_cache_volume: Volume,
    data_volume: Volume,
    checkpoints_volume: Volume,
    checkpoints_mount_path: str,
    download_phase: str,
    download: Callable[[], None],
    download_timeout: int,
    prepare_dataset: Callable[[], None],
) -> None:
    from modal_dojo.common import hf_secrets, proxy_auth_secrets
    from modal_dojo.common.status_reporter import flush as flush_status

    volumes = (hf_cache_volume, checkpoints_volume)

    @app.function(
        image=image,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            checkpoints_mount_path: checkpoints_volume,
        },
        timeout=download_timeout,
        secrets=[*hf_secrets(), *proxy_auth_secrets()],
        serialized=True,
        name="download",
    )
    def _download(
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ) -> None:
        report_phase(
            training_run_id,
            download_phase,
            framework_status_url,
            framework_status_token,
        )
        for volume in volumes:
            volume.reload()
        download()
        for volume in volumes:
            volume.commit()
        flush_status(timeout_seconds=2.0)

    @app.function(
        image=image,
        volumes={str(DATA_PATH): data_volume},
        timeout=24 * 60 * 60,
        secrets=[*hf_secrets(), *_user_secrets(recipe)],
        serialized=True,
        name="prepare_dataset",
    )
    def _prepare_dataset() -> None:
        data_volume.reload()
        prepare_dataset()
        data_volume.commit()

    app.download = _download
    app.prepare_dataset = _prepare_dataset


def ship_recipe_callables(
    image: Image,
    recipe: Any,
    *,
    caller_script: str | None,
    reward_post_process_in_config: bool,
) -> Image:
    hooks = {
        "custom_rm_function": "custom_rm_path",
        "custom_generate_function": "custom_generate_function_path",
        "custom_reward_post_process_function": (
            "custom_reward_post_process_path" if reward_post_process_in_config else None
        ),
        "rollout_function": None,
        **{
            attr: f"modal_dojo_{attr}_path"
            for attr in (
                "custom_rollout_log_function",
                "custom_eval_rollout_log_function",
                "custom_megatron_before_log_prob_hook",
                "custom_megatron_before_train_step_hook",
            )
        },
    }

    for attr, key in hooks.items():
        value = getattr(recipe, attr)
        if not callable(value):  # A str is an import path the user vouches for.
            continue
        image, path = ship_callable(
            image, value, caller_script=caller_script, fallback_name=attr
        )
        if key is None:
            setattr(recipe, attr, path)
        else:
            recipe.extra_config = {**(recipe.extra_config or {}), key: path}
            setattr(recipe, attr, None)
    return image


def write_dataset_if_needed(dataset: Any, path: str) -> bool:
    """Write and validate a dataset unless its cached materialization exists."""
    if os.path.exists(path):
        dataset.validate_written(path)
        return False
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    print(f"Writing dataset ({path})...")
    stem, suffix = os.path.splitext(os.path.basename(path))
    tmp = os.path.join(parent, f"{stem}.{_secrets.token_hex(8)}.tmp{suffix}")
    try:
        dataset.write(tmp)
        dataset.validate_written(tmp)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    os.replace(tmp, path)
    return True


def download_model_if_needed(model: Any, *, always: bool = False) -> None:
    """Download unless cached; ``always`` repatches an already-cached snapshot."""

    def has_files(path: Path) -> bool:
        return path.exists() and (not path.is_dir() or any(path.iterdir()))

    hub = HF_CACHE_PATH / "hub" / f"models--{model.model_name.replace('/', '--')}"
    local = getattr(model, "model_path", None)
    cached = has_files(hub / "snapshots") and (not local or has_files(Path(local)))
    if not cached:
        print(f"Downloading model {model.model_name}...")
    if always or not cached:
        model.download()


def write_datasets(
    dataset: Any,
    eval_dataset: Any,
    dataset_path: str,
    eval_dataset_path: str | None,
) -> bool:
    """Materialize the training and optional framework-evaluation datasets."""
    wrote = write_dataset_if_needed(dataset, dataset_path)
    if eval_dataset is not None and eval_dataset_path is not None:
        wrote = write_dataset_if_needed(eval_dataset, eval_dataset_path) or wrote
    return wrote


async def init_training_run_record(
    *,
    training_run_id: str,
    modal_app_id: str,
    modal_app_url: str,
    framework: "Framework",
    initializing_status: Any,
    recipe: Any,
    model: Any,
    dataset: Any,
    eval_dataset: Any,
    dataset_path: str,
    eval_dataset_path: str | None,
    recipe_metadata: tuple[str, ...] = (),
    metric_entity: str,
    framework_status_token: str,
    checkpoint_dir: str,
    checkpoints_volume_name: str,
    checkpoints_mount_path: str,
) -> tuple[Any, str, str]:
    """Create or resume the ``TrainingRun`` record for this attempt and persist
    the framework-status token. Returns
    ``(run_record, metric_run_id, framework_status_token)``.

    Reuses the record the local ``TrainConfig.train()`` driver creates before
    invoking download/convert (so those phases are visible in the dashboard);
    falls back to a fresh record when ``train()`` is invoked directly.
    """

    def dataset_summary(value: Any) -> dict[str, str]:
        return {"hf_repo": getattr(value, "hf_repo", ""), "name": type(value).__name__}

    config_summary = {
        "model": {"model_name": model.model_name} if model else {},
        "recipe": serialize_recipe_params(
            recipe,
            dataset=dataset,
            eval_dataset=eval_dataset,
            dataset_path=dataset_path,
            eval_dataset_path=eval_dataset_path,
            model=model,
        )
        | {
            key: getattr(recipe, key) for key in recipe_metadata if getattr(recipe, key)
        },
        "metrics": metric_metadata(recipe.metrics, entity=metric_entity, run_id=""),
        "dataset": dataset_summary(dataset),
        "eval_dataset": dataset_summary(eval_dataset)
        if eval_dataset is not None
        else None,
        "lr": recipe.lr,
        "global_batch_size": recipe.global_batch_size,
    }
    metric_cfg = recipe.metrics
    try:
        run_record = await TrainingRun.from_id(training_run_id, is_async=True)
        run_record.modal_app_id = modal_app_id
        run_record.modal_app_url = modal_app_url
        run_record.config = config_summary
        run_record.framework_status = initializing_status
    except KeyError:
        created_at = int(time.time())
        run_record = TrainingRun(
            training_run_id=training_run_id,
            modal_app_id=modal_app_id,
            modal_app_url=modal_app_url,
            framework=framework,
            config=config_summary,
            framework_status=initializing_status,
            created_at=created_at,
            started_at=created_at,
        )
    set_checkpoint_location(
        run_record,
        checkpoint_dir=checkpoint_dir,
        checkpoints_volume_name=checkpoints_volume_name,
        checkpoints_mount_path=checkpoints_mount_path,
    )
    attempt_count = mark_training_attempt_started(
        run_record, started_at=int(time.time())
    )
    metric_run_id = ""
    if metric_cfg is not None:
        metric_run_id = metric_run_id_for_attempt(training_run_id, attempt_count)
        metric_data = metric_metadata(
            metric_cfg, entity=metric_entity, run_id=metric_run_id
        )
        run_record.config["metrics"] = metric_data
        record_metric_attempt(
            run_record,
            **metric_data,
            attempt_count=attempt_count,
        )
    if attempt_count > 1:
        print(
            f"WARNING: training run {training_run_id} is retrying after preemption "
            f"or interruption (attempt {attempt_count})."
        )
    if not framework_status_token:
        framework_status_token = _secrets.token_urlsafe(32)
    await run_record.save(is_async=True)
    await vol_put(
        MetadataStore.FRAMEWORK_STATUS_TOKENS,
        training_run_id,
        {"token": framework_status_token},
        is_async=True,
    )
    print(f"TrainingRun recorded: {training_run_id}")
    return run_record, metric_run_id, framework_status_token


def compute_save_root(
    save: str | None,
    *,
    recipe_default_save_root: str,
    mounted_save_root: str,
    training_run_id: str,
) -> str:
    """Resolve the run-scoped checkpoint save root. A configured ``save`` equal
    to the recipe default is redirected to the mounted volume path so checkpoints
    land on the checkpoints Volume."""
    configured_save_root = str(save).rstrip("/") if save else mounted_save_root
    save_root = (
        mounted_save_root
        if configured_save_root == recipe_default_save_root
        else configured_save_root
    )
    return require_within_volume_mount(
        run_scoped_save_root(save_root, training_run_id),
        mounted_save_root,
    )[0]


def configured_recipe_save(recipe: Any) -> str | None:
    extra = recipe.extra_config
    save = extra.get("save") if isinstance(extra, dict) else None
    if not save:
        save = recipe.save
    return str(save) if save else None


def compute_recipe_save_root(
    recipe: Any,
    *,
    recipe_default_save_root: str,
    mounted_save_root: str,
    training_run_id: str,
) -> str:
    """Resolve the run-scoped save root after ``extra_config`` save precedence.

    Keys in ``extra_config`` drop the matching CLI flag, so a ``save`` override
    is the path training writes. This function does not mutate ``recipe``.
    """
    return compute_save_root(
        configured_recipe_save(recipe),
        recipe_default_save_root=recipe_default_save_root,
        mounted_save_root=mounted_save_root,
        training_run_id=training_run_id,
    )


def apply_scoped_save(recipe: Any, save_root: str) -> None:
    extra = recipe.extra_config
    if isinstance(extra, dict) and extra.get("save"):
        recipe.extra_config = {**extra, "save": save_root}
    if recipe.save:
        recipe.save = save_root


async def complete_training_run(
    run_record: TrainingRun,
    *,
    checkpoints_volume: Any,
    app_name: str,
    model: Any,
    group_id: str | None,
) -> dict[str, Any]:
    assert run_record.metadata is not None
    payload = train_result_payload(
        app_name=app_name,
        framework=run_record.framework,
        training_run_id=run_record.training_run_id,
        model_config=model,
        metrics=run_record.config["metrics"],
        group_id=group_id or "",
        **run_record.metadata[CHECKPOINT_LOCATION_METADATA_KEY],
    )
    run_record.app_name = app_name
    run_record.source_model = payload["model_config"]
    run_record.metrics = payload["metrics"]
    await save_train_result_blob(payload, is_async=True)
    run_record.status = TrainingRunStatus.COMPLETED
    mark_training_attempt_finished(
        run_record, status="completed", ended_at=int(time.time())
    )
    await checkpoints_volume.commit.aio()
    print(f"TrainingRun saved: {run_record.training_run_id}")
    return payload


def mark_run_stopped(run_record: Any) -> None:
    """Mark the run STOPPED (e.g. KeyboardInterrupt)."""
    run_record.status = TrainingRunStatus.STOPPED
    mark_training_attempt_finished(
        run_record, status="stopped", ended_at=int(time.time())
    )


def mark_run_failed(run_record: Any, exc: BaseException) -> None:
    """Mark the run FAILED, preserving any more-specific error already set."""
    run_record.status = TrainingRunStatus.FAILED
    terminal_error = f"{type(exc).__name__}: {exc}"
    # Prefer a more specific message already set (e.g. the raw Ray driver
    # message from the is_success check) over the generic wrapper.
    run_record.error_message = run_record.error_message or terminal_error
    mark_training_attempt_finished(
        run_record, status="failed", ended_at=int(time.time())
    )


async def build_terminal_run_record(run_record: Any, training_run_id: str) -> Any:
    """Re-fetch the latest record from the volume and stamp terminal fields
    (completion fields, status, timestamps, duration, error) onto it so the
    caller can persist a consistent terminal record."""
    finished_at = int(time.time())
    try:
        latest_run_record = await TrainingRun.from_id(training_run_id, is_async=True)
    except Exception:
        latest_run_record = run_record

    latest_run_record.status = run_record.status
    latest_run_record.app_name = run_record.app_name
    latest_run_record.source_model = run_record.source_model
    latest_run_record.metrics = run_record.metrics
    latest_run_record.ended_at = finished_at
    # Propagate the terminal error onto the re-fetched record so the save
    # below persists it (the fresh fetch wouldn't carry it).
    if run_record.error_message:
        latest_run_record.error_message = run_record.error_message
    source_metadata = run_record.metadata or {}
    latest_metadata = dict(latest_run_record.metadata or {})
    for key in ("last_attempt_status", "last_attempt_ended_at", "terminal_reason"):
        if key in source_metadata:
            latest_metadata[key] = source_metadata[key]
    if latest_metadata:
        latest_run_record.metadata = latest_metadata
    if latest_run_record.completed_at is None:
        latest_run_record.completed_at = finished_at
    latest_run_record.duration_seconds = max(
        0, finished_at - latest_run_record.started_at
    )
    return latest_run_record


@contextmanager
def resumed_recipe(recipe: Any, save_root: str, checkpoint: dict[str, Any] | None):
    original = {
        field: getattr(recipe, field)
        for field in ("save", "load", "start_rollout_id", "ref_load", "no_load_optim")
    }
    try:
        if checkpoint is not None:
            iteration = checkpoint.get("resume_from_iteration")
            num_rollout = recipe.num_rollout
            if iteration is not None and recipe.effective_num_epoch() is None:
                if iteration + 1 > num_rollout:
                    raise RuntimeError(
                        f"Resume would start at rollout {iteration + 1}, "
                        f"but num_rollout={num_rollout}; nothing would run."
                    )
                if iteration + 1 == num_rollout:
                    print(
                        "WARNING: Resume checkpoint is already at the final configured "
                        "rollout; the retry will exit without running another rollout.",
                        flush=True,
                    )
            print(
                f"WARNING: detected existing checkpoint in "
                f"{checkpoint['resume_checkpoint_path']}; "
                "resuming training from last saved iteration."
            )
            recipe.load = save_root
            recipe.start_rollout_id = None
            drop_materialized_config_key(recipe, "start_rollout_id")
            if recipe.no_save_optim and not recipe.no_load_optim:
                print(
                    "WARNING: no_save_optim=True — enabling no_load_optim for resume."
                )
            recipe.no_load_optim = recipe.no_save_optim
        yield
    finally:
        for field, value in original.items():
            setattr(recipe, field, value)


@asynccontextmanager
async def training_run_lifecycle(run_record: TrainingRun, status_token: str = ""):
    from modal_dojo.common.status_reporter import enqueue_framework_status

    async def set_status(status: Any) -> None:
        run_record.framework_status = status
        enqueue_framework_status(
            run_record.training_run_id, status.value, token=status_token
        )

    try:
        yield set_status
    except KeyboardInterrupt:
        mark_run_stopped(run_record)
        raise
    except BaseException as exc:
        mark_run_failed(run_record, exc)
        raise
    finally:
        if run_record.status is not TrainingRunStatus.RUNNING:
            latest = await build_terminal_run_record(
                run_record, run_record.training_run_id
            )
            try:
                await latest.save(is_async=True)
            except Exception as exc:
                print(f"Failed to save run record: {exc}")


def check_training_result(result: Any, run_record: TrainingRun) -> None:
    if not result.is_success:
        training_run_id = run_record.training_run_id
        message = result.message or f"Ray job finished with status: {result.status}"
        error = RuntimeError(f"{message} (training_run_id={training_run_id})")
        error.training_run_id = training_run_id  # pyright: ignore[reportAttributeAccessIssue]
        run_record.error_message = str(error)
        raise error
    print(f"Ray job completed: {result.status}")


def training_reporting_env(
    recipe: Any, model: Any, app_name: str, framework_status_url: str
) -> dict[str, str]:
    status_url = os.environ.get("MODAL_DOJO_FRAMEWORK_STATUS_URL") or (
        framework_status_url or ""
    )
    if not status_url:
        print(
            "WARNING: no dashboard URL passed to train() and no "
            "MODAL_DOJO_FRAMEWORK_STATUS_URL set inside the "
            "container. Phase reporting is disabled for this run."
        )
    parser = getattr(model, "response_parser", None) if model is not None else None
    module = getattr(parser, "__module__", "")
    name = getattr(parser, "__qualname__", "") or getattr(parser, "__name__", "")
    return {
        "MODAL_DOJO_APP_NAME": app_name,
        "MODAL_DOJO_TOTAL_STEPS": str(recipe.num_rollout),
        "MODAL_DOJO_LOSS_TYPE": recipe.loss_type,
        "MODAL_DOJO_RESPONSE_PARSER_PATH": f"{module}.{name}"
        if module and name
        else "",
        "MODAL_DOJO_CAPTURE_TRACE": "1" if recipe.capture_trace else "",
        "MODAL_DOJO_TRACE_SAMPLE_LIMIT": str(recipe.trace_sample_limit),
        "MODAL_DOJO_FRAMEWORK_STATUS_URL": status_url,
    }
