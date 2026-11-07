"""Launch the tinker-cookbook RL driver in a detached Modal app, like Slime and Miles."""

from __future__ import annotations

import copy
import dataclasses
import secrets
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.framework import Framework

if TYPE_CHECKING:
    import modal

    from modal_dojo.common.run import TrainingRun

# Harbor sandboxes build from task directories, which are shipped to here.
HARBOR_TASKS_PATH = "/root/harbor_tasks"


def launch(recipe, *, show_output: bool = True) -> TrainingRun:
    """Start the driver in a detached Modal app; return its TrainingRun handle."""
    import chz
    import modal

    from modal_dojo.cli.setup import ensure_dashboard_deployed
    from modal_dojo.common.config import (
        CONFIG_PATH,
        get_framework_status_url,
        require_migrated_config,
    )
    from modal_dojo.common.ids import create_hash
    from modal_dojo.common.modal_lifecycle import stop_app
    from modal_dojo.common.modal_urls import modal_app_dashboard_url
    from modal_dojo.common.train import _terminalize_launch, _TrainStatusDisplay
    from modal_dojo.train_recipes.base import CHECKPOINTS_PATH
    from modal_dojo.utils.metadata import MetadataStore, vol_put

    from .launcher import check_server
    from .reporting import new_training_run

    require_migrated_config()
    config = recipe.rl_config
    # Fail on a bad key, model or context before creating a run or an app.
    server = check_server(config.base_url, recipe._api_key, config)
    training_run_id = create_hash(
        config.model_name,
        config.load_checkpoint_path or "",
        f"SpindleRecipe:{Framework.SPINDLE.value}",
        "",
        "",
    )
    ensure_dashboard_deployed()
    framework_status_url = get_framework_status_url() or ""
    if not framework_status_url:
        raise DojoConfigError(
            "Could not connect to Dojo. Run `modal-dojo setup` before launching Spindle."
        )
    framework_status_token = secrets.token_urlsafe(32)
    # Retries resume from checkpoints.jsonl in this directory.
    config = chz.replace(config, log_path=f"{CHECKPOINTS_PATH}/{training_run_id}")

    status_display = _TrainStatusDisplay(
        run_id=training_run_id,
        framework=Framework.SPINDLE.value,
        model_name=config.model_name,
        dataset_name=type(config.dataset_builder).__name__,
        framework_status_url=framework_status_url,
        config_path=str(CONFIG_PATH),
    )
    if show_output:
        status_display.print_banner()
    run = new_training_run(config, training_run_id)
    run.metadata.update(server)
    run.save()
    vol_put(
        MetadataStore.FRAMEWORK_STATUS_TOKENS,
        training_run_id,
        {"token": framework_status_token},
    )
    print(f"TrainingRun recorded: {training_run_id}")

    app = None
    function_call = None
    launch_error = None
    try:
        app = build_spindle_app(recipe, config, training_run_id)
        with modal.enable_output() if show_output else nullcontext():
            with app.run(detach=True):
                try:
                    run.modal_app_id = app.app_id or ""
                    run.modal_app_url = modal_app_dashboard_url(run.modal_app_id)
                    run.app_name = training_run_id
                    if show_output:
                        status_display.set_modal_app_url(run.modal_app_url)
                    run.save()
                    function_call = app.train.spawn(
                        dashboard_url=framework_status_url.removesuffix(
                            "/api/framework-status"
                        ),
                        status_token=framework_status_token,
                    )
                except BaseException as exc:
                    launch_error = exc
        if launch_error is not None:
            raise launch_error
    except (KeyboardInterrupt, Exception) as exc:
        if function_call is None:
            if run.modal_app_id:
                stop_app(run.modal_app_id)
            _terminalize_launch(run, exc)
            raise
        launch_error = exc

    run.function_call_id = function_call.object_id
    run._function_call = function_call
    run._status_display = status_display if show_output else None
    run.save()
    if isinstance(launch_error, KeyboardInterrupt):
        print(f'Disconnected from Modal app "{app.name}". The app was left running.')
        print(run.modal_app_url)
        raise launch_error
    if launch_error is not None:
        print(
            f"WARNING: training was launched, but Modal client cleanup failed: {launch_error}"
        )
    print(
        f"Launched training {training_run_id}: "
        f"app={run.modal_app_id}, function_call={run.function_call_id}"
    )
    return run


def build_spindle_app(recipe, config, training_run_id: str) -> modal.App:
    """A CPU-only app whose ``train`` function runs tinker-cookbook's RL loop."""
    import modal

    from modal_dojo.common import hf_secrets, proxy_auth_secrets
    from modal_dojo.common.framework import mount_tools_dir
    from modal_dojo.common.launcher_helpers import (
        build_app_tags,
        create_training_volumes,
        mount_caller_source,
        resolve_caller_context,
    )

    # Pickle the caller's reward functions, evaluators and builders by value.
    _, caller_script = resolve_caller_context()
    image = (
        modal.Image.debian_slim(python_version="3.12")
        # tinker-cookbook records the code state with git.
        .apt_install("git")
        .uv_pip_install(*spindle_requirements())
        .add_local_python_source("modal_dojo", copy=True)
    )
    image = mount_caller_source(mount_tools_dir(image), caller_script)
    config, image = ship_harbor_tasks(config, image)
    _, checkpoints_path, volumes = create_training_volumes(
        None, volume_prefix=Framework.SPINDLE.value
    )
    checkpoints = volumes[checkpoints_path]
    app = modal.App(
        training_run_id,
        tags=build_app_tags(
            framework=Framework.SPINDLE.value,
            model=config,
            recipe_app_tags={},
            metrics=None,
        ),
    )

    @app.function(
        image=image,
        volumes=volumes,
        # The key stays out of the pickled function.
        secrets=[
            modal.Secret.from_dict({"TINKER_API_KEY": recipe._api_key}),
            *proxy_auth_secrets(),
            *hf_secrets(),
        ],
        cpu=recipe.cpu,
        memory=recipe.memory,
        timeout=24 * 60 * 60,
        retries=modal.Retries(max_retries=recipe.max_retries, initial_delay=0.0),
        # A preempted driver repeats every step since the last save.
        nonpreemptible=True,
        single_use_containers=True,
        serialized=True,
        name="train",
    )
    async def train(dashboard_url: str, status_token: str) -> None:
        import os

        from modal_dojo.common.run import TrainingRun

        from .launcher import train as run_training

        run = await TrainingRun.from_id(training_run_id, is_async=True)
        try:
            await run_training(
                config,
                api_key=os.environ["TINKER_API_KEY"],
                run=run,
                dashboard_url=dashboard_url,
                status_token=status_token,
            )
        finally:
            await checkpoints.commit.aio()

    app.train = train
    return app


def spindle_requirements() -> list[str]:
    """modal-dojo's own pins plus its ``spindle`` extra."""
    from importlib.metadata import requires

    from packaging.requirements import Requirement

    pins = []
    for line in requires("modal-dojo") or []:
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate(
            {"extra": "spindle"}
        ):
            requirement.marker = None
            pins.append(str(requirement))
    return pins


def ship_harbor_tasks(config, image):
    """Add the task directories a Harbor builder uses, at paths the container sees."""
    import chz

    from .dataset_builders import HarborBuilder

    builder = config.dataset_builder
    if isinstance(builder, HarborBuilder):
        dataset = copy.copy(builder.dataset)
        root = dataset._resolve_task_root()
        remote = f"{HARBOR_TASKS_PATH}/{root.name}"
        # Ship every candidate so the split is chosen the same way remotely.
        for task_dir in dataset._iter_task_dirs():
            image = image.add_local_dir(
                task_dir, remote_path=f"{remote}/{task_dir.relative_to(root)}"
            )
        dataset.path, dataset.dataset_name, dataset.task_root = remote, "", ""
        builder = chz.replace(builder, dataset=dataset)
    elif type(builder).__name__ == "HarborDatasetBuilder":
        shipped: dict[Path, str] = {}
        tasks = []
        for task in builder.tasks:
            if task.task_dir not in shipped:
                remote = f"{HARBOR_TASKS_PATH}/{len(shipped)}/{task.task_dir.name}"
                image = image.add_local_dir(task.task_dir, remote_path=remote)
                shipped[task.task_dir] = remote
            tasks.append(
                dataclasses.replace(task, task_dir=Path(shipped[task.task_dir]))
            )
        builder = chz.replace(builder, tasks=tasks)
    else:
        return config, image
    return chz.replace(config, dataset_builder=builder), image
