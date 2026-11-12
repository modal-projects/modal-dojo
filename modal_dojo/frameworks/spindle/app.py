"""Launch a Spindle driver in a detached Modal app, like Slime and Miles."""

from __future__ import annotations

import dataclasses
import secrets
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
    from modal_dojo.cli.setup import ensure_dashboard_deployed
    from modal_dojo.common.config import (
        CONFIG_PATH,
        get_framework_status_url,
        require_migrated_config,
    )
    from modal_dojo.common.ids import create_hash
    from modal_dojo.common.train import TrainStatusDisplay, launch_detached
    from modal_dojo.train_recipes.base import CHECKPOINTS_PATH

    from .launcher import check_server
    from .reporting import new_training_run

    require_migrated_config()
    program = recipe.program
    max_tokens = _max_tokens(program)
    # Fail on a bad key, model or context before creating a run or an app.
    server = check_server(
        recipe.base_url, recipe._api_key, program.model_name, max_tokens=max_tokens
    )
    training_run_id = create_hash(
        program.model_name, "", f"SpindleRecipe:{Framework.SPINDLE.value}", "", ""
    )
    ensure_dashboard_deployed()
    framework_status_url = get_framework_status_url() or ""
    if not framework_status_url:
        raise DojoConfigError(
            "Could not connect to Dojo. Run `modal-dojo setup` before launching Spindle."
        )
    log_dir = f"{CHECKPOINTS_PATH}/{training_run_id}"
    status_display = TrainStatusDisplay(
        run_id=training_run_id,
        framework=Framework.SPINDLE.value,
        model_name=program.model_name,
        dataset_name=_dataset_name(program),
        framework_status_url=framework_status_url,
        config_path=str(CONFIG_PATH),
    )
    if show_output:
        status_display.print_banner()
    run = new_training_run(program, training_run_id, log_dir)
    run.app_name = training_run_id
    run.metadata.update(server)

    def start(app, framework_status_token: str):
        return app.train.spawn(
            framework_status_url=framework_status_url,
            framework_status_token=framework_status_token,
        )

    return launch_detached(
        run,
        framework_status_token=secrets.token_urlsafe(32),
        build_app=lambda: build_spindle_app(
            recipe, training_run_id, log_dir, max_tokens=max_tokens
        ),
        start=start,
        status_display=status_display,
        show_output=show_output,
    )


def _max_tokens(program) -> int | None:
    return program.config.max_tokens


def _dataset_name(program) -> str:
    return type(program.config.dataset_builder).__name__


def build_spindle_app(
    recipe, training_run_id: str, log_dir: str, *, max_tokens: int | None
) -> modal.App:
    """A CPU-only app whose ``train`` function runs the program's driver."""
    import modal

    from modal_dojo.common import hf_secrets, proxy_auth_secrets
    from modal_dojo.common.framework import mount_tools_dir
    from modal_dojo.common.launcher_helpers import (
        build_app_tags,
        create_training_volumes,
        mount_caller_source,
        resolve_caller_context,
    )

    # Pickle the caller's reward functions and builders by value.
    _, caller_script = resolve_caller_context()
    image = (
        modal.Image.debian_slim(python_version="3.12")
        # tinker-cookbook records the code state with git.
        .apt_install("git")
        .uv_pip_install(*spindle_requirements())
        .add_local_python_source("modal_dojo", copy=True)
    )
    image = mount_caller_source(mount_tools_dir(image), caller_script)
    program, image = ship_harbor_tasks(recipe.program, image)
    _, checkpoints_path, volumes = create_training_volumes(
        None, volume_prefix=Framework.SPINDLE.value
    )
    checkpoints = volumes[checkpoints_path]
    base_url = recipe.base_url
    app = modal.App(
        training_run_id,
        tags=build_app_tags(
            framework=Framework.SPINDLE.value,
            model=program,
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
        single_use_containers=True,
        serialized=True,
        name="train",
    )
    async def train(framework_status_url: str, framework_status_token: str) -> None:
        import os

        from modal_dojo.common.run import TrainingRun

        from .launcher import run_program

        # Dojo's shared reporting reads these, as in Slime and Miles containers.
        os.environ["MODAL_DOJO_FRAMEWORK_STATUS_URL"] = framework_status_url
        os.environ["MODAL_DOJO_FRAMEWORK_STATUS_TOKEN"] = framework_status_token
        run = await TrainingRun.from_id(training_run_id, is_async=True)
        try:
            await run_program(
                program,
                run=run,
                base_url=base_url,
                log_dir=log_dir,
                max_tokens=max_tokens,
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


def ship_harbor_tasks(program, image):
    """Add the task directories a Harbor builder uses, at paths the container sees."""
    import chz
    from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborDatasetBuilder

    from .cookbook import CookbookRL

    builder = program.config.dataset_builder
    if not isinstance(builder, HarborDatasetBuilder):
        return program, image

    shipped: dict[Path, str] = {}
    tasks = []
    for task in builder.tasks:
        if task.task_dir not in shipped:
            remote = f"{HARBOR_TASKS_PATH}/{len(shipped)}/{task.task_dir.name}"
            image = image.add_local_dir(task.task_dir, remote_path=remote)
            shipped[task.task_dir] = remote
        tasks.append(dataclasses.replace(task, task_dir=Path(shipped[task.task_dir])))
    config = chz.replace(
        program.config, dataset_builder=chz.replace(builder, tasks=tasks)
    )
    return CookbookRL(config), image
