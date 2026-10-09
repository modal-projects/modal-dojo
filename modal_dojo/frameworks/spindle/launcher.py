"""Modal launcher for :class:`SpindleRecipe` runs.

Two images are needed because Spindle pins ``tinker<0.26`` while tinker-cookbook
needs ``tinker>=0.30``:

* ``deploy_spindle`` (Spindle image, CPU) deploys the trainer, inference and
  frontend apps for the recipe with ``spindle.deployment_cli.deploy`` and
  returns the frontend URL.
* ``train`` (tinker-cookbook image, CPU) materializes the dataset, waits for the
  deployment and runs the cookbook GRPO loop against the frontend, reporting
  phases to the dashboard like the Miles launcher does.
"""

from __future__ import annotations

import json
import os
from typing import Any

from modal import App, Image, Secret

from modal_dojo.common import hf_secrets, proxy_auth_secrets
from modal_dojo.common import launcher_helpers as shared
from modal_dojo.common.checkpoint import Checkpoint
from modal_dojo.common.dataset import DatasetConfig
from modal_dojo.common.framework import Framework
from modal_dojo.common.launcher_helpers import (
    build_app_tags,
    compute_recipe_save_root,
    configured_recipe_save,
    init_training_run_record,
    mount_caller_source,
    register_recipe_functions,
    resolve_caller_context,
    write_datasets,
)
from modal_dojo.common.metrics import preflight_metric
from modal_dojo.common.models import ModelConfig
from modal_dojo.common.run import TrainingRun
from modal_dojo.common.status import SpindleStatus
from modal_dojo.frameworks.spindle.deployment import spindle_config_values
from modal_dojo.frameworks.spindle.grpo import (
    GrpoSettings,
    lora_target_flags,
    run_grpo,
)
from modal_dojo.train_recipes.base import CHECKPOINTS_PATH, DATA_PATH, HF_CACHE_PATH
from modal_dojo.train_recipes.spindle_recipe.recipe import (
    SPINDLE_GIT_URL,
    TINKER_COOKBOOK_GIT_URL,
    SpindleRecipe,
)

PYTHON_VERSION = "3.12"
DEFAULT_REWARD_PATH = "modal_dojo.frameworks.spindle.rewards.boxed_match_reward"
DEPLOY_TIMEOUT = 2 * 60 * 60


def _git_install_commands(url: str, ref: str, checkout: str) -> list[str]:
    return [
        f"git clone --quiet {url} {checkout}",
        f"cd {checkout} && git fetch --quiet origin {ref} && git checkout --quiet FETCH_HEAD",
        f"pip install --quiet {checkout}",
    ]


DOJO_IMPORT_DEPS = (
    "click>=8.2,<9",
    "cloudpickle",
    "datasets",
    "httpx",
    "pydantic",
    "fastapi",
    "randomname",
    "rich",
    "transformers>=5.12.1",
)


def build_spindle_image(recipe: SpindleRecipe) -> Image:
    return (
        Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("git")
        .run_commands(
            *_git_install_commands(
                SPINDLE_GIT_URL, recipe.spindle_git_ref, "/opt/spindle"
            )
        )
        .uv_pip_install(*DOJO_IMPORT_DEPS)
        .add_local_python_source("modal_dojo", copy=True)
    )


def build_grpo_image(
    recipe: SpindleRecipe,
    dataset: DatasetConfig,
    eval_dataset: DatasetConfig | None,
    caller_script: str | None,
) -> Image:
    image = (
        Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("git")
        .run_commands(
            *_git_install_commands(
                TINKER_COOKBOOK_GIT_URL,
                recipe.tinker_cookbook_git_ref,
                "/opt/tinker-cookbook",
            )
        )
        .uv_pip_install("pyarrow", "randomname")
    )
    if recipe.image_run_commands:
        image = image.run_commands(*recipe.image_run_commands)
    if recipe.image_env:
        image = image.env(dict(recipe.image_env))
    if recipe.image_overlay is not None:
        image = recipe.image_overlay(image)
    image = image.add_local_python_source("modal_dojo", copy=True)
    image = mount_caller_source(image, caller_script)
    return shared.ship_recipe_callables(
        image, recipe, caller_script=caller_script, reward_post_process_in_config=False
    )


def _chat_template_kwargs(recipe: SpindleRecipe) -> dict[str, Any]:
    value = recipe.apply_chat_template_kwargs
    if not value:
        return {}
    return dict(json.loads(value) if isinstance(value, str) else value)


def _use_real_spindle_images() -> None:
    """Make Spindle select its real trainer/rollout images in this container.

    ``spindle.providers.modal.deployment_apps.image_for`` returns a placeholder
    ``debian_slim`` image whenever ``modal.is_local()`` is false, assuming the
    module is only imported remotely by its own workers. Dojo deploys Spindle
    from inside a Modal function, so without this the trainer and SGLang apps
    would be deployed with an image that lacks Spindle itself.
    """
    from spindle.providers.modal import (  # pyright: ignore[reportMissingImports]
        deployment_apps,
    )

    images = {
        "miles": deployment_apps.miles_image,
        "megatron": deployment_apps.megatron_image,
        "sglang": deployment_apps.rollout_image,
    }
    deployment_apps.image_for = lambda backend: images[backend]


def deploy_spindle_deployment(config_values: dict[str, Any]) -> str:
    """Deploy Spindle for ``config_values`` and return the frontend base URL.

    Runs inside the Spindle image.
    """
    import modal
    from spindle.configuration import BaseConfig  # pyright: ignore[reportMissingImports]
    from spindle.deployment_cli import deploy  # pyright: ignore[reportMissingImports]
    from spindle.deployments import DeploymentConfig  # pyright: ignore[reportMissingImports]

    _use_real_spindle_images()
    recipe = BaseConfig(**config_values)
    deployment = DeploymentConfig.create(recipe)
    deploy([deployment])
    environment = recipe.platform["modal"]["environment"]
    server = modal.Function.from_name(
        recipe.platform["frontend"], "server", environment_name=environment
    )
    url = server.get_web_url()
    if not url:
        raise RuntimeError(
            f"Spindle frontend {recipe.platform['frontend']!r} has no web URL"
        )
    return url


def build_spindle_app(
    *,
    training_run_id: str,
    spindle: SpindleRecipe,
    model: ModelConfig,
    dataset: DatasetConfig,
    eval_dataset: DatasetConfig | None = None,
    checkpoint: Checkpoint | None = None,
    name: str | None = None,
    group_id: str | None = None,
) -> App:
    recipe_slug = type(spindle).__name__.lstrip("_").lower()
    app_name = name or spindle.name or f"spindle-{recipe_slug}"
    volume_prefix = spindle.name or f"spindle-{recipe_slug}"
    SpindleRecipe._validate_datasets(dataset, eval_dataset, loss_type=spindle.loss_type)
    spindle.validate_model_parallelism(model)
    dataset_path = SpindleRecipe._resolve_data_paths(dataset)
    eval_dataset_path = (
        SpindleRecipe._resolve_data_paths(eval_dataset)
        if eval_dataset is not None
        else None
    )
    config_values = spindle_config_values(
        spindle,
        model_name=model.model_name,
        training_run_id=training_run_id,
        modal_environment=os.environ.get("MODAL_ENVIRONMENT"),
    )

    _caller_module, caller_script = resolve_caller_context()
    grpo_image = build_grpo_image(spindle, dataset, eval_dataset, caller_script)
    spindle_image = build_spindle_image(spindle)

    checkpoints_volume_name, checkpoints_mount_path, all_volumes = (
        shared.create_training_volumes(checkpoint, volume_prefix=volume_prefix)
    )
    hf_cache_volume = all_volumes[str(HF_CACHE_PATH)]
    data_volume = all_volumes[str(DATA_PATH)]
    checkpoints_volume = all_volumes[checkpoints_mount_path]
    checkpoint_dir = compute_recipe_save_root(
        spindle,
        recipe_default_save_root=str(CHECKPOINTS_PATH),
        mounted_save_root=checkpoints_mount_path,
        training_run_id=training_run_id,
    )
    recorded_checkpoint_dir = checkpoint_dir if configured_recipe_save(spindle) else ""

    tags = build_app_tags(
        framework=Framework.SPINDLE.value,
        model=model,
        recipe_app_tags=spindle.app_tags,
        metrics=spindle.metrics,
    )
    app = App(app_name, tags=tags)

    def download_tokenizer() -> None:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(model.model_name, trust_remote_code=True)

    register_recipe_functions(
        app,
        grpo_image,
        hf_cache_volume=hf_cache_volume,
        data_volume=data_volume,
        checkpoints_volume=checkpoints_volume,
        checkpoints_mount_path=checkpoints_mount_path,
        download_phase=SpindleStatus.DOWNLOAD_MODEL.value,
        download=download_tokenizer,
        download_timeout=spindle.download_timeout_seconds or 60 * 60,
        prepare_dataset=lambda: write_datasets(
            dataset, eval_dataset, dataset_path, eval_dataset_path
        ),
        dataset_timeout=4 * 60 * 60,
    )

    @app.function(
        image=spindle_image,
        timeout=DEPLOY_TIMEOUT,
        secrets=[*hf_secrets()],
        serialized=True,
        name="deploy_spindle",
    )
    def deploy_spindle(values_json: str) -> str:
        return deploy_spindle_deployment(json.loads(values_json))

    api_secret = Secret.from_name(
        spindle.spindle_api_secret, required_keys=["TINKER_API_KEY"]
    )
    train_kwargs = dict(spindle.train_function_kwargs or {})
    user_secrets = train_kwargs.pop("secrets", None) or []
    if not isinstance(user_secrets, (list, tuple)):
        user_secrets = [user_secrets]

    @app.function(
        image=grpo_image,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            str(DATA_PATH): data_volume,
            checkpoints_mount_path: checkpoints_volume,
        },
        timeout=24 * 60 * 60,
        memory=spindle.memory,
        cpu=spindle.cpu,
        cloud=spindle.cloud,
        region=spindle.region,
        secrets=[api_secret, *hf_secrets(), *proxy_auth_secrets(), *user_secrets],
        serialized=True,
        name="train",
        **train_kwargs,
    )
    async def train(
        modal_app_id: str = "",
        modal_app_url: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        if framework_status_url:
            os.environ["MODAL_DOJO_FRAMEWORK_STATUS_URL"] = framework_status_url
        if framework_status_token:
            os.environ["MODAL_DOJO_FRAMEWORK_STATUS_TOKEN"] = framework_status_token
        metric_entity = preflight_metric(spindle.metrics)
        print(f"Training run id: {training_run_id}")
        run_record: TrainingRun
        (
            run_record,
            metric_run_id,
            framework_status_token,
        ) = await init_training_run_record(
            training_run_id=training_run_id,
            modal_app_id=modal_app_id,
            modal_app_url=modal_app_url,
            framework=Framework.SPINDLE,
            initializing_status=SpindleStatus.INITIALIZING,
            recipe=spindle,
            model=model,
            dataset=dataset,
            eval_dataset=eval_dataset,
            dataset_path=dataset_path,
            eval_dataset_path=eval_dataset_path,
            recipe_metadata=("gpu_type",),
            metric_entity=metric_entity,
            framework_status_token=framework_status_token,
            checkpoint_dir=recorded_checkpoint_dir,
            checkpoints_volume_name=checkpoints_volume_name,
            checkpoints_mount_path=checkpoints_mount_path,
        )

        async with shared.training_run_lifecycle(
            run_record, framework_status_token
        ) as set_status:
            await set_status(SpindleStatus.PREPARE_DATASET)
            if write_datasets(dataset, eval_dataset, dataset_path, eval_dataset_path):
                await data_volume.commit.aio()

            await set_status(SpindleStatus.DEPLOY_SPINDLE)
            base_url = await deploy_spindle.remote.aio(json.dumps(config_values))
            print(f"Spindle frontend: {base_url}")

            os.makedirs(checkpoint_dir, exist_ok=True)
            train_attn, train_mlp, train_unembed = lora_target_flags(
                spindle.target_modules
            )
            settings = GrpoSettings(
                model_name=model.model_name,
                base_url=base_url,
                log_path=os.path.join(checkpoint_dir, "grpo"),
                dataset_path=dataset_path,
                eval_dataset_path=eval_dataset_path,
                input_key=dataset.input_key(),
                label_key=dataset.label_key(),
                apply_chat_template=dataset.apply_chat_template(),
                chat_template_kwargs=_chat_template_kwargs(spindle),
                custom_rm_path=(spindle.extra_config or {}).get("custom_rm_path")
                or DEFAULT_REWARD_PATH,
                num_rollout=spindle.num_rollout,
                rollout_batch_size=spindle.rollout_batch_size,
                n_samples_per_prompt=spindle.n_samples_per_prompt,
                n_samples_per_eval_prompt=spindle.n_samples_per_eval_prompt,
                rollout_max_response_len=spindle.rollout_max_response_len,
                rollout_temperature=spindle.rollout_temperature,
                rollout_shuffle=spindle.rollout_shuffle,
                rollout_stop_token_ids=list(spindle.rollout_stop_token_ids or []),
                max_context_length=spindle.max_context_length,
                lr=spindle.lr,
                lora_rank=spindle.lora_rank or 32,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
                num_substeps=spindle.grpo_num_substeps,
                kl_penalty_coef=spindle.grpo_kl_penalty_coef,
                save_every=spindle.save_interval or 0,
                eval_every=spindle.eval_interval or 0,
                remove_constant_reward_groups=spindle.remove_constant_reward_groups,
                wandb_name=metric_run_id or None,
                config_overrides=dict(spindle.grpo_config_overrides),
            )
            print(
                f"Training {app_name} with tinker-cookbook GRPO: "
                f"{spindle.num_rollout} steps x {spindle.rollout_batch_size} prompts "
                f"x {spindle.n_samples_per_prompt} samples"
            )
            await set_status(SpindleStatus.TRAINING)
            await run_grpo(settings)
            await checkpoints_volume.commit.aio()
            return await shared.complete_training_run(
                run_record,
                checkpoints_volume=checkpoints_volume,
                app_name=app_name,
                model=model,
                group_id=group_id,
            )

    app.deploy_spindle = deploy_spindle
    app.train = train
    return app
