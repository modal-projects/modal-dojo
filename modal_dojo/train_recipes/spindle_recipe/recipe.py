"""Run tinker-cookbook RL against a Spindle server."""

from __future__ import annotations

import copy
import os
import sys
from typing import TYPE_CHECKING

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.config import modal_proxy_auth_headers
from modal_dojo.common.errors import DojoConfigError

if TYPE_CHECKING:
    from spindle.configuration import BaseConfig
    from tinker_cookbook.rl.train import Config

    from modal_dojo.common.run import TrainingRun

# Modal's always-on internal Spindle server.
SHARED_SPINDLE_URL = "https://modal-labs-spindle--prod.modal.run"


class SpindleRecipe:
    """Run a tinker-cookbook RL config against a Spindle server, driven from Modal.

    ``config`` is a tinker-cookbook ``rl_train.Config``. Requires
    ``uv sync --extra spindle``. Like Slime and Miles, ``launch()`` starts the
    driver in a detached Modal app and returns a ``TrainingRun`` handle;
    ``train()`` also waits for it. Defaults to the shared Spindle server; pass
    ``base_url`` (or set ``config.base_url``) to use another. ``api_key``
    defaults to ``TINKER_API_KEY``. Requests carry the Modal proxy token from
    ``MODAL_KEY``/``MODAL_SECRET``, which must be allowed for the server's
    environment. Checkpoints stay on the server. The config's ``log_path`` is
    replaced by the run's directory on the ``spindle-checkpoints`` volume, from
    which a retried driver resumes, so ``max_retries`` defaults to 3.

    As a fallback, ``deploy_config`` (a Spindle ``BaseConfig`` instance or
    subclass) has the driver deploy a fresh server in the active Modal
    environment and stop it when the run ends. Validating it locally needs
    ``uv sync --extra spindle-deploy``; deploying needs a sampler proxy
    secret. Its API requires the same Modal proxy auth, and since it cannot
    resume across deployments, it runs without retries.
    """

    def __init__(
        self,
        config: Config,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        deploy_config: BaseConfig | type[BaseConfig] | None = None,
        detach: bool = True,
        max_retries: int | None = None,
        cpu: float = 4.0,
        memory: int = 8192,
    ):
        try:
            import chz
            from tinker_cookbook.rl.train import Config
        except ImportError as exc:
            raise ImportError(
                "SpindleRecipe requires modal-dojo[spindle]. "
                "Install with `uv sync --extra spindle` in this repository."
            ) from exc
        from modal_dojo.frameworks.spindle.cookbook import CookbookRL

        if not isinstance(config, Config):
            raise DojoConfigError("config must be a tinker-cookbook rl_train.Config")
        if deploy_config is not None:
            if base_url or api_key or config.base_url:
                raise DojoConfigError(
                    "deploy_config deploys its own server; omit base_url and api_key"
                )
            if max_retries:
                raise DojoConfigError(
                    "deploy_config runs without retries: a retry would deploy "
                    "another server without this run's checkpoints"
                )
            max_retries = 0
        else:
            if base_url and config.base_url and base_url != config.base_url:
                raise DojoConfigError("base_url conflicts with config.base_url")
            base_url = base_url or config.base_url or SHARED_SPINDLE_URL
            api_key = api_key or os.environ.get("TINKER_API_KEY")
            if not api_key:
                raise DojoConfigError(
                    "Pass api_key or set TINKER_API_KEY for the Spindle server"
                )
            max_retries = 3 if max_retries is None else max_retries
        if max_retries < 0:
            raise DojoConfigError("max_retries must be non-negative")
        program = CookbookRL(chz.replace(config, base_url=base_url))
        if deploy_config is not None:
            deploy_config = _validated_deploy_config(program, deploy_config)
        # launch() checks the model and context length against the server.
        self.program = program
        self.base_url = base_url
        self.detach = detach
        self.max_retries = max_retries
        self.cpu = cpu
        self.memory = memory
        self.deploy_config = deploy_config
        self._api_key = api_key

    def launch(self, *, show_output: bool = True) -> TrainingRun:
        """Start training in a detached Modal app and return its handle."""
        from modal_dojo.frameworks.spindle.app import launch

        return launch(self, show_output=show_output)

    def train(self, *, show_output: bool = True) -> TrainingRun:
        """Launch, then wait for the finished TrainingRun.

        Ctrl-C stops waiting; the run continues unless ``detach=False``.
        """
        from modal_dojo.common.modal_lifecycle import stop_app

        run = self.launch(show_output=show_output)
        try:
            return run.result(stop_app_on_success=True)
        except BaseException:
            if not self.detach and run.modal_app_id:
                stop_app(run.modal_app_id)
            raise


def _validated_deploy_config(
    program, deploy_config: BaseConfig | type[BaseConfig]
) -> BaseConfig:
    """Check a deploy-on-the-fly config locally, before requesting GPUs."""
    try:
        from spindle.configuration import BaseConfig
        from spindle.deployments import DeploymentConfig
    except ImportError as exc:
        raise ImportError(
            "Deploying Spindle requires Python 3.12 and modal-dojo[spindle-deploy]. "
            "Install with `uv sync --extra spindle-deploy` in this repository."
        ) from exc
    if sys.version_info[:2] != (3, 12):
        raise DojoConfigError("Deploying Spindle requires Python 3.12")
    if not modal_proxy_auth_headers():
        raise DojoConfigError(
            "A deployed Spindle requires Modal proxy auth: set MODAL_KEY/MODAL_SECRET "
            "to a proxy token allowed for the active Modal environment"
        )
    if isinstance(deploy_config, type) and issubclass(deploy_config, BaseConfig):
        deploy_config = deploy_config()
    if not isinstance(deploy_config, BaseConfig):
        raise DojoConfigError("deploy_config must be a Spindle BaseConfig or subclass")
    deploy_config = copy.deepcopy(deploy_config)
    if program.model_name != deploy_config.model:
        raise DojoConfigError("model_name must match deploy_config.model")
    if deploy_config.parameterization != "lora":
        raise DojoConfigError("SpindleRecipe v0 supports LoRA training only")
    config = program.config
    if config.max_tokens >= deploy_config.max_context_length:
        raise DojoConfigError(
            "max_tokens must leave room for prompts in the context window"
        )
    compiled = DeploymentConfig.create(deploy_config)
    max_rank = compiled.trainer_settings.get(deploy_config.backend, {}).get(
        "max_lora_rank", 32
    )
    if not 0 < config.lora_rank <= max_rank:
        raise DojoConfigError(f"lora_rank must be between 1 and {max_rank}")
    # Checkpoints live with the deployment, which stops when the run ends.
    if config.load_checkpoint_path is not None:
        raise DojoConfigError(
            "Omit load_checkpoint_path: a deployed Spindle cannot resume "
            "another deployment's checkpoints"
        )
    return deploy_config


# Keep the experimental v0 out of the published API reference.
exclude_from_api_reference(SpindleRecipe)
