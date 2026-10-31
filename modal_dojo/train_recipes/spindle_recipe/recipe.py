"""Run a raw tinker-cookbook RL configuration against a Spindle server."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from synchronicity import Synchronizer

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.errors import DojoConfigError

if TYPE_CHECKING:
    from spindle.configuration import BaseConfig
    from tinker_cookbook.rl.train import Config
    from modal_dojo.common.run import TrainingRun

# Modal's always-on internal Spindle server.
SHARED_SPINDLE_URL = "https://modal-labs-spindle--prod.modal.run"


class _SpindleRecipe:
    """Run tinker-cookbook LoRA RL locally against a Spindle server.

    Requires ``uv sync --extra spindle``. Defaults to the shared Spindle
    server; pass ``base_url`` (or set ``rl_config.base_url``) to use another.
    ``api_key`` defaults to ``TINKER_API_KEY``. Requests carry the Modal proxy
    token from ``MODAL_KEY``/``MODAL_SECRET``, which must be allowed for the
    server's environment. Returns a ``TrainingRun`` with trajectories and
    metrics in Dojo; checkpoints stay on the server.

    As a fallback, ``deploy_config`` (a Spindle ``BaseConfig`` instance or
    subclass) deploys a fresh server in the active Modal environment and stops
    it on exit. This needs Python 3.12, ``uv sync --extra spindle-deploy`` and
    a sampler proxy secret, and cannot resume across deployments.
    """

    def __init__(
        self,
        rl_config: Config,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        deploy_config: BaseConfig | type[BaseConfig] | None = None,
    ):
        try:
            import chz
            from tinker_cookbook.rl.train import Config
        except ImportError as exc:
            raise ImportError(
                "SpindleRecipe requires modal-dojo[spindle]. "
                "Install with `uv sync --extra spindle` in this repository."
            ) from exc
        if not isinstance(rl_config, Config):
            raise DojoConfigError("rl_config must be tinker_cookbook.rl.train.Config")
        if not callable(rl_config.dataset_builder):
            raise DojoConfigError(
                "dataset_builder must be a tinker-cookbook builder; convert DatasetConfig "
                "with DojoDatasetBuilder.from_dataset(...) first"
            )
        if (
            rl_config.async_config is not None
            or rl_config.stream_minibatch_config is not None
        ):
            raise DojoConfigError(
                "SpindleRecipe v0 supports synchronous tinker-cookbook RL only"
            )
        if rl_config.max_tokens <= 0:
            raise DojoConfigError("max_tokens must be positive")
        if rl_config.max_steps is not None and rl_config.max_steps <= 0:
            raise DojoConfigError("max_steps must be positive")
        if deploy_config is not None:
            if base_url or api_key or rl_config.base_url:
                raise DojoConfigError(
                    "deploy_config deploys its own server; omit base_url and api_key"
                )
            self._rl_config = rl_config
            self._api_key = None
            self._deploy_config = _validated_deploy_config(rl_config, deploy_config)
            return
        if base_url and rl_config.base_url and base_url != rl_config.base_url:
            raise DojoConfigError("base_url conflicts with rl_config.base_url")
        api_key = api_key or os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise DojoConfigError(
                "Pass api_key or set TINKER_API_KEY for the Spindle server"
            )
        # The server checks the model, context window and LoRA rank at train time.
        self._rl_config = chz.replace(
            rl_config, base_url=base_url or rl_config.base_url or SHARED_SPINDLE_URL
        )
        self._api_key = api_key
        self._deploy_config = None

    @property
    def rl_config(self) -> Config:
        return self._rl_config

    @property
    def deploy_config(self) -> BaseConfig | None:
        return self._deploy_config

    async def train(self) -> TrainingRun:
        """Return a TrainingRun; use ``await recipe.train.aio()`` in async code."""
        from modal_dojo.frameworks.spindle.launcher import train

        return await train(
            self.rl_config, api_key=self._api_key, deploy_config=self.deploy_config
        )


def _validated_deploy_config(
    rl_config: Config, deploy_config: BaseConfig | type[BaseConfig]
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
    if isinstance(deploy_config, type) and issubclass(deploy_config, BaseConfig):
        deploy_config = deploy_config()
    if not isinstance(deploy_config, BaseConfig):
        raise DojoConfigError("deploy_config must be a Spindle BaseConfig or subclass")
    deploy_config = copy.deepcopy(deploy_config)
    if rl_config.model_name != deploy_config.model:
        raise DojoConfigError("RL model_name must match deploy_config.model")
    if deploy_config.parameterization != "lora":
        raise DojoConfigError("SpindleRecipe v0 supports LoRA training only")
    if rl_config.max_tokens >= deploy_config.max_context_length:
        raise DojoConfigError(
            "max_tokens must leave room for prompts in the context window"
        )
    compiled = DeploymentConfig.create(deploy_config)
    max_rank = compiled.trainer_settings.get(deploy_config.backend, {}).get(
        "max_lora_rank", 32
    )
    if not 0 < rl_config.lora_rank <= max_rank:
        raise DojoConfigError(f"lora_rank must be between 1 and {max_rank}")
    # Checkpoints live with the deployment, which stops when the run ends.
    checkpoints = Path(rl_config.log_path) / "checkpoints.jsonl"
    if rl_config.load_checkpoint_path is not None or (
        checkpoints.exists() and checkpoints.stat().st_size
    ):
        raise DojoConfigError(
            "Use a fresh log_path without load_checkpoint_path: a deployed "
            "Spindle cannot resume another deployment's checkpoints"
        )
    return deploy_config


_synchronizer = Synchronizer()
SpindleRecipe = _synchronizer.create_blocking(
    _SpindleRecipe, name="SpindleRecipe", target_module=__name__
)
# Keep the experimental v0 out of the published API reference.
exclude_from_api_reference(SpindleRecipe)
