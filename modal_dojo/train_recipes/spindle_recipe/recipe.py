"""Run a raw tinker-cookbook RL configuration against an owned Spindle server."""

from __future__ import annotations

import copy
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


class _SpindleRecipe:
    """Run tinker-cookbook LoRA RL locally against a managed Spindle deployment.

    Requires Python 3.12, ``uv sync --extra spindle``, and a sampler proxy secret.
    ``deploy_config`` accepts a Spindle ``BaseConfig`` instance or subclass.
    Returns a ``TrainingRun`` with trajectories and metrics in Dojo. Backend
    apps stop on exit; checkpoints remain on Spindle's volume.
    """

    def __init__(
        self, rl_config: Config, *, deploy_config: BaseConfig | type[BaseConfig]
    ):
        try:
            from spindle.configuration import BaseConfig
            from spindle.deployments import DeploymentConfig
            from tinker_cookbook.rl.train import Config
        except ImportError as exc:
            raise ImportError(
                "SpindleRecipe requires Python 3.12 and modal-dojo[spindle]. "
                "Install with `uv sync --extra spindle` in this repository."
            ) from exc
        if sys.version_info[:2] != (3, 12):
            raise DojoConfigError("SpindleRecipe requires Python 3.12")
        if not isinstance(rl_config, Config):
            raise DojoConfigError("rl_config must be tinker_cookbook.rl.train.Config")
        if isinstance(deploy_config, type) and issubclass(deploy_config, BaseConfig):
            deploy_config = deploy_config()
        if not isinstance(deploy_config, BaseConfig):
            raise DojoConfigError(
                "deploy_config must be a Spindle BaseConfig or subclass"
            )
        self._rl_config = rl_config
        self._deploy_config = copy.deepcopy(deploy_config)
        if rl_config.model_name != deploy_config.model:
            raise DojoConfigError("RL model_name must match deploy_config.model")
        if deploy_config.parameterization != "lora":
            raise DojoConfigError("SpindleRecipe v0 supports LoRA training only")
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
        if rl_config.load_checkpoint_path is not None:
            raise DojoConfigError(
                "SpindleRecipe v0 cannot resume across owned deployments"
            )
        checkpoints = Path(rl_config.log_path) / "checkpoints.jsonl"
        if checkpoints.exists() and checkpoints.stat().st_size:
            raise DojoConfigError(
                "Use a fresh log_path: Spindle checkpoints belong to their deployment; "
                "v0 cannot resume them in a new owned deployment"
            )
        if rl_config.base_url is not None:
            raise DojoConfigError("Omit base_url: SpindleRecipe deploys its own server")
        if (
            rl_config.max_tokens <= 0
            or rl_config.max_tokens >= deploy_config.max_context_length
        ):
            raise DojoConfigError(
                "max_tokens must leave room for prompts in the context window"
            )
        if rl_config.max_steps is not None and rl_config.max_steps <= 0:
            raise DojoConfigError("max_steps must be positive")
        compiled = DeploymentConfig.create(self.deploy_config)
        max_rank = compiled.trainer_settings.get(deploy_config.backend, {}).get(
            "max_lora_rank", 32
        )
        if not 0 < rl_config.lora_rank <= max_rank:
            raise DojoConfigError(f"lora_rank must be between 1 and {max_rank}")

    @property
    def rl_config(self) -> Config:
        return self._rl_config

    @property
    def deploy_config(self) -> BaseConfig:
        return self._deploy_config

    async def train(self) -> TrainingRun:
        """Return a TrainingRun; use ``await recipe.train.aio()`` in async code."""
        from modal_dojo.frameworks.spindle.launcher import train

        return await train(self.rl_config, self.deploy_config)


_synchronizer = Synchronizer()
SpindleRecipe = _synchronizer.create_blocking(
    _SpindleRecipe, name="SpindleRecipe", target_module=__name__
)
# Keep the experimental v0 out of the published API reference.
exclude_from_api_reference(SpindleRecipe)
