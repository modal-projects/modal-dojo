"""Run a raw Tinker Cookbook RL configuration against an owned Spindle server."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from modal_dojo.common.errors import DojoConfigError

if TYPE_CHECKING:
    from spindle.configuration import BaseConfig
    from tinker_cookbook.rl.train import Config


class SpindleRecipe:
    """Minimal LoRA RL launcher; the cookbook driver runs in the caller process.

    Install from a source checkout with ``uv sync --extra spindle`` and Python
    3.12 (Spindle is not yet on PyPI). ``deploy_config`` is a
    Spindle ``BaseConfig`` instance or subclass. Each call deploys its own server,
    trainer, and sampler, then stops those apps on normal exit or failure.
    Checkpoints remain in Spindle's configured volume. The platform must name
    an existing sampler proxy secret. Hugging Face auth is optional for public
    models; a missing HF secret is omitted.

    Raw trajectories are printed and appended to ``log_path/dojo-trajectories.jsonl``.
    This v0 does not return a Dojo ``TrainingRun`` or ingest dashboard metadata.
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
        self.rl_config = rl_config
        self.deploy_config = copy.deepcopy(deploy_config)
        if rl_config.model_name != deploy_config.model:
            raise DojoConfigError("RL model_name must match deploy_config.model")
        if deploy_config.parameterization != "lora":
            raise DojoConfigError("SpindleRecipe v0 supports LoRA training only")
        if not callable(rl_config.dataset_builder):
            raise DojoConfigError(
                "dataset_builder must be a cookbook builder; convert DatasetConfig "
                "with DojoDatasetBuilder.from_dataset(...) first"
            )
        if (
            rl_config.async_config is not None
            or rl_config.stream_minibatch_config is not None
        ):
            raise DojoConfigError(
                "SpindleRecipe v0 supports synchronous cookbook RL only"
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

    async def train(self) -> None:
        """Deploy Spindle, await cookbook RL, and clean up this run's apps."""
        from modal_dojo.frameworks.spindle.launcher import train

        await train(self.rl_config, self.deploy_config)
