"""Run a raw tinker-cookbook RL configuration against a Spindle server."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from synchronicity import Synchronizer

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.errors import DojoConfigError

if TYPE_CHECKING:
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
    """

    def __init__(
        self,
        rl_config: Config,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
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
        if base_url and rl_config.base_url and base_url != rl_config.base_url:
            raise DojoConfigError("base_url conflicts with rl_config.base_url")
        api_key = api_key or os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise DojoConfigError(
                "Pass api_key or set TINKER_API_KEY for the Spindle server"
            )
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
        # The server checks the model, context window and LoRA rank at train time.
        self._rl_config = chz.replace(
            rl_config, base_url=base_url or rl_config.base_url or SHARED_SPINDLE_URL
        )
        self._api_key = api_key

    @property
    def rl_config(self) -> Config:
        return self._rl_config

    async def train(self) -> TrainingRun:
        """Return a TrainingRun; use ``await recipe.train.aio()`` in async code."""
        from modal_dojo.frameworks.spindle.launcher import train

        return await train(self.rl_config, api_key=self._api_key)


_synchronizer = Synchronizer()
SpindleRecipe = _synchronizer.create_blocking(
    _SpindleRecipe, name="SpindleRecipe", target_module=__name__
)
# Keep the experimental v0 out of the published API reference.
exclude_from_api_reference(SpindleRecipe)
