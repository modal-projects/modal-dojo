"""Run tinker-cookbook RL against a Spindle server."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.errors import DojoConfigError

if TYPE_CHECKING:
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
    which a retried driver resumes.
    """

    def __init__(
        self,
        config: Config,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        detach: bool = True,
        max_retries: int = 3,
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
        if base_url and config.base_url and base_url != config.base_url:
            raise DojoConfigError("base_url conflicts with config.base_url")
        base_url = base_url or config.base_url or SHARED_SPINDLE_URL
        api_key = api_key or os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise DojoConfigError(
                "Pass api_key or set TINKER_API_KEY for the Spindle server"
            )
        if max_retries < 0:
            raise DojoConfigError("max_retries must be non-negative")
        # launch() checks the model and context length against the server.
        self.program = CookbookRL(chz.replace(config, base_url=base_url))
        self.base_url = base_url
        self.detach = detach
        self.max_retries = max_retries
        self.cpu = cpu
        self.memory = memory
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


# Keep the experimental v0 out of the published API reference.
exclude_from_api_reference(SpindleRecipe)
