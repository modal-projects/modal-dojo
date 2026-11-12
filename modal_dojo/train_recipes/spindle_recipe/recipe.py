"""Run Tinker training code against a Spindle server."""

from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.errors import DojoConfigError

if TYPE_CHECKING:
    from tinker_cookbook.rl.train import Config

    from modal_dojo.common.run import TrainingRun
    from modal_dojo.tinker import TinkerProgram

# Modal's always-on internal Spindle server.
SHARED_SPINDLE_URL = "https://modal-labs-spindle--prod.modal.run"


class SpindleRecipe:
    """Run Tinker training code against a Spindle server, driven from Modal.

    ``program`` is a tinker-cookbook ``rl_train.Config``, or a ``TinkerProgram``
    whose ``run(client, ctx)`` is your own training loop. Requires
    ``uv sync --extra spindle``. Like Slime and Miles, ``launch()`` starts the
    driver in a detached Modal app and returns a ``TrainingRun`` handle;
    ``train()`` also waits for it. Defaults to the shared Spindle server; pass
    ``base_url`` (or set ``rl_config.base_url``) to use another. ``api_key``
    defaults to ``TINKER_API_KEY``. Requests carry the Modal proxy token from
    ``MODAL_KEY``/``MODAL_SECRET``, which must be allowed for the server's
    environment. Checkpoints stay on the server. A tinker-cookbook config's
    ``log_path`` is replaced by the run's directory on the
    ``spindle-checkpoints`` volume, from which a retried driver resumes, so its
    ``max_retries`` defaults to 3; a program's defaults to 0.
    """

    def __init__(
        self,
        program: Config | TinkerProgram,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
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
        from modal_dojo.tinker import TinkerProgram

        if isinstance(program, Config):
            if base_url and program.base_url and base_url != program.base_url:
                raise DojoConfigError("base_url conflicts with rl_config.base_url")
            base_url = base_url or program.base_url or SHARED_SPINDLE_URL
            program = CookbookRL(chz.replace(program, base_url=base_url))
            default_retries = 3
        elif isinstance(program, TinkerProgram):
            if not inspect.iscoroutinefunction(program.run):
                raise DojoConfigError("TinkerProgram.run must be an async method")
            base_url = base_url or SHARED_SPINDLE_URL
            # A retry reruns run() from the start unless it resumes from ctx.log_dir.
            default_retries = 0
        else:
            raise DojoConfigError(
                "program must be a tinker-cookbook rl_train.Config or a TinkerProgram"
            )
        api_key = api_key or os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise DojoConfigError(
                "Pass api_key or set TINKER_API_KEY for the Spindle server"
            )
        max_retries = default_retries if max_retries is None else max_retries
        if max_retries < 0:
            raise DojoConfigError("max_retries must be non-negative")
        # launch() checks the model (and a tinker-cookbook config's context) against the server.
        self.program = program
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
