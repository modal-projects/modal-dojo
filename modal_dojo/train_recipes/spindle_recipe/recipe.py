"""Run tinker-cookbook RL against a Spindle server."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.errors import DojoConfigError

if TYPE_CHECKING:
    import modal
    from tinker_cookbook.rl.train import Config

    from modal_dojo.common.run import TrainingRun
    from modal_dojo.frameworks.spindle.cookbook import CookbookRL


class SpindleRecipe:
    """Run a tinker-cookbook RL config against a Spindle server, driven from Modal.

    ``program`` is a tinker-cookbook ``rl_train.Config`` or its
    ``modal_dojo.tinker.cookbook_rl`` wrapper. Requires
    ``uv sync --extra spindle``. Like Slime and Miles, ``launch()`` starts the
    driver in a detached Modal app and returns a ``TrainingRun`` handle;
    ``train()`` also waits for it. Pass ``base_url`` and ``api_key`` to connect
    directly, or reuse the saved connection / versioned Spindle app in the
    active Modal environment. Creating a server requires confirmation or
    ``train(yes=True)``. Spindle model backends are reused or added as needed;
    the server and its backends persist after the run. ``context_length`` is
    total sequence capacity; GPU overrides constrain the backend selection.
    Provisioning requires the ``spindle-deploy`` extra and Python 3.12.
    Checkpoints stay on the server. The config's ``log_path`` is
    replaced by the run's directory on the ``spindle-checkpoints`` volume, from
    which a retried driver resumes.
    """

    def __init__(
        self,
        program: CookbookRL | Config,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        context_length: int | None = None,
        trainer_gpu: str | None = None,
        trainer_gpus_per_node: int | None = None,
        trainer_nodes: int | None = None,
        inference_gpu: str | None = None,
        inference_gpus_per_node: int | None = None,
        detach: bool = True,
        max_retries: int = 3,
        cpu: float = 4.0,
        memory: int = 8192,
        image_overlay: Callable[[modal.Image], modal.Image] | None = None,
    ):
        try:
            from tinker_cookbook.rl.train import Config
        except ImportError as exc:
            raise ImportError(
                "SpindleRecipe requires modal-dojo[spindle]. "
                "Install with `uv sync --extra spindle` in this repository."
            ) from exc
        from modal_dojo.frameworks.spindle.cookbook import CookbookRL

        if isinstance(program, Config):
            program = CookbookRL(program)
        if not isinstance(program, CookbookRL):
            raise DojoConfigError(
                "Pass a tinker-cookbook rl_train.Config or cookbook_rl(config)."
            )
        config = program.config
        if base_url and config.base_url and base_url != config.base_url:
            raise DojoConfigError("base_url conflicts with the config's base_url")
        base_url = base_url or config.base_url or os.environ.get("TINKER_BASE_URL")
        if api_key and not base_url:
            raise DojoConfigError("An explicit api_key requires a base_url.")
        api_key = (api_key or os.environ.get("TINKER_API_KEY")) if base_url else None
        overrides = {
            key: value
            for key, value in {
                "trainer_gpu": trainer_gpu,
                "trainer_gpus_per_node": trainer_gpus_per_node,
                "trainer_nodes": trainer_nodes,
                "inference_gpu": inference_gpu,
                "inference_gpus_per_node": inference_gpus_per_node,
            }.items()
            if value is not None
        }
        for name, value in {"context_length": context_length, **overrides}.items():
            if value is None:
                continue
            if name.endswith("_gpu"):
                if not isinstance(value, str) or not value.strip():
                    raise DojoConfigError(f"{name} must be a GPU type such as 'H100'.")
            elif type(value) is not int or value <= 0:
                raise DojoConfigError(f"{name} must be a positive integer.")
        if context_length is not None and config.max_tokens >= context_length:
            raise DojoConfigError(
                "max_tokens must leave room for prompts within context_length."
            )
        if max_retries < 0:
            raise DojoConfigError("max_retries must be non-negative")
        # launch() checks the model and context length against the server.
        from modal_dojo.frameworks.spindle.deployment import Requirements

        self.program = program
        self.requirements = Requirements(
            program.model_name,
            config.max_tokens,
            config.lora_rank,
            context_length,
            overrides,
        )
        self.base_url = base_url
        self.detach = detach
        self.max_retries = max_retries
        self.cpu = cpu
        self.memory = memory
        self.image_overlay = image_overlay
        self._api_key = api_key

    def launch(self, *, show_output: bool = True, yes: bool = False) -> TrainingRun:
        """Start training in a detached Modal app and return its handle."""
        from modal_dojo.frameworks.spindle.app import launch

        return launch(self, show_output=show_output, yes=yes)

    def train(self, *, show_output: bool = True, yes: bool = False) -> TrainingRun:
        """Launch, then wait for the finished TrainingRun.

        Ctrl-C stops waiting; the run continues unless ``detach=False``.
        """
        from modal_dojo.common.modal_lifecycle import stop_app

        run = self.launch(show_output=show_output, yes=yes)
        try:
            return run.result(stop_app_on_success=True)
        except BaseException:
            if not self.detach and run.modal_app_id:
                stop_app(run.modal_app_id)
            raise


# Keep the experimental v0 out of the published API reference.
exclude_from_api_reference(SpindleRecipe)
