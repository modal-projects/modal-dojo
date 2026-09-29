import dataclasses as _dc
import os
import secrets as _secrets
import sys
import threading
import time
import warnings
from contextlib import nullcontext
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.checkpoint import Checkpoint, CheckpointType
from modal_training_gym.common.dataset import DatasetConfig, OnlineRollout
from modal_training_gym.common.errors import (
    TrainingGymConfigError,
    TrainingGymError,
)
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.ids import create_hash
from modal_training_gym.common.launcher_helpers import mark_run_failed, mark_run_stopped
from modal_training_gym.common.memory_estimate import maybe_warn_gpu_oom
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.run import TrainingRun, metric_run_id_for_attempt
from modal_training_gym.common.status import (
    FrameworkStatus,
    MilesStatus,
    SlimeStatus,
)
from modal_training_gym.frameworks.miles import build_miles_app
from modal_training_gym.frameworks.slime import build_slime_app
from modal_training_gym.train_recipes.base import BaseTrainRecipe
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe
from modal_training_gym.utils.metadata import MetadataStore, vol_put


def _megatron_load_dir(checkpoint: Checkpoint) -> str:
    if checkpoint.checkpoint_type != CheckpointType.megatron:
        raise TrainingGymConfigError(
            "Training can only resume from a Megatron checkpoint; "
            "Hugging Face exports are serving artifacts."
        )
    return os.path.dirname(checkpoint.path.rstrip("/"))


def _try_validate_model_parallelism(
    recipe: BaseTrainRecipe, model: ModelConfig
) -> None:
    # Not every framework recipe implements this preflight.
    if validate := getattr(recipe, "validate_model_parallelism", None):
        validate(model)


def _terminalize_launch(run_record: TrainingRun, exc: BaseException) -> None:
    finished_at = int(time.time())
    if isinstance(exc, KeyboardInterrupt):
        mark_run_stopped(run_record)
    else:
        mark_run_failed(run_record, exc)
    run_record.ended_at = finished_at
    run_record.completed_at = finished_at
    if run_record.started_at:
        run_record.duration_seconds = max(0, finished_at - run_record.started_at)
    try:
        run_record.save()
    except Exception as save_exc:
        print(f"WARNING: could not save run {run_record.training_run_id}: {save_exc}")


def _convert_checkpoint_on_cache_miss(
    app: Any,
    *,
    training_run_id: str,
    framework_status_url: str,
    framework_status_token: str,
) -> bool:
    call_kwargs = {
        "training_run_id": training_run_id,
        "framework_status_url": framework_status_url,
        "framework_status_token": framework_status_token,
    }
    hf_path = app.resolve_checkpoint.remote(**call_kwargs)
    if hf_path is None:
        return False
    app.convert_checkpoint.remote(hf_path=hf_path, **call_kwargs)
    return True


def _warn_if_external_build_app() -> None:
    """Warn when ``_build_app()`` is called from outside the package.

    Hand-rolling ``_build_app()`` + ``app.run()`` + ``app.train.spawn()`` is a
    trap: the nested ``app.run()`` is ephemeral, so exiting the block (or
    Ctrl-C) stops the app and kills the spawned run — and ``modal run
    --detach`` does not help, since it only detaches the CLI's own entrypoint
    app. ``launch()`` / ``train()`` open the app with ``detach=True`` and
    persist the function-call id so the run can be waited on from anywhere.
    """
    try:
        caller = sys._getframe(2).f_globals.get("__name__", "")
    except ValueError:  # no such frame — never block a launch over a warning
        return
    if caller.startswith("modal_training_gym"):
        return
    warnings.warn(
        "TrainConfig._build_app() is private and does not start a detached "
        "app: spawning train() on it yourself means the run dies when the "
        "enclosing app.run() block exits or is interrupted. Use "
        "TrainConfig.launch() (returns a TrainingRun handle immediately) or "
        "TrainConfig.train() (blocks for the TrainingRun) instead.",
        stacklevel=3,
    )


_STAGE_LABELS: dict[str, str] = {
    SlimeStatus.INITIALIZING.value: "Initializing",
    SlimeStatus.DOWNLOAD_MODEL.value: "Downloading model",
    SlimeStatus.CONVERT_MODEL.value: "Converting model",
    SlimeStatus.PREPARE_DATASET.value: "Preparing dataset",
    SlimeStatus.ROLLOUT_INITIALIZING.value: "Initializing rollouts",
    SlimeStatus.ROLLOUT_LOGGING.value: "Generating rollouts",
    SlimeStatus.EVAL_ROLLOUT_LOGGING.value: "Evaluating rollouts",
    SlimeStatus.COMPUTE_LOG_PROBS.value: "Computing log probs",
    SlimeStatus.OPTIMIZER_STEP.value: "Optimizer step",
    SlimeStatus.WEIGHT_SYNC.value: "Weight sync",
    SlimeStatus.OFFLOAD_ROLLOUT.value: "Offload rollout",
    SlimeStatus.OFFLOAD_TRAIN.value: "Offload train",
    SlimeStatus.CHECKPOINT_SAVE.value: "Saving checkpoint",
    SlimeStatus.TRAINING.value: "Training",
}


class _TrainStatusDisplay:
    """Terminal status helper for ``train()``.

    Prints a static banner once at the start, then a single concise status
    line each time the stage changes. We deliberately do **not** use
    ``rich.Live`` here: ``modal.enable_output()`` drives its own ANSI cursor
    moves for the spinner and "Running app..." line, which fight with Live's
    redraw and produce stacked/truncated panels.
    """

    def __init__(
        self,
        run_id: str,
        framework: str,
        model_name: str,
        dataset_name: str,
        framework_status_url: str,
        config_path: str,
    ) -> None:
        self.run_id = run_id
        self.framework = framework
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.framework_status_url = framework_status_url
        self.config_path = config_path
        self.started_at: float = time.time()
        self._console = None  # lazily constructed
        self._modal_app_url: str = ""
        # Stage dedupe is shared between the local orchestrator (which emits
        # download/convert directly) and the background poller (which emits
        # in-container phases), so identical consecutive stages print once.
        self._stage_lock = threading.Lock()
        self._last_stage: str | None = None
        self._poll_stop: threading.Event | None = None
        self._poll_thread: threading.Thread | None = None

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _get_console(self):
        from rich.console import Console

        if self._console is None:
            self._console = Console()
        return self._console

    def print_banner(self) -> None:
        from rich.panel import Panel
        from rich.style import Style
        from rich.table import Table
        from rich.text import Text

        body = Table.grid(padding=(0, 2))
        body.add_column(style="dim", no_wrap=True)
        body.add_column(overflow="fold")

        body.add_row("Run", Text(self.run_id, style="bold cyan"))
        if self.model_name:
            body.add_row("Model", self.model_name)
        if self.dataset_name:
            body.add_row("Dataset", self.dataset_name)
        body.add_row("Framework", self.framework)

        if self.framework_status_url:
            base = self.framework_status_url.replace(
                "/api/framework-status", ""
            ).rstrip("/")
            run_url = f"{base}/training/{self.run_id}"
            body.add_row(
                "Dashboard",
                Text(
                    run_url,
                    style=Style(color="blue", underline=True, link=run_url),
                ),
            )
        else:
            body.add_row(
                "Dashboard",
                Text(
                    f"(run `training-gym setup` to populate {self.config_path})",
                    style="yellow",
                ),
            )

        self._get_console().print(
            Panel(
                body,
                title="[bold]Training Gym[/bold]",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def set_modal_app_url(self, url: str) -> None:
        if url and url != self._modal_app_url:
            self._modal_app_url = url

            from rich.style import Style
            from rich.text import Text

            self._get_console().print(
                Text.assemble(
                    ("Modal app:", "dim"),
                    " ",
                    (url, Style(color="blue", underline=True, link=url)),
                )
            )

    def emit_stage(self, stage: str, detail: str = "") -> None:
        # Dedupe on the phase only (not the progress detail) so we print one
        # line per phase transition rather than spamming a line per rollout
        # step. The local orchestrator and the poller share this guard.
        with self._stage_lock:
            if stage == self._last_stage:
                return
            self._last_stage = stage
        label = _STAGE_LABELS.get(stage, stage or "—")
        elapsed = self._format_elapsed(time.time() - self.started_at)
        suffix = f" [dim]{detail}[/dim]" if detail else ""
        # One short scrolling line per stage transition. Cyan ▶ marker
        # makes it easy to spot in a wall of Modal log output.
        self._get_console().print(
            f"[cyan]▶[/cyan] [dim]\\[{elapsed}][/dim] "
            f"[bold]{label}[/bold]{suffix} [dim]({self.run_id})[/dim]"
        )

    @staticmethod
    def _progress_detail(run: "TrainingRun") -> str:
        progress = (run.metadata or {}).get("framework_progress")
        if not isinstance(progress, dict):
            return ""
        current = progress.get("current")
        total = progress.get("total")
        if isinstance(current, int) and isinstance(total, int) and total > 0:
            return f"({current}/{total})"
        return ""

    def start_polling(self, training_run_id: str, interval: float = 4.0) -> None:
        """Track in-container phases while ``app.train.remote()`` blocks.

        The local orchestrator hands off to the remote container and then
        blocks for the whole run; without this, the terminal ``Stage`` line
        freezes at the last locally-emitted stage. The container POSTs its
        phases to the dashboard, which persists them onto the run record — so
        we poll that record and emit each new phase to the terminal.
        """
        if self._poll_thread is not None:
            return
        stop = threading.Event()

        def _poll() -> None:
            while not stop.is_set():
                try:
                    run = TrainingRun.from_id(training_run_id)
                except Exception:
                    run = None
                if run is not None and run.framework_status is not None:
                    self.emit_stage(
                        run.framework_status.value, self._progress_detail(run)
                    )
                stop.wait(interval)

        thread = threading.Thread(
            target=_poll, name="training-gym-status-poller", daemon=True
        )
        self._poll_stop = stop
        self._poll_thread = thread
        thread.start()

    def stop_polling(self) -> None:
        if self._poll_stop is not None:
            self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
        self._poll_stop = None
        self._poll_thread = None


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class TrainConfig:
    """A dataset, model, and recipe for one training run.

    Args:
        dataset:
            Training dataset materialized in the framework's ``/data`` volume.
        eval_dataset : DatasetConfig | None
            Optional dataset used by the framework's internal evaluation loop.
            It is materialized independently from the training dataset.
        model:
            Model identity and weight download behavior.
        recipe:
            Training framework, Modal resources, and framework arguments.
        resume_from_checkpoint:
            Start a new run from this Megatron checkpoint's weights with a
            fresh optimizer and LR schedule; ``recipe.num_rollout`` counts
            from zero. To continue the source run in place with its Adam
            state and iteration count, leave ``resume_from_checkpoint`` unset
            and point ``recipe.load`` at the checkpoint directory.
        detach:
            Keep training on Modal if the local ``train()`` wait is interrupted.
            ``False`` stops the app.
        group_id:
            Sweep ID assigned by ``TrainingGroup``.
        group_overrides:
            Variant overrides keyed by dotted field path.
        group_axes:
            Swept field paths that group dashboard variants.
    """

    # ── Composed configs (required) ─────────────────────────────────────────
    dataset: DatasetConfig
    model: ModelConfig
    recipe: SlimeRecipe | MilesRecipe
    eval_dataset: DatasetConfig | None = None
    resume_from_checkpoint: Checkpoint | None = None
    # Whether a run outlives the local client. The app itself is always started
    # detached (the CLI's ``modal run --detach`` only detaches the entrypoint,
    # not the nested ``app.run()`` the driver opens), so this only decides
    # whether an interrupted ``train()`` stops the app on its way out.
    detach: bool = True
    # Set by TrainingGroup so every run in a sweep shares one id — written into
    # the TrainingRun record so the dashboard can group variants together.
    group_id: str | None = None
    group_overrides: dict[str, Any] | None = None
    group_axes: list[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, OnlineRollout):
            return
        path = (self.recipe.extra_config or {}).get("custom_generate_function_path")
        if self.recipe.custom_generate_function is None and not isinstance(path, str):
            raise TrainingGymConfigError(
                "OnlineRollout requires recipe.custom_generate_function or "
                "recipe.extra_config['custom_generate_function_path']"
            )

    def _generate_training_run_id(self) -> str:
        """Mint a new run id. ``launch()`` calls this once per invocation, so
        each launch of the same config gets its own TrainingRun record."""
        return create_hash(
            self.model.model_name,
            self.resume_from_checkpoint.path
            if self.resume_from_checkpoint is not None
            else "",
            f"{type(self.recipe).__name__}:{self.framework.value}",
            "",
            self.model.model_path or "",
        )

    def _prepare_recipe(self) -> SlimeRecipe | MilesRecipe:
        if self.resume_from_checkpoint is None:
            recipe = _dc.replace(self.recipe)
        else:
            recipe = _dc.replace(
                self.recipe,
                load=_megatron_load_dir(self.resume_from_checkpoint),
                start_rollout_id=(
                    0
                    if self.recipe.start_rollout_id is None
                    else self.recipe.start_rollout_id
                ),
                no_load_optim=True,
            )
        _try_validate_model_parallelism(recipe, self.model)
        return recipe

    def _build_app(self, training_run_id: str | None = None):
        _warn_if_external_build_app()
        if training_run_id is None:
            training_run_id = self._generate_training_run_id()
        recipe = self._prepare_recipe()
        if isinstance(recipe, MilesRecipe):
            return build_miles_app(
                training_run_id=training_run_id,
                miles=recipe,
                model=self.model,
                dataset=self.dataset,
                eval_dataset=self.eval_dataset,
                checkpoint=self.resume_from_checkpoint,
                name=training_run_id,
                group_id=self.group_id,
            )
        if isinstance(recipe, SlimeRecipe):
            return build_slime_app(
                training_run_id=training_run_id,
                slime=recipe,
                model=self.model,
                dataset=self.dataset,
                eval_dataset=self.eval_dataset,
                checkpoint=self.resume_from_checkpoint,
                name=training_run_id,
                group_id=self.group_id,
            )
        raise TrainingGymConfigError(
            f"Unknown training recipe: {type(recipe).__name__}"
        )

    # ── Run-record helpers ─────────────────────────────────────────────────

    @property
    def framework(self) -> Framework:
        if isinstance(self.recipe, SlimeRecipe):
            return Framework.SLIME
        if isinstance(self.recipe, MilesRecipe):
            return Framework.MILES
        raise TrainingGymConfigError(
            f"Unknown training recipe: {type(self.recipe).__name__}"
        )

    def _initializing_status(self) -> FrameworkStatus:
        if self.framework is Framework.SLIME:
            return SlimeStatus.INITIALIZING
        if self.framework is Framework.MILES:
            return MilesStatus.INITIALIZING
        raise TrainingGymConfigError(f"Unknown training framework: {self.framework}")

    def _build_config_summary(self, training_run_id: str) -> dict[str, Any]:
        """Framework-specific TrainingRun.config summary."""
        model = self.model
        dataset = self.dataset
        recipe = self._prepare_recipe()

        metrics = getattr(recipe, "metrics", None)
        summary: dict[str, Any] = {
            "model": {"model_name": model.model_name} if model else {},
            "metrics": (
                metrics.metadata(
                    entity=getattr(metrics, "entity", ""),
                    run_id=metric_run_id_for_attempt(training_run_id, 1),
                )
                if metrics
                else {}
            ),
            "dataset": {
                "hf_repo": getattr(dataset, "hf_repo", ""),
                "name": type(dataset).__name__,
            },
            "eval_dataset": (
                {
                    "hf_repo": getattr(self.eval_dataset, "hf_repo", ""),
                    "name": type(self.eval_dataset).__name__,
                }
                if self.eval_dataset is not None
                else None
            ),
            "lr": getattr(recipe, "lr", None),
            "global_batch_size": getattr(recipe, "global_batch_size", None),
        }

        if isinstance(recipe, SlimeRecipe | MilesRecipe):
            from modal_training_gym.common.launcher_utils import (
                serialize_recipe_params,
            )

            summary["recipe"] = {
                # gpu_type is a launcher-only field (in _MILES_SKIP) so it is
                # absent from serialize_recipe_params for miles; the dashboard
                # cluster column reads recipe.gpu_type, so keep it here too.
                "gpu_type": getattr(recipe, "gpu_type", None),
                **serialize_recipe_params(
                    recipe,
                    dataset=dataset,
                    eval_dataset=self.eval_dataset,
                    model=model,
                ),
            }

        return summary

    def _build_run_metadata(self) -> dict[str, Any] | None:
        metadata: dict[str, Any] = {}
        if self.group_id:
            metadata["group_id"] = self.group_id
        if self.group_overrides is not None or self.group_axes is not None:
            overrides = dict(self.group_overrides or {})
            axes = list(self.group_axes or overrides)
            metadata["group_tags"] = {
                "group_id": self.group_id or "",
                "axes": axes,
                "overrides": overrides,
                "tags": [
                    {
                        "key": key,
                        "label": key.split(".")[-1].replace("_", " "),
                        "value": value,
                    }
                    for key, value in overrides.items()
                ],
            }
        return metadata or None

    def _build_status_display(
        self,
        training_run_id: str,
        framework_status_url: str,
        config_path: Any,
    ) -> "_TrainStatusDisplay":
        dataset_name = ""
        if self.dataset:
            dataset_name = (
                getattr(self.dataset, "hf_repo", "") or type(self.dataset).__name__
            )
        return _TrainStatusDisplay(
            run_id=training_run_id,
            framework=self.framework.value,
            model_name=getattr(self.model, "model_name", "") if self.model else "",
            dataset_name=dataset_name,
            framework_status_url=framework_status_url,
            config_path=str(config_path),
        )

    def context_plan_line(self) -> str | None:
        """Summarize effective context length and parallelism on one line."""
        recipe = self.recipe
        max_tokens_per_gpu = getattr(recipe, "max_tokens_per_gpu", None)
        if max_tokens_per_gpu is None:
            return None

        context_parallel_size = getattr(recipe, "context_parallel_size", 1) or 1
        effective_context = max_tokens_per_gpu * context_parallel_size
        return (
            f"effective_train_context={effective_context:,} tokens "
            f"(max_tokens_per_gpu={max_tokens_per_gpu:,} x cp={context_parallel_size}; "
            f"tp={getattr(recipe, 'tensor_model_parallel_size', 'n/a')}, "
            f"pp={getattr(recipe, 'pipeline_model_parallel_size', 'n/a')}, "
            f"ep={getattr(recipe, 'expert_model_parallel_size', 'n/a')})"
        )

    def train(self, *, show_output: bool = True) -> TrainingRun:
        """Run one training configuration.

        Returns:
            The completed training run.
        """
        from modal_training_gym.common.modal_lifecycle import stop_app

        launch = self.launch(show_output=show_output)
        try:
            return launch.result(stop_app_on_success=True)
        except BaseException:
            if not self.detach and launch.modal_app_id:
                stop_app(launch.modal_app_id)
            raise

    def launch(
        self,
        *,
        show_output: bool = True,
    ) -> TrainingRun:
        """Start one training run in a detached Modal app.

        Returns:
            The launched training run.
        """
        import modal

        from modal_training_gym.cli.setup import ensure_dashboard_deployed
        from modal_training_gym.common.config import (
            CONFIG_PATH,
            get_framework_status_url,
        )

        from modal_training_gym.common.modal_lifecycle import stop_app
        from modal_training_gym.common.status_reporter import enqueue_framework_status

        maybe_warn_gpu_oom(self.recipe, self.model)
        training_run_id = self._generate_training_run_id()
        ensure_dashboard_deployed()
        framework_status_url = get_framework_status_url() or ""
        framework_status_token = _secrets.token_urlsafe(32)

        status_display = self._build_status_display(
            training_run_id, framework_status_url, CONFIG_PATH
        )
        if show_output:
            status_display.print_banner()
            if context_plan_line := self.context_plan_line():
                print(f"Training context: {context_plan_line}")

        created_at = int(time.time())
        run_record = TrainingRun(
            training_run_id=training_run_id,
            modal_app_id="",
            modal_app_url="",
            framework=self.framework,
            config=self._build_config_summary(training_run_id),
            framework_status=self._initializing_status(),
            created_at=created_at,
            started_at=created_at,
            metadata=self._build_run_metadata(),
        )
        try:
            run_record.save()
            vol_put(
                MetadataStore.FRAMEWORK_STATUS_TOKENS,
                training_run_id,
                {"token": framework_status_token},
            )
        except RuntimeError:
            framework_status_token = ""
        print(f"TrainingRun recorded: {training_run_id}")

        app = None
        function_call = None
        launch_error = None
        try:
            app = self._build_app(training_run_id)
            output_context = modal.enable_output() if show_output else nullcontext()
            with output_context:
                with app.run(detach=True):
                    try:
                        modal_app_id = app.app_id or ""
                        modal_app_url = modal_app_dashboard_url(modal_app_id)
                        if show_output:
                            status_display.set_modal_app_url(modal_app_url)

                        run_record.modal_app_id = modal_app_id
                        run_record.modal_app_url = modal_app_url
                        try:
                            run_record.save()
                        except RuntimeError:
                            pass

                        def _set_status(
                            status: FrameworkStatus, *, is_active: bool = True
                        ) -> None:
                            run_record.framework_status = status
                            if show_output:
                                status_display.emit_stage(status.value)
                            enqueue_framework_status(
                                training_run_id,
                                status.value,
                                token=framework_status_token,
                                is_active=is_active,
                            )

                        megatron_to_hf_mode = getattr(
                            self.recipe, "megatron_to_hf_mode", ""
                        )
                        needs_conversion = megatron_to_hf_mode != "bridge"
                        download_status, convert_status = (
                            (SlimeStatus.DOWNLOAD_MODEL, SlimeStatus.CONVERT_MODEL)
                            if isinstance(self.recipe, SlimeRecipe)
                            else (MilesStatus.DOWNLOAD_MODEL, MilesStatus.CONVERT_MODEL)
                        )
                        _set_status(download_status, is_active=False)
                        app.download.remote(
                            training_run_id=training_run_id,
                            framework_status_url=framework_status_url,
                            framework_status_token=framework_status_token,
                        )
                        if needs_conversion:
                            _set_status(convert_status, is_active=False)
                            _convert_checkpoint_on_cache_miss(
                                app,
                                training_run_id=training_run_id,
                                framework_status_url=framework_status_url,
                                framework_status_token=framework_status_token,
                            )

                        function_call = app.train.spawn(
                            modal_app_id=modal_app_id,
                            modal_app_url=modal_app_url,
                            framework_status_url=framework_status_url,
                            framework_status_token=framework_status_token,
                        )
                    except BaseException as exc:
                        launch_error = exc

            if launch_error is not None:
                raise launch_error
        except (KeyboardInterrupt, Exception) as exc:
            if function_call is None:
                app_id = run_record.modal_app_id or (app.app_id if app else "")
                if app_id:
                    stop_app(app_id)
                if isinstance(exc, KeyboardInterrupt) or app is None:
                    _terminalize_launch(run_record, exc)
                    raise
                error = TrainingGymError(
                    f'Setup of Modal app "{app.name}" was interrupted before training '
                    "could begin. Relaunch your TrainConfig to try again."
                )
                _terminalize_launch(run_record, error)
                raise error from exc
            launch_error = exc

        run_record.function_call_id = function_call.object_id
        run_record._function_call = function_call
        run_record._status_display = status_display if show_output else None
        try:
            run_record.save()
        except RuntimeError:
            pass
        if isinstance(launch_error, KeyboardInterrupt):
            print(
                f'Disconnected from Modal app "{app.name}". The app was left running.'
            )
            print(run_record.modal_app_url)
            raise launch_error
        if launch_error is not None:
            print(
                f"WARNING: training was launched, but Modal client cleanup failed: {launch_error}"
            )
        print(
            f"Launched training {run_record.training_run_id}: "
            f"app={run_record.modal_app_id}, function_call={run_record.function_call_id}"
        )
        return run_record
