"""Fall back to other recipes when a run waits too long for GPUs.

Proof of concept. ``Fallback([primary, *alternates], threshold=...)`` gives
``ManagedTrainConfig`` an ordered list of recipes, typically for different GPU
types. Each is launched in turn; if its training function has not started
within ``threshold`` seconds, its app is stopped and the next recipe is
launched. The last recipe waits as long as it takes.

"Started" means the training function began executing, which for a
multi-node recipe means Modal placed every node. The training function marks
that in a shared ``modal.Dict`` (``mark_training_started``) as its first act;
Modal has no supported way to ask whether a function call is still queued
for capacity.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.errors import DojoConfigError

if TYPE_CHECKING:
    from modal_dojo.common.train import TrainConfig, TrainingRun

TRAINING_STARTS_DICT = "modal-dojo-training-starts"
_POLL_SECONDS = 15.0


# TODO: add a reference page once the fallback API leaves proof of concept.
@exclude_from_api_reference
@dataclass(frozen=True)
class Fallback:
    """Recipes to try in order when one waits too long for GPUs.

    ```python
    Fallback([h200_recipe, h100_recipe, b200_recipe], threshold=60 * 60)
    ```

    Attributes:
        recipes: The primary recipe, then alternates in order of preference.
        threshold: Seconds to wait for a recipe's training to start before
            trying the next one. The last recipe waits indefinitely.
    """

    recipes: Sequence[Any]
    threshold: float

    def __post_init__(self) -> None:
        if not self.recipes:
            raise DojoConfigError("Fallback needs at least one recipe")
        if not self.threshold > 0:
            raise DojoConfigError(
                f"Fallback.threshold must be positive, got {self.threshold!r}"
            )


def _starts() -> Any:
    import modal

    return modal.Dict.from_name(TRAINING_STARTS_DICT, create_if_missing=True)


def mark_training_started(training_run_id: str) -> None:
    """Record that this run's training function is running. Never raises."""
    try:
        _starts()[training_run_id] = time.time()
    except Exception as exc:  # noqa: BLE001 — a missed mark only delays fallback
        print(f"WARNING: could not mark training {training_run_id} started: {exc!r}")


def training_started(training_run_id: str) -> bool:
    """Whether ``training_run_id``'s training function has started."""
    return _starts().get(training_run_id) is not None


def wait_for_training_start(
    training_run_id: str,
    timeout: float,
    *,
    started: Callable[[str], bool] = training_started,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Poll until the run's training starts; False if ``timeout`` passes first."""
    deadline = clock() + timeout
    while True:
        if started(training_run_id):
            return True
        remaining = deadline - clock()
        if remaining <= 0:
            return False
        sleep(min(_POLL_SECONDS, remaining))


def _abandon(run: TrainingRun) -> None:
    """Stop a run that never started, and record why."""
    from modal_dojo.common.launcher_helpers import mark_run_stopped
    from modal_dojo.common.modal_lifecycle import stop_app

    stop_app(run.modal_app_id)
    mark_run_stopped(run)
    run.error_message = "Fell back to another recipe: training did not start in time"
    try:
        run.save()
    except RuntimeError:
        pass


def launch_with_fallback(
    configs: Sequence[TrainConfig],
    threshold: float,
    *,
    launch: Callable[[TrainConfig], TrainingRun] = lambda config: config.launch(),
    wait: Callable[[str, float], bool] = wait_for_training_start,
    abandon: Callable[[TrainingRun], None] = _abandon,
) -> TrainingRun:
    """Launch ``configs`` in order until one's training starts in ``threshold``.

    Returns the run that is training, or the last config's run, which is
    left queued however long it waits.
    """
    for index, config in enumerate(configs):
        run = launch(config)
        if index == len(configs) - 1:
            return run
        gpu = f"{config.recipe.gpu_type} x{config.recipe.gpu_allocation.total_gpus}"
        print(
            f"Waiting up to {threshold / 60:g} min for {gpu} to start "
            f"(run {run.training_run_id})..."
        )
        if wait(run.training_run_id, threshold):
            return run
        print(f"{gpu} did not start in time; falling back to the next recipe.")
        abandon(run)
    raise AssertionError("unreachable: configs is never empty")
