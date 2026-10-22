"""Flyte launch entrypoints: ``flyte_launch`` / ``flyte_train``.

These replace the Modal ``TrainConfig.launch()/train()`` backend: they build the
Flyte task pipeline for a ``TrainConfig`` and submit it to the Flyte cluster
through ``flyte.run``, which returns a ``Run`` handle we wrap in
``FlyteTrainingRun``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from modal_dojo.common.train import TrainConfig
    from modal_dojo.flyte.run import FlyteTrainingRun


def flyte_config_path() -> str:
    """Resolve the Flyte config file.

    ``MODAL_DOJO_FLYTE_CONFIG`` overrides the default ``~/.flyte/config.yaml``
    (Union) — point it at ``~/.flyte/config-devbox.yaml`` for the devbox.
    """
    env = os.environ.get("MODAL_DOJO_FLYTE_CONFIG")
    if env:
        return env
    default = Path.home() / ".flyte" / "config.yaml"
    return str(default)


def _init_flyte() -> None:
    """Initialize the Flyte SDK against the cluster config (idempotent)."""
    import flyte

    try:
        flyte.init_from_config(path_or_config=flyte_config_path())
    except Exception as exc:  # already-initialized or connection convenience
        print(f"[modal-dojo/flyte] init_from_config skipped: {exc}")


def _resolve_mode(config: "TrainConfig", mode: str | None) -> str:
    from modal_dojo.flyte import TrainingMode

    if mode is not None:
        return TrainingMode(mode).value
    from modal_dojo.train_recipes.miles_recipe import MilesRecipe
    from modal_dojo.train_recipes.slime_recipe import SlimeRecipe

    recipe = config.recipe
    if isinstance(recipe, MilesRecipe):
        return TrainingMode.MILES.value
    if isinstance(recipe, SlimeRecipe):
        return TrainingMode.SLIME.value
    return TrainingMode.MINIMAL.value


_JSON_VALUE_TYPES = (str, int, float, bool, type(None))


def _recipe_as_plain(recipe: Any) -> dict:
    """Serialize a recipe to a JSON-safe dict for Flyte transport.

    Recipes (``SlimeRecipe`` / ``MilesRecipe``) import ``modal`` at module scope,
    so pickling the object onto a Modal-free Flyte image fails with
    ``No module named 'modal'``. We instead ship the recipe as a plain dict of
    its JSON-serializable field values; non-serializable launcher fields
    (callables, custom images) are dropped — they are client-side deployment
    concerns and not needed inside Flyte.

    Phase 2 (framework training) reconstructs a lightweight recipe from this
    dict on the remote side; ``minimal`` mode does not touch it at all.
    """
    try:
        import dataclasses

        values = dataclasses.asdict(recipe)
    except Exception:  # pragma: no cover - defensive
        values = {}
    plain: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, _JSON_VALUE_TYPES):
            plain[key] = value
        elif isinstance(value, (list, tuple)):
            plain[key] = [v for v in value if isinstance(v, _JSON_VALUE_TYPES)]
        elif isinstance(value, dict):
            plain[key] = {
                k: v
                for k, v in value.items()
                if isinstance(v, _JSON_VALUE_TYPES) or isinstance(v, dict)
            }
        # skip callables, Image/Secret objects, metrics configs, etc.
    return plain


def flyte_launch(
    config: "TrainConfig", *, mode: str | None = None
) -> "FlyteTrainingRun":
    """Start one training run on the configured Flyte cluster (non-blocking).

    Args:
        config: the ``TrainConfig`` to run.
        mode: ``TrainingMode`` value (``minimal``/``slime``/``miles``).
            Defaults to the recipe's framework, else ``minimal``.
    """
    from modal_dojo.flyte.pipeline import build_pipeline
    from modal_dojo.flyte.run import FlyteTrainingRun

    run_id = config._generate_training_run_id()
    mode = _resolve_mode(config, mode)

    _init_flyte()
    import flyte

    main = build_pipeline(config, mode=mode)
    run = flyte.run(
        main,
        run_id=run_id,
        model=config.model,
        dataset=config.dataset,
        recipe=_recipe_as_plain(config.recipe),
        mode=mode,
    )
    handle = FlyteTrainingRun(run_id, run)
    print(f"Launched Flyte training {run_id} (mode={mode}): {handle.url}")
    return handle


def flyte_train(
    config: "TrainConfig", *, mode: str | None = None
) -> "FlyteTrainingRun":
    """Run one training configuration to completion (blocking).

    Same as :func:`flyte_launch` but waits for the run to finish.
    """
    handle = flyte_launch(config, mode=mode)
    handle.wait()
    return handle
