"""POC (option B): Dojo declares evals; a Modal Evals-side scheduler runs them.

Not for merge. Training writes one ``eval.json`` manifest into the run's
checkpoint folder when it starts, and registers the checkpoints Volume once.
A scheduler that Modal Evals would own (stand-in: ``evals_scheduler_poc.py``)
scans registered Volumes, finds committed checkpoints next to a manifest, and
evaluates the due ones. Training has no runtime dependency on Modal Evals, and
any trainer that writes the same manifest (Epoch, Spindle, external) is
evaluated the same way.

    TrainConfig(
        ...,
        eval_registration=EvalRegistration(
            evaluations=["terminal-bench-2-1", "aime-2026"],
            every_n_checkpoints=2,
        ),
    )
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from modal_dojo.common.train import TrainConfig

MANIFEST_NAME = "eval.json"
MANIFEST_METADATA_KEY = "eval_manifest"
#: modal.Dict of registered Volumes: ``{volume_name: {"environment": ...}}``.
REGISTRATIONS_DICT = "modal-evals-registrations"


class EvalRegistration(BaseModel):
    """What to evaluate; the scheduler decides when and where.

    Args:
        evaluations: Modal Evals keys or release IDs to run on each due checkpoint.
        every_n_checkpoints: Evaluate each checkpoint whose 1-based position
            is a multiple of this.
        baseline: Also evaluate the untrained source model once.
        params: Evaluation params (sampling, reply limit); applied to every
            evaluation so settings match across checkpoints.
    """

    model_config = ConfigDict(extra="forbid")

    evaluations: list[str] = Field(min_length=1)
    every_n_checkpoints: int = Field(default=1, ge=1)
    baseline: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class EvalManifest(EvalRegistration):
    """The ``eval.json`` contract between any trainer and Modal Evals."""

    version: int = 1
    training_run_id: str
    base_model: str
    # Dojo-only: lets the shim serve with Dojo's Endpoint. A Modal Evals
    # CheckpointTarget (option D) would only need ``base_model``.
    model_config_json: dict[str, Any] = Field(default_factory=dict)


def build_manifest(train_config: TrainConfig, training_run_id: str) -> dict[str, Any]:
    """Manifest for this run, stored on the run record at launch."""
    from dataclasses import asdict, is_dataclass

    registration = train_config.eval_registration
    assert registration is not None
    model = train_config.model
    model_json = asdict(model) if is_dataclass(model) else {}
    manifest = EvalManifest(
        **registration.model_dump(),
        training_run_id=training_run_id,
        base_model=model.model_name,
        model_config_json=json.loads(json.dumps(model_json, default=str)),
    )
    return manifest.model_dump(mode="json")


def write_manifest(metadata: dict[str, Any] | None, checkpoint_dir: str) -> None:
    """Remote, at training start: write the run's manifest next to its
    checkpoints. A resumed run keeps its first manifest. Never raises."""
    manifest = (metadata or {}).get(MANIFEST_METADATA_KEY)
    if not manifest or not checkpoint_dir:
        return
    path = os.path.join(checkpoint_dir, MANIFEST_NAME)
    try:
        if os.path.exists(path):
            return
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
    except OSError as exc:
        print(f"WARNING: could not write {path}: {exc}")


def register_volume(volume_name: str, environment: str | None = None) -> None:
    """Local, at launch: ask the scheduler to watch ``volume_name``. Idempotent."""
    import modal

    registrations = modal.Dict.from_name(REGISTRATIONS_DICT, create_if_missing=True)
    registrations[volume_name] = {"environment": environment}


def due_checkpoints(
    manifest: EvalManifest,
    checkpoint_names: list[str],
    started: set[str],
) -> list[str]:
    """Checkpoints (and ``"base"``) to evaluate now, oldest first. Pure."""
    due = ["base"] if manifest.baseline else []
    due += [
        name
        for position, name in enumerate(sorted(checkpoint_names), 1)
        if position % manifest.every_n_checkpoints == 0
    ]
    return [name for name in due if name not in started]
