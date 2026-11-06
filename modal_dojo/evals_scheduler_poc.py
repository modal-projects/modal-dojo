"""POC (option B): the scheduler Modal Evals would own, as a standalone app.

Not for merge. Deploy with ``modal deploy modal_dojo/evals_scheduler_poc.py``.
Every 2 minutes, one container scans each registered Volume for run folders
with an ``eval.json`` manifest, lists committed ``iter_*`` checkpoints, and
spawns one evaluation per due (checkpoint, evaluation). Each start is recorded
before it is spawned, so a crash never starts the same eval twice. This is the
midtrain-eval scheduler's loop, with the trainer's manifest replacing
per-project policy edits.
"""

from __future__ import annotations

import json
import time

import modal

from modal_dojo.common.eval_registration import (
    MANIFEST_NAME,
    REGISTRATIONS_DICT,
    EvalManifest,
    due_checkpoints,
)

STATE_DICT = "modal-evals-scheduler-state"

app = modal.App("modal-evals-scheduler-poc")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("modal", "pydantic>=2,<3", "cloudpickle", "httpx")
    .add_local_python_source("modal_dojo", "modal_evals")
)


def _manifests(volume: modal.Volume) -> dict[str, EvalManifest]:
    """``{run_folder: manifest}`` for every run folder with a manifest."""
    found: dict[str, EvalManifest] = {}
    for entry in volume.iterdir("/", recursive=True):
        if entry.path.endswith(f"/{MANIFEST_NAME}"):
            folder = entry.path.rsplit("/", 1)[0]
            raw = b"".join(volume.read_file(entry.path))
            try:
                found[folder] = EvalManifest.model_validate(json.loads(raw))
            except ValueError as exc:
                print(f"skipping invalid manifest {entry.path}: {exc}")
    return found


def _committed_checkpoints(volume: modal.Volume, folder: str) -> list[str]:
    names = []
    for entry in volume.iterdir(folder):
        name = entry.path.rstrip("/").rsplit("/", 1)[-1]
        if (
            name.startswith("iter_")
            and entry.type == modal.volume.FileEntryType.DIRECTORY
        ):
            names.append(name)
    return sorted(names)


@app.function(image=image, schedule=modal.Period(minutes=2), max_containers=1)
def tick() -> None:
    registrations = modal.Dict.from_name(REGISTRATIONS_DICT, create_if_missing=True)
    state = modal.Dict.from_name(STATE_DICT, create_if_missing=True)
    for volume_name, info in registrations.items():
        volume = modal.Volume.from_name(
            volume_name, environment_name=(info or {}).get("environment")
        )
        for folder, manifest in _manifests(volume).items():
            for evaluation in manifest.evaluations:
                prefix = f"{volume_name}:{folder}:{evaluation}:"
                started = {
                    key.removeprefix(prefix)
                    for key in state.keys()
                    if key.startswith(prefix)
                }
                for checkpoint in due_checkpoints(
                    manifest, _committed_checkpoints(volume, folder), started
                ):
                    key = prefix + checkpoint
                    state[key] = {"status": "starting", "at": time.time()}
                    call = evaluate.spawn(
                        volume_name, folder, checkpoint, evaluation, manifest
                    )
                    state[key] = {"status": "running", "call_id": call.object_id}


@app.function(image=image, timeout=6 * 60 * 60)
def evaluate(
    volume_name: str,
    folder: str,
    checkpoint_name: str,
    evaluation: str,
    manifest: EvalManifest,
) -> dict:
    """Serve one checkpoint (or the base model), run one evaluation, record it."""
    from modal_dojo.common.checkpoint import Checkpoint, CheckpointType
    from modal_dojo.common.endpoint import Endpoint
    from modal_dojo.common.evaluations import _record_evaluation, run_evaluation
    from modal_dojo.common.models import ModelConfig

    model = ModelConfig(**manifest.model_config_json)
    checkpoint = None
    if checkpoint_name != "base":
        checkpoint = Checkpoint(
            checkpoint_type=CheckpointType.megatron,
            name=checkpoint_name,
            path=f"/checkpoints/{folder}/{checkpoint_name}",
            timestamp=0.0,
            training_run_id=manifest.training_run_id,
            checkpoints_volume_name=volume_name,
            checkpoints_mount_path="/checkpoints",
        )
    endpoint = Endpoint.launch(model, checkpoint)
    try:
        endpoint.wait_until_ready()
        job = run_evaluation(
            evaluation,
            endpoint,
            params=manifest.params,
            tags={
                "training_run_id": manifest.training_run_id,
                "checkpoint": checkpoint_name,
            },
            wait=True,
        )
        results = job.results()
        record = {
            "iteration": checkpoint_name,
            "evaluation": evaluation,
            "job_id": job.job_id,
            "status": results.job.status,
            "url": results.output_volume.url,
        }
    finally:
        endpoint.stop()
    state = modal.Dict.from_name(STATE_DICT, create_if_missing=True)
    state[f"{volume_name}:{folder}:{evaluation}:{checkpoint_name}"] = record
    # Dojo-only courtesy: also surface it on TrainingRun.evaluations().
    try:
        _record_evaluation(manifest.training_run_id, record)
    except Exception as exc:
        print(f"could not record on TrainingRun: {exc}")
    return record
