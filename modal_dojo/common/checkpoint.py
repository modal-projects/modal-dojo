# ── Checkpoint ───────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import posixpath
import time
from dataclasses import dataclass
from enum import Enum

import modal
from modal import Volume
from modal.exception import NotFoundError

from modal_dojo.common.errors import TrainingGymConfigError
from modal_dojo.common.models import ModelConfig
from modal_dojo.common.run import TrainingRun
from modal_dojo.common.torch_dist_checkpoint import (
    TORCH_DIST_TRACKER_NAME,
    is_complete_torch_dist_checkpoint,
    parse_torch_dist_iteration,
    parse_torch_dist_tracker,
)
from modal_dojo.deploy_recipes import SglangRecipe, VllmRecipe

_CHECKPOINTS_MOUNT_FALLBACK = "/checkpoints"
_CONVERT_COMPLETE_MARKER = ".training_gym_convert_complete"


class CheckpointType(Enum):
    """Whether a checkpoint is Hugging Face or Megatron weights."""

    hf = "hf"
    megatron = "megatron"


@dataclass
class Checkpoint:
    """A complete training checkpoint discovered on a Modal Volume.

    Attributes:
        checkpoint_type: Hugging Face or Megatron layout.
        name: Volume directory basename, such as ``iter_0000050``.
        path: Absolute path on the checkpoints volume mount.
        timestamp: POSIX mtime of the checkpoint directory.
        training_run_id: Training run that wrote this checkpoint.
        app_name: Modal app name for the run.
        checkpoints_volume_name: Modal Volume that stores the checkpoint.
        checkpoints_mount_path: Container mount path for that volume.
    """

    checkpoint_type: CheckpointType
    name: str
    path: str
    timestamp: float
    training_run_id: str = ""
    app_name: str = ""
    checkpoints_volume_name: str = ""
    checkpoints_mount_path: str = ""

    @property
    def path_relative_to_volume(self) -> str:
        """Path relative to the checkpoints volume mount."""
        return volume_relative_path(
            self.path, self.checkpoints_mount_path or _CHECKPOINTS_MOUNT_FALLBACK
        )


def require_within_volume_mount(path: str, mount_path: str) -> tuple[str, str]:
    normalized_path = posixpath.normpath(path)
    normalized_mount = posixpath.normpath(mount_path)
    if not posixpath.isabs(normalized_path) or not posixpath.isabs(normalized_mount):
        raise TrainingGymConfigError(
            f"Path {path!r} and Volume mount {mount_path!r} must be absolute POSIX paths."
        )
    if posixpath.commonpath([normalized_path, normalized_mount]) != normalized_mount:
        raise TrainingGymConfigError(
            f"Path {path!r} is outside Volume mount {mount_path!r}."
        )
    return normalized_path, normalized_mount


def volume_relative_path(path: str, mount_path: str) -> str:
    normalized_path, normalized_mount = require_within_volume_mount(path, mount_path)
    relative_path = posixpath.relpath(normalized_path, normalized_mount)
    return "" if relative_path == "." else relative_path


def _list_checkpoints(
    checkpoint_dir: str,
    checkpoints_volume_name: str,
    checkpoints_mount_path: str,
    *,
    training_run_id: str = "",
    app_name: str = "",
) -> list[Checkpoint]:
    def _entry_name(entry: object) -> str:
        return getattr(entry, "path", "").rstrip("/").rsplit("/", 1)[-1]

    def _is_dir_entry(entry: object) -> bool:
        is_dir_fn = getattr(entry, "is_dir", None)
        if callable(is_dir_fn):
            return bool(is_dir_fn())
        entry_type = getattr(entry, "type", None)
        if entry_type is None:
            return False
        entry_type_name = getattr(entry_type, "name", "")
        if isinstance(entry_type_name, str):
            return entry_type_name.upper() == "DIRECTORY"
        return False

    checkpoint_dir = checkpoint_dir.rstrip("/")
    if checkpoint_dir == "" or not checkpoints_volume_name:
        return []
    checkpoints_mount_path = checkpoints_mount_path or _CHECKPOINTS_MOUNT_FALLBACK

    rel = volume_relative_path(checkpoint_dir, checkpoints_mount_path)
    volume = Volume.from_name(checkpoints_volume_name, create_if_missing=False)

    try:
        entries = list(volume.iterdir(rel or "/", recursive=False))
    except (FileNotFoundError, NotFoundError):
        return []

    tracker_rel = f"{rel}/{TORCH_DIST_TRACKER_NAME}" if rel else TORCH_DIST_TRACKER_NAME
    try:
        raw = b"".join(volume.read_file(tracker_rel))
        tracker_iteration = parse_torch_dist_tracker(raw.decode())
    except (FileNotFoundError, NotFoundError):
        tracker_iteration = None
    checkpoints: list[Checkpoint] = []
    for entry in sorted(
        (entry for entry in entries if _is_dir_entry(entry)),
        key=_entry_name,
    ):
        name = _entry_name(entry)
        if not name.startswith("iter_") or name.endswith("_hf"):
            continue
        child_rel = f"{rel}/{name}" if rel else name
        try:
            child_names = {
                _entry_name(child)
                for child in volume.iterdir(child_rel, recursive=False)
            }
        except (FileNotFoundError, NotFoundError):
            child_names = set()
        iteration = parse_torch_dist_iteration(name)
        if (
            tracker_iteration is None
            or iteration is None
            or iteration > tracker_iteration
            or not is_complete_torch_dist_checkpoint(child_names)
        ):
            continue
        checkpoints.append(
            Checkpoint(
                checkpoint_type=CheckpointType.megatron,
                name=name,
                path=posixpath.join(checkpoint_dir, name),
                timestamp=float(getattr(entry, "mtime", 0.0)),
                training_run_id=training_run_id,
                app_name=app_name,
                checkpoints_volume_name=checkpoints_volume_name,
                checkpoints_mount_path=checkpoints_mount_path,
            )
        )
    return checkpoints


def _conversion_gpu_spec(
    checkpoint: Checkpoint, recipe: VllmRecipe | SglangRecipe
) -> str:
    run_id = getattr(checkpoint, "training_run_id", "")
    if run_id:
        try:
            training_run = TrainingRun.from_id(run_id)
        except KeyError:
            training_run = None
        if training_run:
            recipe_config = training_run.config.get("recipe", {})
            gpu_type = recipe_config.get("gpu_type")
            n_gpu = recipe_config.get("actor_num_gpus_per_node")
            if gpu_type and n_gpu:
                n_gpu = int(n_gpu)
                return f"{gpu_type}:{n_gpu}" if n_gpu > 1 else str(gpu_type)

    if isinstance(recipe, SglangRecipe):
        gpu = recipe.gpu
        n_gpu = recipe.tp or 1
    else:
        gpu = recipe.gpu
        n_gpu = recipe.n_gpu or 1
    return f"{gpu}:{n_gpu}" if n_gpu > 1 else str(gpu)


def convert_megatron_checkpoint_to_hf(
    checkpoint: Checkpoint,
    model: ModelConfig,
    recipe: VllmRecipe | SglangRecipe = SglangRecipe(),
) -> Checkpoint:
    """Convert a Megatron checkpoint to Hugging Face format."""
    if checkpoint.checkpoint_type == CheckpointType.hf:
        return checkpoint

    checkpoints_volume_name = checkpoint.checkpoints_volume_name
    if checkpoints_volume_name in (None, ""):
        raise TrainingGymConfigError(
            "Cannot convert checkpoint without checkpoints volume metadata."
        )
    checkpoints_mount_path = (
        checkpoint.checkpoints_mount_path or _CHECKPOINTS_MOUNT_FALLBACK
    )
    output_path = f"{checkpoint.path}_hf"
    volume = Volume.from_name(checkpoints_volume_name, create_if_missing=False)
    rel = volume_relative_path(output_path, checkpoints_mount_path)
    marker_rel = (
        f"{rel}/{_CONVERT_COMPLETE_MARKER}" if rel else _CONVERT_COMPLETE_MARKER
    )
    try:
        b"".join(volume.read_file(marker_rel))
    except (FileNotFoundError, NotFoundError):
        pass
    else:
        return Checkpoint(
            checkpoint_type=CheckpointType.hf,
            name=os.path.basename(output_path.rstrip("/")),
            path=output_path,
            timestamp=checkpoint.timestamp,
            training_run_id=checkpoint.training_run_id,
            app_name=checkpoint.app_name,
            checkpoints_volume_name=checkpoints_volume_name,
            checkpoints_mount_path=checkpoints_mount_path,
        )

    model_ref = model.model_name or model.model_path
    if model_ref in (None, ""):
        raise TrainingGymConfigError(
            "Cannot convert a megatron checkpoint without model_name or model_path."
        )

    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    checkpoints_volume = volume
    from modal_dojo.common import hf_secrets
    from modal_dojo.frameworks.slime.launcher import _build_slime_base_image

    image = _build_slime_base_image().add_local_python_source("modal_dojo", copy=True)
    conversion_app = modal.App("training-gym-checkpoint-convert")
    gpu_spec = _conversion_gpu_spec(checkpoint, recipe)

    @conversion_app.function(
        image=image,
        gpu=gpu_spec,
        volumes={
            "/root/.cache/huggingface": hf_cache_volume,
            checkpoints_mount_path: checkpoints_volume,
        },
        timeout=4 * 60 * 60,
        secrets=hf_secrets(),
        serialized=True,
        name="convert_megatron_to_hf",
    )
    def convert_megatron_to_hf(
        input_dir: str,
        output_dir: str,
        model_ref: str,
    ) -> str:
        import importlib.util
        import shlex
        import subprocess

        from huggingface_hub import snapshot_download

        hf_cache_volume.reload()
        checkpoints_volume.reload()

        if os.path.isabs(model_ref) and os.path.isdir(model_ref):
            hf_path = model_ref
        else:
            hf_path = snapshot_download(model_ref, local_files_only=True)

        spec = importlib.util.find_spec(
            "modal_dojo.frameworks.slime.modal_helpers.convert_torch_dist_to_hf"
        )
        convert_script = spec.origin if spec else None
        if convert_script in (None, ""):
            raise RuntimeError(
                "modal_dojo.frameworks.slime.modal_helpers.convert_torch_dist_to_hf is missing"
            )
        cmd = (
            f"python {convert_script} "
            f"--input-dir {shlex.quote(input_dir)} "
            f"--output-dir {shlex.quote(output_dir)} "
            f"--origin-hf-dir {shlex.quote(hf_path)} "
            f"--force"
        )
        print(f"Converting checkpoint for serving: {cmd}")
        subprocess.run(["bash", "-c", cmd], check=True)
        with open(os.path.join(output_dir, _CONVERT_COMPLETE_MARKER), "w"):
            pass
        checkpoints_volume.commit()
        return output_dir

    with modal.enable_output():
        with conversion_app.run():
            output_path = convert_megatron_to_hf.remote(
                input_dir=checkpoint.path,
                output_dir=output_path,
                model_ref=model_ref,
            )

    return Checkpoint(
        checkpoint_type=CheckpointType.hf,
        name=os.path.basename(output_path),
        path=output_path,
        timestamp=time.time(),
        training_run_id=checkpoint.training_run_id,
        app_name=checkpoint.app_name,
        checkpoints_volume_name=checkpoints_volume_name,
        checkpoints_mount_path=checkpoints_mount_path,
    )
