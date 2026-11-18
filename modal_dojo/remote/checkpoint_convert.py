"""Checkpoint converters run by ``modal_dojo.common.checkpoint``."""

from __future__ import annotations

import os

CONVERT_COMPLETE_MARKER = ".training_gym_convert_complete"
NATIVE_LORA_CONVERTER_PATH = "/root/convert_native_lora_to_peft.py"


def _resolve_hf_path(model_ref: str, revision: str | None) -> str:
    from huggingface_hub import snapshot_download

    if os.path.isabs(model_ref) and os.path.isdir(model_ref):
        return model_ref
    return snapshot_download(model_ref, revision=revision, local_files_only=True)


def convert_megatron_to_hf(
    input_dir: str,
    output_dir: str,
    model_ref: str,
    revision: str | None,
    hf_cache_volume_name: str,
    checkpoints_volume_name: str,
) -> str:
    """Convert a Megatron torch-dist checkpoint to a Hugging Face directory."""
    import importlib.util
    import shlex
    import subprocess

    import modal

    checkpoints_volume = modal.Volume.from_name(checkpoints_volume_name)
    modal.Volume.from_name(hf_cache_volume_name).reload()
    checkpoints_volume.reload()

    hf_path = _resolve_hf_path(model_ref, revision)

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
    with open(os.path.join(output_dir, CONVERT_COMPLETE_MARKER), "w"):
        pass
    checkpoints_volume.commit()
    return output_dir


def convert_native_lora_to_peft(
    input_dir: str,
    output_dir: str,
    model_ref: str,
    revision: str | None,
    checkpoints_volume_name: str,
) -> str:
    """Convert a Miles native-LoRA adapter directory to a PEFT adapter."""
    import subprocess

    import modal

    hf_path = _resolve_hf_path(model_ref, revision)
    subprocess.run(
        [
            "python",
            NATIVE_LORA_CONVERTER_PATH,
            "--input-dir",
            input_dir,
            "--output-dir",
            output_dir,
            "--origin-hf-dir",
            hf_path,
        ],
        check=True,
    )
    modal.Volume.from_name(checkpoints_volume_name).commit()
    return output_dir
