"""Factory that builds a Modal app for a slime training run from config objects.

Usage (from a tutorial file):

    from modal_dojo.common.train import TrainConfig
    from modal_dojo.train_recipes.slime_recipe import SlimeRecipe

    config = TrainConfig(
        model=my_model,
        dataset=my_dataset,
        recipe=SlimeRecipe(...),
    )
    app = config.build_app()

Then: `uv run modal run <tutorial_file>.py::train`.
"""

import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from modal import App, Image, Secret

from modal_dojo.common import hf_secrets, proxy_auth_secrets


from modal_dojo.common.dataset import DatasetConfig, HarborDataset, _SftDataset
from modal_dojo.common.framework import (
    mount_tools_dir,
)
from modal_dojo.common.models import ModelConfig
from modal_dojo.common.ray_cluster import (
    _supports_rdma,
    clustered_if,
)
from modal_dojo.common.run import (
    has_torch_dist_checkpoint,
    record_resume_checkpoint,
    torch_dist_resume_checkpoint,
)
from modal_dojo.common import launcher_helpers as shared
from modal_dojo.common.launcher_helpers import (
    compute_recipe_save_root,
    build_app_tags,
    apply_scoped_save,
    init_training_run_record,
    mount_caller_source,
    resolve_caller_context,
    register_recipe_functions,
    report_phase,
    start_training_cluster,
    write_datasets,
)
from modal_dojo.common.launcher_utils import (
    timing_debug_env,
)
from modal_dojo.common.metrics import (
    apply_metric_image,
    metric_runtime_env,
    metric_secrets,
    preflight_metric,
)
from modal_dojo.common.trackio import resolve_trackio_destination
from modal_dojo.common.wandb import WandbConfig
from modal_dojo.common.status_reporter import flush as flush_status_reporter
from modal_dojo.common.status import SlimeStatus
from modal_dojo.common.torch_dist_checkpoint import (
    is_complete_torch_dist_checkpoint_dir,
)

from modal_dojo.train_recipes.slime_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    SlimeRecipe,
)
from .modal_helpers.utils import (
    build_train_cmd,
    get_checkpoint_conversion_policy,
    get_modal_cluster_context,
    prepare_slime_config,
    resolve_checkpoint_ref,
)
from modal_dojo.common.patches import _MEGATRON_PATCHES, encode_patch
from modal_dojo.common.checkpoint import Checkpoint
from modal_dojo.common.framework import Framework


SLIME_ROOT = "/root/slime"
# Pin by digest to prevent mutable-tag drift.  Tag: nightly-dev-20260722a
SLIME_IMAGE = "slimerl/slime@sha256:a97ec147e37bef050337a9b229036eda00b4aa9c4d02b31a0109dc850f8ca342"
# v0.8.0+ makes per-task CPU/memory requests configurable via enforcement
# policies ("limit"/"ignore"), letting sandboxes burst on Modal and bill by
# actual CPU-/RAM-second usage instead of over-provisioning a static reservation.
HARBOR_PKG_VERSION = "0.8.0"

_SLIME_PATCHES = Path(__file__).parent / "modal_helpers" / "patches"
_PATCH_VALIDATION_B64 = encode_patch("patch_validation", _MEGATRON_PATCHES)
_PATCH_MEGATRON_BRIDGE_B64 = encode_patch("patch_megatron_bridge", _SLIME_PATCHES)
_PATCH_TORCH_LOAD_B64 = encode_patch("patch_torch_load", _MEGATRON_PATCHES)
_PATCH_GLOBAL_PLAN_B64 = encode_patch("patch_global_plan", _SLIME_PATCHES)
_PATCH_CHECKPOINT_SAVE_B64 = encode_patch("patch_checkpoint_save", _MEGATRON_PATCHES)
_PATCH_CHECKPOINT_COMMIT_B64 = encode_patch(
    "patch_checkpoint_commit", _MEGATRON_PATCHES
)
_PATCH_ADVANTAGES_B64 = encode_patch("patch_advantages", _SLIME_PATCHES)
_PATCH_BRIDGE_NONE_TASK_B64 = encode_patch("patch_bridge_none_task", _SLIME_PATCHES)
_PATCH_GDN_PACKED_SEQ_B64 = encode_patch("patch_gdn_packed_seq", _MEGATRON_PATCHES)
_PATCH_BRIDGE_PER_TOKEN_LOSS_B64 = encode_patch(
    "patch_bridge_provider_per_token_loss", _SLIME_PATCHES
)
_PATCH_STOP_TOKEN_DIAG_B64 = encode_patch("patch_stop_token_diagnostic", _SLIME_PATCHES)
# The Qwen3-ASR Megatron->HF converter (registers the qwen3_asr mapping incl. the
# audio tower). It lives in the base image — not the ASR recipe — because torch_dist
# -> HF conversion runs in the shared convert_megatron_checkpoint_to_hf path (deploy/eval),
# which has no recipe; baking it here makes both train-time export and deploy-time
# conversion ASR-capable. Additive + idempotent, so non-ASR runs are untouched.
_PATCH_QWEN3_ASR_EXPORT_B64 = encode_patch(
    "patch_qwen3_asr_export",
    _SLIME_PATCHES / "model_specific_patches" / "qwen3_asr",
)
# The Qwen3-VL Megatron->HF converters: a qwen3_vl per-param mapping (language
# stack + frozen-ViT identity passthrough) and a torch_dist->HF shim that skips
# the frozen ViT's stacked layers.
_PATCH_QWEN3_VL_EXPORT_B64 = encode_patch(
    "patch_qwen3_vl_export",
    _SLIME_PATCHES / "model_specific_patches" / "qwen3_vl",
)
_PATCH_QWEN3_VL_TORCH_DIST_B64 = encode_patch(
    "patch_qwen3_vl_torch_dist",
    _SLIME_PATCHES / "model_specific_patches" / "qwen3_vl",
)
_PATCH_ROLLOUT_STATUS_B64 = encode_patch(
    "patch_rollout_status_reporting", _SLIME_PATCHES
)
_PATCH_SUBSTEP_TIMING_B64 = encode_patch("patch_substep_timing", _SLIME_PATCHES)
_PATCH_ADVANTAGE_DIST_B64 = encode_patch("patch_advantage_distribution", _SLIME_PATCHES)
_PATCH_LOG_ELIDE_B64 = encode_patch("patch_log_elide", _SLIME_PATCHES)
# Backport of NVIDIA/Megatron-LM #3845: dequantize quantized CUDA tensors in the
# async dist-checkpoint writer before serialization. slime pins a pre-#3845
# Megatron, so FP8/TE _extra_state tensors otherwise crash the torch_dist save
# with inline_container.cc "unexpected pos" (e.g. the GLM-5.2 convert). No-op for
# non-quantized tensors, so safe for every image.
_PATCH_DIST_CKPT_QUANTIZED_B64 = encode_patch(
    "patch_dist_ckpt_quantized", _MEGATRON_PATCHES
)
_PATCH_DIST_CKPT_NOFORK_B64 = encode_patch("patch_dist_ckpt_nofork", _MEGATRON_PATCHES)
# OPD / multi-turn: zero-std metrics must skip non-numeric rewards (dict/None).
_PATCH_ZERO_STD_METRICS_B64 = encode_patch("patch_zero_std_metrics", _SLIME_PATCHES)
_PATCH_SGLANG_PARALLEL_ALIASES_B64 = encode_patch(
    "patch_sglang_parallel_aliases", _SLIME_PATCHES
)
_PATCH_QWEN3_5_HF_DISPATCH_B64 = encode_patch(
    "patch_qwen3_5_hf_dispatch", _SLIME_PATCHES
)
_PATCH_SGLANG_TRTLLM_MOE_REPACK_B64 = encode_patch(
    "patch_sglang_trtllm_moe_repack", _SLIME_PATCHES
)

# Patches targeting /root/slime* — a git overlay replaces that directory, so
# these are skipped in the base image when an overlay is configured and applied
# after the replacement instead.
_SLIME_ROOT_PATCHES_B64 = (
    _PATCH_MEGATRON_BRIDGE_B64,
    _PATCH_ADVANTAGES_B64,
    _PATCH_STOP_TOKEN_DIAG_B64,
    _PATCH_QWEN3_ASR_EXPORT_B64,
    _PATCH_QWEN3_VL_EXPORT_B64,
    _PATCH_QWEN3_VL_TORCH_DIST_B64,
    _PATCH_ROLLOUT_STATUS_B64,
    _PATCH_ADVANTAGE_DIST_B64,
    _PATCH_ZERO_STD_METRICS_B64,
    _PATCH_SGLANG_PARALLEL_ALIASES_B64,
    _PATCH_QWEN3_5_HF_DISPATCH_B64,
    _PATCH_SUBSTEP_TIMING_B64,
)

# Patches targeting Megatron-LM or site-packages — survive a git overlay.
_SLIME_EXTERNAL_PATCHES_B64 = (
    _PATCH_BRIDGE_NONE_TASK_B64,
    _PATCH_LOG_ELIDE_B64,
    _PATCH_DIST_CKPT_QUANTIZED_B64,
    _PATCH_DIST_CKPT_NOFORK_B64,
    _PATCH_SGLANG_TRTLLM_MOE_REPACK_B64,
)


def _patch_commands(patches: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"echo {patch} | base64 -d | python3" for patch in patches)


def _build_slime_base_image(*, apply_root_patches: bool = True) -> "Image":
    patches = _SLIME_EXTERNAL_PATCHES_B64
    if apply_root_patches:
        patches = patches + _SLIME_ROOT_PATCHES_B64
    return (
        Image.from_registry(SLIME_IMAGE)
        .entrypoint([])
        .run_commands("rm -rf /root/.cache/huggingface", *_patch_commands(patches))
    )


def _slime_git_overlay_command(repository: str, revision: str) -> str:
    """Build the reproducible image command for a fork source overlay."""
    repo = shlex.quote(repository)
    sha = shlex.quote(revision)
    checkout = "/tmp/training-gym-slime"
    return (
        "set -eux; "
        "command -v git >/dev/null; "
        f"rm -rf {checkout}; "
        f"git init {checkout}; "
        f"git -C {checkout} remote add origin {repo}; "
        f"git -C {checkout} fetch --depth=1 origin {sha}; "
        f"git -C {checkout} checkout --detach FETCH_HEAD; "
        f'test "$(git -C {checkout} rev-parse HEAD)" = {sha}; '
        f"rm -rf {checkout}/.git {SLIME_ROOT}; "
        f"mv {checkout} {SLIME_ROOT}"
    )


def _overlay_slime_source(image: "Image", slime: SlimeRecipe) -> "Image":
    if slime.local_slime:
        # Preserve the local dev overlay's existing semantics: use the checkout
        # exactly as supplied rather than requiring it to match patch anchors.
        return image.add_local_dir(
            slime.local_slime,
            remote_path=SLIME_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )
    if not (slime.slime_git_repository and slime.slime_git_revision):
        return image

    image = image.run_commands(
        _slime_git_overlay_command(slime.slime_git_repository, slime.slime_git_revision)
    )

    # The pinned source replaced /root/slime after the base-image patches ran.
    # Fail if required patches no longer apply rather than run an incompatible
    # fork with silently missing Modal Dojo behavior.
    return image.run_commands(*_patch_commands(_SLIME_ROOT_PATCHES_B64))


def _build_conversion_config(slime_cfg: Any, model: Any = None) -> dict[str, Any]:
    """Build a dict of parameters that affect the torch_dist checkpoint layout.

    Written to ``.conversion_config.json`` inside the checkpoint directory after
    conversion so that later runs can detect stale checkpoints whose parallelism
    no longer matches the current recipe.
    """
    from modal_dojo.frameworks.slime.modal_helpers.utils import (
        get_checkpoint_conversion_policy,
    )

    num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(
        slime_cfg, model=model
    )
    return {
        "num_nodes": num_nodes,
        "nproc_per_node": nproc_per_node,
        "extra_args": extra_args,
        "model_name": model.model_name if model else None,
        "slime_model_script": getattr(slime_cfg, "slime_model_script", ""),
    }


_CONVERSION_CONFIG_FILE = ".conversion_config.json"


_PIPELINE_SPLIT_FLAGS = (
    "--decoder-first-pipeline-num-layers",
    "--decoder-last-pipeline-num-layers",
)


def _conversion_config_matches(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Whether a recorded conversion still describes the current layout.

    The record stores the emitted ``extra_args``, so a checkpoint converted before
    the pipeline-split flags stopped being emitted at conversion PP1 would otherwise
    read as stale and be re-converted for nothing. Dropping those flags is tolerated;
    changing their values is not, since at PP>1 they define the split.
    """
    if stored == current:
        return True
    stored_rest, current_rest = dict(stored), dict(current)
    stored_args = stored_rest.pop("extra_args", None)
    current_args = current_rest.pop("extra_args", None)
    if stored_rest != current_rest:
        return False
    if not isinstance(stored_args, list) or not isinstance(current_args, list):
        return False
    if [a for a in stored_args if not a.startswith(_PIPELINE_SPLIT_FLAGS)] != [
        a for a in current_args if not a.startswith(_PIPELINE_SPLIT_FLAGS)
    ]:
        return False
    return {a for a in current_args if a.startswith(_PIPELINE_SPLIT_FLAGS)} <= {
        a for a in stored_args if a.startswith(_PIPELINE_SPLIT_FLAGS)
    }


def _checkpoint_conversion_cache_status(
    save_path: str, current_config: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Return the state of a cached HF-to-torch-dist conversion."""
    if not os.path.exists(save_path):
        return "missing", None
    if not has_torch_dist_checkpoint(
        save_path, is_complete=is_complete_torch_dist_checkpoint_dir
    ):
        return "incomplete", None

    import json

    config_path = os.path.join(save_path, _CONVERSION_CONFIG_FILE)
    if not os.path.isfile(config_path):
        return "stale", None
    try:
        with open(config_path) as f:
            stored_config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "stale", None
    if not _conversion_config_matches(stored_config, current_config):
        return "stale", stored_config
    return "hit", stored_config


def _preflight_wandb(wandb_cfg: WandbConfig) -> str:
    """Backward-compatible wrapper for the W&B preflight helper."""
    from modal_dojo.common.wandb import preflight_wandb

    return preflight_wandb(wandb_cfg)


def build_slime_app(
    *,
    training_run_id: str,
    slime: SlimeRecipe,
    model: ModelConfig,
    dataset: DatasetConfig,
    eval_dataset: DatasetConfig | None = None,
    checkpoint: Checkpoint | None = None,
    name: str | None = None,
    group_id: str | None = None,
) -> App:
    """Return a Modal App with `download`, `prepare_dataset`, `convert_checkpoint`, and `train` defined."""
    app_name = name or f"slime-{type(slime).__name__.lstrip('_').lower()}"
    volume_prefix = f"slime-{type(slime).__name__.lstrip('_').lower()}"

    SlimeRecipe._validate_custom_model_architecture(model)
    SlimeRecipe._validate_datasets(dataset, eval_dataset, loss_type=slime.loss_type)
    if slime.loss_type == "sft_loss":
        dataset = _SftDataset(dataset)
    dataset_path = SlimeRecipe._resolve_data_paths(dataset)
    eval_dataset_path = (
        SlimeRecipe._resolve_data_paths(eval_dataset)
        if eval_dataset is not None
        else None
    )

    # Models that can't do THD packing (model.requires_bshd, e.g. Qwen3-ASR) must
    # train on padded (bshd) batches; fail fast with the fix if the recipe didn't.
    if model and getattr(model, "requires_bshd", False):
        cfg = slime.extra_config or {}
        if cfg.get("qkv_format") != "bshd" or slime.use_dynamic_batch_size:
            raise ValueError(
                f"{model.model_name} requires padded (bshd) batches: its "
                "megatron-bridge forward doesn't implement THD sequence packing. "
                'Set extra_config={"qkv_format": "bshd", "micro_batch_size": N} and '
                "use_dynamic_batch_size=False — or use Qwen3_ASR_1_7B_Recipe, which sets "
                f"these. Got qkv_format={cfg.get('qkv_format')!r}, "
                f"use_dynamic_batch_size={slime.use_dynamic_batch_size}."
            )

    if (
        model
        and getattr(slime, "megatron_to_hf_mode", "") != "bridge"
        and not slime.ref_load
    ):
        # Non-bridge: pre-convert HF -> torch_dist (convert_checkpoint) and load that as the
        # reference checkpoint. In bridge mode we instead load the HF weights directly via
        # AutoBridge; ref_load is set to the local HF snapshot dir at train time.
        slug = model.model_name.replace("/", "--")
        object.__setattr__(slime, "ref_load", f"/checkpoints/torch_dist/{slug}-v31")

    # ── GDN compatibility ─────────────────────────────────────────────────
    # Models with Gated Delta Net (GDN) layers (use_gated_attention=True)
    # don't support packed sequences in the older Megatron-LM bundled with
    # slime.  Slime's get_batch() always creates PackedSeqParams for THD
    # format, which GDN rejects with NotImplementedError.  A build-time
    # patch (patch_gdn_packed_seq.py) neutralises the raise so GDN falls
    # back to unpacked processing.
    _has_gdn = (
        model
        and getattr(model, "architecture", None)
        and getattr(model.architecture, "use_gated_attention", False)
        and not slime.slime_model_script
    )

    _caller_module, caller_script = resolve_caller_context()

    # ── Image ────────────────────────────────────────────────────────────────
    # When a git overlay will replace /root/slime, skip root patches here so they
    # run once after the replacement instead of being applied and then discarded.
    _needs_git_overlay = bool(slime.slime_git_repository and slime.slime_git_revision)
    image = _build_slime_base_image(apply_root_patches=not _needs_git_overlay)

    # Hybrid models have layers with different parameter sets (e.g. GDN
    # layers carry linear_attn.dt_bias that standard attention layers lack).
    # Megatron's validate_sharding_integrity rejects this because not every
    # position in the global tensor is covered.  Patch both the conversion
    # and training images so saving and loading both succeed.
    _has_hybrid_spec = (
        model
        and getattr(model, "architecture", None)
        and getattr(model.architecture, "megatron_spec", None)
        and not slime.slime_model_script
    )
    if slime.image_overlay is not None:
        image = slime.image_overlay(image)
        object.__setattr__(slime, "image_overlay", None)

    for patch in slime.patch_files:
        image = image.add_local_file(
            patch,
            remote_path=f"/tmp/{os.path.basename(patch)}",
            copy=True,
        )

    if isinstance(dataset, HarborDataset) or isinstance(eval_dataset, HarborDataset):
        image = image.uv_pip_install(f"harbor=={HARBOR_PKG_VERSION}")

    image = _overlay_slime_source(image, slime)

    if slime.image_run_commands:
        image = image.run_commands(*slime.image_run_commands)
    if slime.image_env:
        image = image.env(slime.image_env)

    if slime.metrics is not None and slime.metrics.provider == "trackio":
        resolve_trackio_destination(slime.metrics)
    image = apply_metric_image(image, slime.metrics)
    image = image.add_local_python_source("modal_dojo", copy=True)
    image = image.uv_pip_install("randomname")
    image = mount_tools_dir(image)
    image = mount_caller_source(image, caller_script)

    # Patch both conversion and training images for hybrid models.
    # The validation patch lets save/load succeed despite non-uniform
    # layer parameters.  The torch.py patch handles BytesIO entries
    # from _extra_state during checkpoint loading.
    if _has_hybrid_spec:
        image = image.run_commands(
            f"echo {_PATCH_VALIDATION_B64} | base64 -d | python3",
        )

    image = shared.ship_recipe_callables(
        image,
        slime,
        caller_script=caller_script,
        reward_post_process_in_config=True,
    )

    # ── SGLang request params auto-wiring ─────────────────────────────────
    if slime.sglang_request_params:
        cfg = dict(slime.extra_config or {})
        cfg["sglang_request_params"] = slime.sglang_request_params
        if "custom_rm_path" not in cfg:
            cfg["custom_rm_path"] = "modal_dojo.frameworks.slime.opd_reward.reward_func"
        if "custom_reward_post_process_path" not in cfg:
            cfg["custom_reward_post_process_path"] = (
                "modal_dojo.frameworks.slime.opd_reward.post_process_rewards"
            )
        object.__setattr__(slime, "extra_config", cfg)

    train_image = image
    if _has_hybrid_spec:
        train_image = image.run_commands(
            f"echo {_PATCH_TORCH_LOAD_B64} | base64 -d | python3",
            f"echo {_PATCH_GLOBAL_PLAN_B64} | base64 -d | python3",
            f"echo {_PATCH_CHECKPOINT_SAVE_B64} | base64 -d | python3",
        )
    train_image = train_image.run_commands(
        f"echo {_PATCH_CHECKPOINT_COMMIT_B64} | base64 -d | python3"
    )
    if _has_gdn:
        train_image = train_image.run_commands(
            f"echo {_PATCH_GDN_PACKED_SEQ_B64} | base64 -d | python3",
        )
    if slime.megatron_to_hf_mode == "bridge":
        train_image = train_image.run_commands(
            f"echo {_PATCH_BRIDGE_PER_TOKEN_LOSS_B64} | base64 -d | python3",
        )

    # ── Volumes ──────────────────────────────────────────────────────────────
    checkpoints_volume_name, checkpoints_mount_path, all_volumes = (
        shared.create_training_volumes(
            checkpoint,
            volume_prefix=volume_prefix,
            data_volume_name=slime.data_volume_name,
            mount_metadata=True,
        )
    )
    hf_cache_volume = all_volumes[str(HF_CACHE_PATH)]
    data_volume = all_volumes[str(DATA_PATH)]
    checkpoints_volume = all_volumes[checkpoints_mount_path]
    checkpoint_dir = compute_recipe_save_root(
        slime,
        recipe_default_save_root=str(CHECKPOINTS_PATH),
        mounted_save_root=checkpoints_mount_path,
        training_run_id=training_run_id,
    )

    tags = build_app_tags(
        framework="slime",
        model=model,
        recipe_app_tags=slime.app_tags,
        metrics=slime.metrics,
    )
    app = App(app_name, tags=tags)
    gpu_spec = f"{slime.gpu_type}:{slime.gpu_allocation.gpus_per_node}"

    register_recipe_functions(
        app,
        image,
        slime,
        hf_cache_volume=hf_cache_volume,
        data_volume=data_volume,
        checkpoints_volume=checkpoints_volume,
        checkpoints_mount_path=checkpoints_mount_path,
        download_phase=SlimeStatus.DOWNLOAD_MODEL.value,
        download=model.download,
        download_timeout=6 * 60 * 60,
        prepare_dataset=lambda: write_datasets(
            dataset, eval_dataset, dataset_path, eval_dataset_path
        ),
    )

    convert_nnodes, convert_nproc, _ = get_checkpoint_conversion_policy(
        slime, model=model
    )
    convert_gpu = f"{slime.gpu_type}:{convert_nproc}"

    @app.function(
        image=image,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            checkpoints_mount_path: checkpoints_volume,
        },
        timeout=60 * 60,
        secrets=[*hf_secrets(), *proxy_auth_secrets()],
        serialized=True,
        single_use_containers=True,
        name="resolve_checkpoint",
    )
    def resolve_checkpoint(
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ) -> str | None:
        report_phase(
            training_run_id,
            SlimeStatus.CONVERT_MODEL.value,
            framework_status_url,
            framework_status_token,
        )

        # Bridge mode loads HF weights directly into Megatron at train time.
        if getattr(slime, "megatron_to_hf_mode", None) == "bridge":
            print(
                "Bridge mode — HF weights loaded directly via AutoBridge; no conversion needed."
            )
            flush_status_reporter(timeout_seconds=2.0)
            return None

        hf_cache_volume.reload()
        checkpoints_volume.reload()

        save_path = str(slime.ref_load)
        current_config = _build_conversion_config(slime, model=model)
        cache_status, stored_config = _checkpoint_conversion_cache_status(
            save_path, current_config
        )
        if cache_status == "hit":
            print(f"Using existing torch_dist checkpoint at {save_path}.")
            flush_status_reporter(timeout_seconds=2.0)
            return None
        print(f"torch_dist checkpoint at {save_path} is {cache_status}.")

        if cache_status == "stale":
            if stored_config is None:
                print(
                    f"Checkpoint at {save_path} has missing or unreadable conversion "
                    "config metadata — reconverting."
                )
            else:
                print(
                    f"Checkpoint at {save_path} was built with different config:"
                    f"\n  stored: {stored_config}\n  current: {current_config}"
                )
        if cache_status in {"stale", "incomplete"}:
            print(f"Removing {cache_status} torch_dist checkpoint at {save_path}.")
            import shutil

            shutil.rmtree(save_path, ignore_errors=True)
            checkpoints_volume.commit()

        if slime.megatron_conversion_hf_checkpoint:
            return resolve_checkpoint_ref(slime.megatron_conversion_hf_checkpoint)
        if model.model_path:
            return str(model.model_path)

        from huggingface_hub import snapshot_download

        return snapshot_download(model.model_name, local_files_only=True)

    @app.function(
        image=image,
        gpu=convert_gpu,
        memory=slime.memory,
        cpu=slime.cpu,
        cloud=slime.cloud,
        region=slime.region,
        volumes=all_volumes,
        timeout=4 * 60 * 60,
        secrets=proxy_auth_secrets() or None,
        experimental_options=shared.experimental_options(
            slime, clustered=convert_nnodes > 1
        ),
        serialized=True,
        single_use_containers=True,
        name="convert_checkpoint",
    )
    @clustered_if(convert_nnodes > 1, convert_nnodes, gpu_type=slime.gpu_type)
    def convert_checkpoint(
        hf_path: str,
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        save_path = str(slime.ref_load)

        num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(
            slime, model=model
        )
        node_rank, master_addr, _, nnodes = get_modal_cluster_context(num_nodes)

        import json

        current_config = _build_conversion_config(slime, model=model)

        torchrun_args = [f"--nproc-per-node={nproc_per_node}"]
        if nnodes > 1:
            torchrun_args += [
                f"--nnodes={nnodes}",
                f"--node-rank={node_rank}",
                f"--master-addr={master_addr}",
                "--master-port=12355",
            ]

        import importlib.util

        mmt = ""
        if model and getattr(model, "architecture", None):
            mmt = getattr(model.architecture, "megatron_model_type", "")

        spec = importlib.util.find_spec(
            "modal_dojo.frameworks.slime.modal_helpers.convert_hf_to_torch_dist"
        )
        convert_script = spec.origin if spec is not None else None
        if not convert_script:
            raise RuntimeError(
                "modal_dojo.frameworks.slime.modal_helpers.convert_hf_to_torch_dist not found"
            )
        if mmt or slime.slime_model_script:
            model_script = (
                f"{SLIME_ROOT}/{slime.slime_model_script}"
                if slime.slime_model_script
                else f"{SLIME_ROOT}/scripts/models/{mmt}.sh"
            )
            cmd = (
                f"source {model_script} && "
                f"torchrun {' '.join(torchrun_args)} {convert_script} "
                '"${MODEL_ARGS[@]}" '
                f"{' '.join(extra_args)} "
                f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
            )
        else:
            cmd = (
                f"torchrun {' '.join(torchrun_args)} {convert_script} "
                f"{' '.join(extra_args)} "
                f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
            )

        env = {**os.environ, **slime.environment}
        env.pop("NCCL_NVLS_ENABLE", None)
        if any(arg.startswith("--pipeline-model-parallel-size ") for arg in extra_args):
            env["SKIP_PP_AUTOINFLATE"] = "1"
        if num_nodes > 1:
            env["SKIP_RELEASE_RENAME"] = "1"
        print(
            f"Conversion layout: nodes={num_nodes}, "
            f"nproc_per_node={nproc_per_node}, node_rank={node_rank}"
        )
        print(f"Running: bash -c {cmd!r}")
        subprocess.run(["bash", "-c", cmd], check=True, env=env)

        if node_rank == 0:
            config_path = os.path.join(save_path, _CONVERSION_CONFIG_FILE)
            try:
                with open(config_path, "w") as f:
                    json.dump(current_config, f)
            except OSError as exc:
                print(f"WARNING: could not write conversion config: {exc}")
        checkpoints_volume.commit()

        if node_rank == 0:
            print(f"Saved torch_dist checkpoint to {save_path}")

        flush_status_reporter(timeout_seconds=2.0)

    # Use Modal's clustered scheduler with RDMA when using a full node (8+ GPUs)
    # on RDMA-capable hardware, or for any multi-node run.  The `rdma=True` flag
    # provides CAP_IPC_LOCK and NVSwitch device access that slime's colocated
    # weight sync (UpdateWeightFromTensor) needs for fast CUDA IPC transfers.
    _multi_node = slime.total_nodes > 1
    _full_node = slime.actor_num_gpus_per_node >= 8
    _use_clustered = _multi_node or (_full_node and _supports_rdma(slime.gpu_type))

    train_secrets: list[Secret] = []
    if slime.metrics is not None:
        train_secrets.extend(metric_secrets(slime.metrics))
        if (
            slime.metrics.provider == "trackio"
            and getattr(slime.metrics, "modal_secret_name", "") == "huggingface-secret"
        ):
            train_secrets.extend(hf_secrets())
    # Proxy-auth tokens for any custom_rm / generate hook that calls a
    # CustomDeployment.launch() endpoint (teacher /generate, etc.).
    train_secrets.extend(proxy_auth_secrets())

    @app.function(
        image=train_image,
        volumes=all_volumes,
        **shared.training_function_options(
            slime,
            framework="slime",
            secrets=train_secrets,
            experimental_options={},
        ),
    )
    @clustered_if(_use_clustered, slime.total_nodes, gpu_type=slime.gpu_type)
    async def train(
        modal_app_id: str = "",
        modal_app_url: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        cluster, modal_app_id, modal_app_url = await start_training_cluster(
            slime,
            (hf_cache_volume, data_volume, checkpoints_volume),
            "SLIME_HOST_IP",
            "SGLANG_HOST_IP",
            modal_app_id=modal_app_id,
            modal_app_url=modal_app_url,
            framework_status_url=framework_status_url,
            framework_status_token=framework_status_token,
        )
        cluster.start_ray()

        if not cluster.is_head:
            await cluster.wait_forever()
            return

        # Fail fast on tracker access before the framework starts training.
        metric_entity = preflight_metric(slime.metrics)

        print(f"Training run id: {training_run_id}")
        (
            run_record,
            metric_run_id,
            framework_status_token,
        ) = await init_training_run_record(
            training_run_id=training_run_id,
            modal_app_id=modal_app_id,
            modal_app_url=modal_app_url,
            framework=Framework.SLIME,
            initializing_status=SlimeStatus.INITIALIZING,
            recipe=slime,
            model=model,
            dataset=dataset,
            eval_dataset=eval_dataset,
            dataset_path=dataset_path,
            eval_dataset_path=eval_dataset_path,
            recipe_metadata=(
                "slime_git_repository",
                "slime_git_revision",
                "data_volume_name",
            ),
            metric_entity=metric_entity,
            framework_status_token=framework_status_token,
            checkpoint_dir=checkpoint_dir,
            checkpoints_volume_name=checkpoints_volume_name,
            checkpoints_mount_path=checkpoints_mount_path,
        )

        async with shared.training_run_lifecycle(
            run_record, framework_status_token
        ) as set_status:
            if model:
                await set_status(SlimeStatus.DOWNLOAD_MODEL)
                model.download()
                await hf_cache_volume.commit.aio()

            if dataset:
                await set_status(SlimeStatus.PREPARE_DATASET)
                if write_datasets(
                    dataset, eval_dataset, dataset_path, eval_dataset_path
                ):
                    await data_volume.commit.aio()

            await set_status(SlimeStatus.CONVERT_MODEL)
            save_root = checkpoint_dir
            apply_scoped_save(slime, save_root)
            prepare_slime_config(slime, model, tempfile.mkdtemp())

            os.makedirs(save_root, exist_ok=True)

            # Resolve the local HF snapshot dir (used for bridge-mode load below).
            _hf_ref: str | None = None
            if model and (slime.megatron_to_hf_mode == "bridge" or slime.ref_load):
                from huggingface_hub import snapshot_download as _snap0

                _hf_ref = (
                    str(model.model_path)
                    if model.model_path
                    else _snap0(model.model_name, local_files_only=True)
                )

            resume_checkpoint = torch_dist_resume_checkpoint(
                save_root, is_complete=is_complete_torch_dist_checkpoint_dir
            )
            record_resume_checkpoint(run_record, resume_checkpoint)
            await run_record.save(is_async=True)

            with shared.resumed_recipe(slime, save_root, resume_checkpoint):
                if (
                    resume_checkpoint is None
                    and slime.megatron_to_hf_mode == "bridge"
                    and not slime.ref_load
                    and _hf_ref
                ):
                    object.__setattr__(slime, "ref_load", _hf_ref)
                cmd = build_train_cmd(
                    slime,
                    SLIME_ROOT,
                    model=model,
                    dataset=dataset,
                    eval_dataset=eval_dataset,
                    dataset_path=dataset_path,
                    eval_dataset_path=eval_dataset_path,
                )

            runtime_env = {
                "env_vars": {
                    **slime.environment,
                    "no_proxy": ",".join(
                        value
                        for value in (
                            "127.0.0.1",
                            cluster.head_addr,
                            slime.environment.get("no_proxy", ""),
                        )
                        if value
                    ),
                    "MASTER_ADDR": cluster.head_addr,
                    **shared.training_reporting_env(
                        slime, model, app_name, framework_status_url
                    ),
                    "MODAL_DOJO_SUBSTEP_TIMING": slime.substep_timing,
                    **metric_runtime_env(
                        slime.metrics,
                        run_id=metric_run_id,
                        entity=metric_entity,
                    ),
                    **timing_debug_env(),
                    "MODAL_DOJO_TRAINING_RUN_ID": training_run_id,
                    "MODAL_DOJO_CHECKPOINTS_VOLUME_NAME": checkpoints_volume_name,
                    "MODAL_DOJO_FRAMEWORK_STATUS_TOKEN": framework_status_token,
                }
            }

            mode = "async" if slime.train_async else "sync"
            print(
                f"Training {app_name} — {slime.total_nodes} node(s) × {gpu_spec}  ({mode})"
            )
            print(slime.gpu_allocation.summary())
            print(f"Command: {cmd}")
            print(f"Runtime environment variables: {sorted(runtime_env['env_vars'])}")

            await set_status(SlimeStatus.ROLLOUT_INITIALIZING)
            async with cluster.forward_dashboard() as tunnel:
                print(f"Ray dashboard: {tunnel.url}")
                result = await cluster.submit_and_tail(cmd, runtime_env=runtime_env)
                shared.check_training_result(result, run_record)

            return await shared.complete_training_run(
                run_record,
                checkpoints_volume=checkpoints_volume,
                app_name=app_name,
                model=model,
                group_id=group_id,
            )

    app.resolve_checkpoint = resolve_checkpoint
    app.convert_checkpoint = convert_checkpoint
    app.train = train

    return app
