import asyncio
import contextlib
import hashlib
import os
import shlex
import subprocess
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from modal import App, Dict as ModalDict, Image

from modal_dojo.common import (
    hf_secrets,
    proxy_auth_secrets,
)
from modal_dojo.common.checkpoint import Checkpoint
from modal_dojo.common.dataset import DatasetConfig, HarborDataset, _SftDataset
from modal_dojo.common.framework import (
    Framework,
    mount_tools_dir,
)
from modal_dojo.common.launcher_utils import (
    is_local_checkpoint_ref,
    materialize_remote_code,
    prewarm_remote_code,
    timing_debug_env,
)
from modal_dojo.common.metrics import (
    apply_metric_image,
    metric_runtime_env,
    metric_secrets,
    preflight_metric,
)
from modal_dojo.common.models import ModelConfig
from modal_dojo.common.ray_cluster import (
    clustered_if,
)
from modal_dojo.common.run import (
    TrainingRun,
    has_torch_dist_checkpoint,
    record_resume_checkpoint,
    torch_dist_resume_checkpoint,
)
from modal_dojo.common import launcher_helpers as shared
from modal_dojo.common.launcher_helpers import (
    compute_recipe_save_root,
    build_app_tags,
    apply_scoped_save,
    configured_recipe_save,
    init_training_run_record,
    mount_caller_source,
    resolve_caller_context,
    register_recipe_functions,
    report_phase,
    start_training_cluster,
    write_datasets,
)
from modal_dojo.common.status_reporter import flush as flush_status_reporter
from modal_dojo.common.status import MilesStatus
from modal_dojo.common.torch_dist_checkpoint import (
    is_complete_torch_dist_checkpoint_dir,
)
from modal_dojo.train_recipes.miles_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    MilesRecipe,
)
from modal_dojo.common.patches import _MEGATRON_PATCHES, encode_patch
from modal_dojo.frameworks.miles.modal_helpers.utils import (
    build_train_cmd,
    get_checkpoint_conversion_policy,
    model_args_command,
    prepare_miles_config,
    resolve_checkpoint_ref,
)


MILES_ROOT = "/root/miles"
# Editable install location of sglang inside the miles images.
SGLANG_ROOT = "/sgl-workspace/sglang"
# Disagg multi-node mooncake needs matching libibverbs/libmlx5. The apt
# reinstall strips NCCL NET plugins, so 2-node SGLang dies in
# ncclCommInitRank ("invalid usage" / "Failed to initialize any NET
# plugin"). Colocate syncs over CUDA IPC and must keep the image NET
# stack. 1-node jobs never take this path.
RDMA_RUNTIME_INSTALL_COMMAND = (
    "apt-get update && apt-get install -y --no-install-recommends "
    "--reinstall libibverbs1 ibverbs-providers && "
    "rm -rf /var/lib/apt/lists/*"
)
# v0.8.0+ makes per-task CPU/memory requests configurable via enforcement
# policies ("limit"/"ignore"), letting sandboxes burst on Modal and bill by
# actual CPU-/RAM-second usage instead of over-provisioning a static reservation.
HARBOR_PKG_VERSION = "0.8.0"

_MILES_PATCHES = Path(__file__).parent / "modal_helpers" / "patches"
_PATCH_SGLANG_ABORT_B64 = encode_patch("patch_sglang_abort", _MILES_PATCHES)
_PATCH_ROLLOUT_STATUS_B64 = encode_patch(
    "patch_rollout_status_reporting", _MILES_PATCHES
)
_PATCH_ADVANTAGE_DIST_B64 = encode_patch("patch_advantage_distribution", _MILES_PATCHES)
_PATCH_SUBSTEP_TIMING_B64 = encode_patch("patch_substep_timing", _MILES_PATCHES)
_PATCH_ZERO_STD_B64 = encode_patch("patch_zero_std_metrics", _MILES_PATCHES)

_REPORTING_PATCH_COMMANDS = (
    f"echo {_PATCH_ZERO_STD_B64} | base64 -d | python3",
    f"echo {_PATCH_ROLLOUT_STATUS_B64} | base64 -d | python3",
    f"echo {_PATCH_ADVANTAGE_DIST_B64} | base64 -d | python3",
)

_PATCH_DIST_CKPT_QUANTIZED_B64 = encode_patch(
    "patch_dist_ckpt_quantized", _MEGATRON_PATCHES
)
_PATCH_DIST_CKPT_NOFORK_B64 = encode_patch("patch_dist_ckpt_nofork", _MEGATRON_PATCHES)
_PATCH_CHECKPOINT_SAVE_B64 = encode_patch("patch_checkpoint_save", _MEGATRON_PATCHES)
_PATCH_CHECKPOINT_COMMIT_B64 = encode_patch(
    "patch_checkpoint_commit", _MEGATRON_PATCHES
)
_MEGATRON_TORCH_STRATEGY_PY = (
    "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py"
)


_CONVERT_LOCK_DICT_NAME = "training-gym-convert-lock"
_CONVERT_LOCK_TTL_S = 900.0
_CONVERT_LOCK_REFRESH_S = 300.0
_CONVERT_TOKEN_TTL_S = 4 * _CONVERT_LOCK_TTL_S


def _convert_lock_dict() -> Any:
    return ModalDict.from_name(_CONVERT_LOCK_DICT_NAME, create_if_missing=True)


def _convert_lock_key(volume_name: str, save_path: str) -> str:
    return f"lock:{volume_name}:{save_path}"


def _sweep_convert_tokens(locks: Any, key: str) -> None:
    """Drop takeover tokens old enough that no live contender can act on one.

    A token has to outlive the takeover it settles, or a contender still holding the
    stale read it was written for could win a fresh one and pop the new owner's claim.
    Past that it is only litter, and nothing else reclaims it, so tokens are swept on
    release and on the next takeover for the same path.
    """
    prefix = f"{key}:takeover:"
    cutoff = time.time() - _CONVERT_TOKEN_TTL_S
    try:
        names = [str(name) for name in locks.keys() if str(name).startswith(prefix)]
        for name in names:
            token = locks.get(name)
            written_at = token.get("at") if isinstance(token, dict) else None
            if not isinstance(written_at, (int, float)) or written_at < cutoff:
                locks.pop(name, None)
    except Exception as exc:
        print(f"WARNING: could not sweep stale conversion tokens: {exc}")


def _acquire_convert_lock(run_id: str, volume_name: str, save_path: str) -> str:
    """Claim the right to convert into ``save_path``; return the current holder.

    Two launches of the same recipe share one ``ref_load``, so without this the
    loser's cleanup deletes the winner's still-metadata-less conversion output.

    ``Dict.put(..., skip_if_exists=True)`` is the atomic compare-and-set that decides
    the winner; a read-then-write claim would let two runs both observe an unheld lock
    and both proceed. The claim expires after ``_CONVERT_LOCK_TTL_S``, sized against the
    heartbeat interval so a holder that cannot run its release path frees the path in a
    TTL rather than a conversion's worth of time.

    Taking a stale claim over cannot be a delete followed by a put: two runs that both
    read the same stale holder would delete each other's fresh claim and both come away
    believing they own the path. Instead each contender first claims a token naming the
    stale holder it saw, which only one can win, and only that winner replaces the
    claim. Those tokens are swept by age, not on use — see ``_sweep_convert_tokens``.
    """
    locks = _convert_lock_dict()
    key = _convert_lock_key(volume_name, save_path)
    claim = {"run_id": run_id, "claimed_at": time.time()}
    if locks.put(key, claim, skip_if_exists=True):
        return run_id

    owner, claimed_at = "", None
    holder = locks.get(key)
    if isinstance(holder, dict):
        owner = str(holder.get("run_id") or "")
        claimed_at = holder.get("claimed_at")
        if owner == run_id:
            return run_id
        if (
            owner
            and isinstance(claimed_at, (int, float))
            and time.time() - claimed_at < _CONVERT_LOCK_TTL_S
        ):
            return owner

    takeover_key = f"{key}:takeover:{owner}:{claimed_at}"
    token = {"run_id": run_id, "at": time.time()}
    if not locks.put(takeover_key, token, skip_if_exists=True):
        winner = locks.get(takeover_key)
        if isinstance(winner, dict) and winner.get("run_id"):
            return str(winner["run_id"])
        return owner or run_id

    _sweep_convert_tokens(locks, key)
    try:
        locks.pop(key)
    except KeyError:
        pass
    if locks.put(key, claim, skip_if_exists=True):
        return run_id
    holder = locks.get(key)
    if isinstance(holder, dict) and holder.get("run_id"):
        return str(holder["run_id"])
    return run_id


def _refresh_convert_lock(run_id: str, volume_name: str, save_path: str) -> None:
    """Extend an existing claim of this run's; never create one.

    A refresh that wrote unconditionally would resurrect the claim when a tick
    straddles the release, leaving it held with nobody to give it back.
    """
    locks = _convert_lock_dict()
    key = _convert_lock_key(volume_name, save_path)
    holder = locks.get(key)
    if not isinstance(holder, dict) or str(holder.get("run_id") or "") != run_id:
        return
    locks[key] = {"run_id": run_id, "claimed_at": time.time()}


@contextlib.contextmanager
def _convert_lock_heartbeat(run_id: str, volume_name: str, save_path: str):
    """Keep this run's claim fresh for as long as the body runs.

    Refreshing on elapsed time rather than on work done: a torch_dist conversion
    writes only about one file per rank, so any progress-based cadence never fires,
    and the claim would lapse mid-conversion and let a later launch delete the output
    being written.
    """
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(_CONVERT_LOCK_REFRESH_S):
            try:
                _refresh_convert_lock(run_id, volume_name, save_path)
            except Exception as exc:
                print(f"WARNING: could not refresh conversion claim: {exc}")

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5.0)


def _release_convert_lock(run_id: str, volume_name: str, save_path: str) -> None:
    """Drop the conversion claim, but only if this run still holds it."""
    locks = _convert_lock_dict()
    key = _convert_lock_key(volume_name, save_path)
    holder = locks.get(key)
    if isinstance(holder, dict) and str(holder.get("run_id") or "") != run_id:
        return
    try:
        locks.pop(key)
    except KeyError:
        pass
    _sweep_convert_tokens(locks, key)


def _is_resumable_checkpoint(path: str) -> bool:
    """Whether ``path`` holds a training save that can be resumed from.

    Looser than ``is_complete_torch_dist_checkpoint_dir`` because miles writes
    more than one save shape: a LoRA run stores ``adapter/`` with per-rank
    ``.pt`` files and no ``.metadata``/``common.pt``/``.distcp`` at all, so the
    conversion predicate would report every adapter checkpoint as absent and
    silently restart from ``ref_load``. Only the crashed-torch_dist signature is
    rejected: ``.distcp`` shards present but the ``.metadata`` written last
    missing.
    """
    try:
        names = os.listdir(path)
    except OSError:
        return False
    if not names:
        return False
    if any(name.endswith(".distcp") for name in names):
        return ".metadata" in names
    return True


def _unresumable_save_dirs(save_root: str) -> list[str]:
    """Save directories that exist but cannot be resumed from."""
    try:
        names = os.listdir(save_root)
    except OSError:
        return []
    return sorted(
        name
        for name in names
        if (name == "release" or name.startswith("iter_"))
        and os.path.isdir(os.path.join(save_root, name))
        and not _is_resumable_checkpoint(os.path.join(save_root, name))
    )


def _build_miles_base_image(
    recipe: MilesRecipe,
    dataset: DatasetConfig | None = None,
    eval_dataset: DatasetConfig | None = None,
    caller_script: str | None = None,
) -> Image:
    image = (
        Image.from_registry(recipe.docker_image)
        .entrypoint([])
        .run_commands(
            f"rm -rf {HF_CACHE_PATH} 2>/dev/null || true",
            f"echo {_PATCH_SGLANG_ABORT_B64} | base64 -d | python3",
            f"echo {_PATCH_DIST_CKPT_QUANTIZED_B64} | base64 -d | python3",
            f"echo {_PATCH_DIST_CKPT_NOFORK_B64} | base64 -d | python3",
            (
                f"if test -f {_MEGATRON_TORCH_STRATEGY_PY}; then "
                f"echo {_PATCH_CHECKPOINT_SAVE_B64} | base64 -d | python3; "
                f"else echo 'WARNING: {_MEGATRON_TORCH_STRATEGY_PY} not found, "
                "skipping checkpoint-save patch'; fi"
            ),
            *_REPORTING_PATCH_COMMANDS,
            f"echo {_PATCH_SUBSTEP_TIMING_B64} | base64 -d | python3",
        )
    )
    if (
        recipe.total_nodes > 1
        and not recipe.colocate
        and recipe.environment.get("MILES_REINSTALL_RDMA", "1") != "0"
    ):
        image = image.run_commands(RDMA_RUNTIME_INSTALL_COMMAND)
    if recipe.image_env:
        image = image.env(recipe.image_env)

    for patch_file in recipe.patch_files:
        image = image.add_local_file(
            patch_file,
            remote_path=f"/tmp/{os.path.basename(patch_file)}",
            copy=True,
        )

    if recipe.local_miles:
        image = image.add_local_dir(
            recipe.local_miles,
            remote_path=MILES_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )
        image = image.run_commands(
            f"echo {_PATCH_SGLANG_ABORT_B64} | base64 -d | python3"
            " || echo 'WARNING: sglang abort patch did not apply to the"
            " local_miles checkout; transient router failures during rollout"
            " cleanup may crash the run'",
            *_REPORTING_PATCH_COMMANDS,
        )

    image = apply_source_overlays(image, recipe)

    if recipe.image_run_commands:
        image = image.run_commands(*recipe.image_run_commands)

    if recipe.image_overlay is not None:
        image = recipe.image_overlay(image)
        recipe.image_overlay = None

    if isinstance(dataset, HarborDataset) or isinstance(eval_dataset, HarborDataset):
        image = image.uv_pip_install(f"harbor=={HARBOR_PKG_VERSION}")

    image = apply_metric_image(image, recipe.metrics)
    image = image.add_local_python_source("modal_dojo", copy=True)
    image = image.uv_pip_install("randomname")
    image = mount_tools_dir(image)
    image = mount_caller_source(image, caller_script)

    image = shared.ship_recipe_callables(
        image,
        recipe,
        caller_script=caller_script,
        reward_post_process_in_config=False,
    )

    image = image.run_commands(
        f"echo {_PATCH_CHECKPOINT_COMMIT_B64} | base64 -d | python3"
    )

    return image


def _remap_kernel_cache_dirs(
    environment: dict[str, str], checkpoints_mount_path: str
) -> dict[str, str]:
    environment = dict(environment)
    for key in (
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TILELANG_CACHE_DIR",
        "SGLANG_CACHE_DIR",
    ):
        cache_dir = environment.get(key)
        if cache_dir and Path(cache_dir).is_relative_to(CHECKPOINTS_PATH):
            environment[key] = str(
                Path(checkpoints_mount_path)
                / Path(cache_dir).relative_to(CHECKPOINTS_PATH)
            )
    return environment


def build_ray_runtime_env(
    *,
    head_addr: str,
    metric_env: dict[str, str],
    environment: dict,
    extra_env: dict[str, str] | None = None,
    framework_status_token: str = "",
    substep_timing: str = "auto",
) -> dict:
    """Runtime env for the Ray job that runs miles.

    Ray workers inherit the container's ``LD_LIBRARY_PATH``. On EFA workers
    Modal puts its host-mounted ``/opt/amazon`` lib dirs first there, which
    the aws-ofi-nccl plugin needs to resolve the host libfabric rather than
    an older copy in the image. A recipe can override it through
    ``environment``.
    """
    env_vars: dict[str, str] = {
        "no_proxy": f"127.0.0.1,{head_addr}",
        "MASTER_ADDR": head_addr,
        "MODAL_DOJO_SUBSTEP_TIMING": substep_timing,
    }
    env_vars.update(environment)
    # Tracker identity and credentials must match the preflight configuration.
    env_vars.update(metric_env)
    env_vars.update(timing_debug_env())
    env_vars.update(extra_env or {})
    if framework_status_token:
        # Applied after `environment` so a recipe override can't blank the
        # dashboard auth token by accident.
        env_vars["MODAL_DOJO_FRAMEWORK_STATUS_TOKEN"] = framework_status_token
    return {"env_vars": env_vars}


def apply_source_overlays(image: Image, miles: MilesRecipe) -> Image:
    if miles.sglang_git_ref:
        image = image.run_commands(
            f"cd {SGLANG_ROOT} && git fetch --depth=1 -- origin"
            f" {shlex.quote(miles.sglang_git_ref)} && git checkout -f FETCH_HEAD"
        )

    if miles.miles_git_ref:
        image = image.run_commands(
            f"cd {MILES_ROOT} && git fetch --depth=1 -- origin"
            f" {shlex.quote(miles.miles_git_ref)} && git checkout -f FETCH_HEAD",
            # The checkout just reverted the patched miles sources.
            f"echo {_PATCH_SGLANG_ABORT_B64} | base64 -d | python3"
            " || echo 'WARNING: sglang abort patch did not apply to the"
            " miles_git_ref checkout; transient router failures during rollout"
            " cleanup may crash the run'",
            *_REPORTING_PATCH_COMMANDS,
            f"echo {_PATCH_SUBSTEP_TIMING_B64} | base64 -d | python3"
            " || echo 'WARNING: substep timing patch did not apply to the"
            " miles_git_ref checkout; substep timings will be missing'",
        )
    return image


def build_miles_app(
    *,
    training_run_id: str,
    miles: MilesRecipe,
    model: ModelConfig,
    dataset: DatasetConfig,
    eval_dataset: DatasetConfig | None = None,
    checkpoint: Checkpoint | None = None,
    name: str | None = None,
    group_id: str | None = None,
) -> App:
    app_name = name or miles.name or f"miles-{type(miles).__name__.lstrip('_').lower()}"
    volume_prefix = miles.name or f"miles-{type(miles).__name__.lstrip('_').lower()}"
    MilesRecipe._validate_datasets(dataset, eval_dataset, loss_type=miles.loss_type)
    if miles.loss_type == "sft_loss":
        dataset = _SftDataset(dataset)
    dataset_path = MilesRecipe._resolve_data_paths(dataset)
    eval_dataset_path = (
        MilesRecipe._resolve_data_paths(eval_dataset)
        if eval_dataset is not None
        else None
    )

    _caller_module, caller_script = resolve_caller_context()

    image = _build_miles_base_image(miles, dataset, eval_dataset, caller_script)

    checkpoints_volume_name, checkpoints_mount_path, all_volumes = (
        shared.create_training_volumes(
            checkpoint,
            volume_prefix=volume_prefix,
        )
    )
    hf_cache_volume = all_volumes[str(HF_CACHE_PATH)]
    data_volume = all_volumes[str(DATA_PATH)]
    checkpoints_volume = all_volumes[checkpoints_mount_path]
    environment = _remap_kernel_cache_dirs(miles.environment, checkpoints_mount_path)
    checkpoint_dir = compute_recipe_save_root(
        miles,
        recipe_default_save_root=str(CHECKPOINTS_PATH),
        mounted_save_root=checkpoints_mount_path,
        training_run_id=training_run_id,
    )
    recorded_checkpoint_dir = checkpoint_dir if configured_recipe_save(miles) else ""

    tags = build_app_tags(
        framework="miles",
        model=model,
        recipe_app_tags=miles.app_tags,
        metrics=miles.metrics,
    )
    app = App(app_name, tags=tags)
    gpu_spec = f"{miles.gpu_type}:{miles.gpu_allocation.gpus_per_node}"
    download_timeout = miles.download_timeout_seconds or 4 * 60 * 60
    dataset_timeout = 4 * 60 * 60

    def materialize_model_remote_code() -> None:
        # Rewrite remote-code symlinks before the head commits the cache Volume.
        if model and model.model_name and not is_local_checkpoint_ref(model.model_name):
            materialize_remote_code(
                resolve_checkpoint_ref(model.model_path or model.model_name)
            )

    def download_inputs() -> None:
        model.download()
        materialize_model_remote_code()
        miles.download_model()
        miles.post_process_model()

    register_recipe_functions(
        app,
        image,
        hf_cache_volume=hf_cache_volume,
        data_volume=data_volume,
        checkpoints_volume=checkpoints_volume,
        checkpoints_mount_path=checkpoints_mount_path,
        download_phase=MilesStatus.DOWNLOAD_MODEL.value,
        download=download_inputs,
        download_timeout=download_timeout,
        prepare_dataset=lambda: write_datasets(
            dataset, eval_dataset, dataset_path, eval_dataset_path
        ),
        dataset_timeout=dataset_timeout,
    )

    convert_nnodes, convert_nproc, _ = get_checkpoint_conversion_policy(
        miles, model=model
    )
    convert_gpu = f"{miles.gpu_type}:{convert_nproc}"
    convert_multi_node = convert_nnodes > 1

    @app.function(
        image=image,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            checkpoints_mount_path: checkpoints_volume,
        },
        timeout=60 * 60,
        secrets=[*hf_secrets(), *proxy_auth_secrets()],
        serialized=True,
        name="resolve_checkpoint",
    )
    def resolve_checkpoint(
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ) -> str | None:
        report_phase(
            training_run_id,
            MilesStatus.CONVERT_MODEL.value,
            framework_status_url,
            framework_status_token,
        )

        if getattr(miles, "megatron_to_hf_mode", None) == "bridge":
            print("Bridge mode - no conversion needed.")
            flush_status_reporter(timeout_seconds=2.0)
            return None

        hf_cache_volume.reload()
        checkpoints_volume.reload()

        save_path = str(miles.ref_load)
        if has_torch_dist_checkpoint(
            save_path, is_complete=is_complete_torch_dist_checkpoint_dir
        ):
            print(
                f"Found existing torch_dist checkpoint at {save_path}; "
                "skipping conversion."
            )
            flush_status_reporter(timeout_seconds=2.0)
            return None

        holder = _acquire_convert_lock(
            training_run_id, checkpoints_volume_name, save_path
        )
        if holder != training_run_id:
            raise RuntimeError(
                f"Run {holder} is already converting into {save_path}. Two runs of "
                "this recipe share one ref_load, so continuing would delete that "
                "run's in-flight conversion and interleave shards. Wait for it to "
                "finish and relaunch to pick up the cached checkpoint, or point this "
                "run at a different ref_load."
            )

        try:
            # Clear partial torch_dist writes from an earlier crash here — the
            # single-container step that decides to convert — so the conversion cannot mix
            # fresh shards with stale ones from a different parallelism. Only the
            # ``iter_*``/``release`` directories the converter itself writes, plus the
            # iteration tracker the resume scan prefers over them, are removed, and only
            # those failing the completeness check: ``ref_load`` is user-settable
            # and may hold a hand-placed checkpoint in a layout this predicate rejects.
            if os.path.isdir(save_path):
                stale = [
                    name
                    for name in sorted(os.listdir(save_path))
                    if (name == "release" or name.startswith("iter_"))
                    and os.path.isdir(os.path.join(save_path, name))
                    and not is_complete_torch_dist_checkpoint_dir(
                        os.path.join(save_path, name)
                    )
                ]
                tracker = "latest_checkpointed_iteration.txt"
                tracker_path = os.path.join(save_path, tracker)
                has_tracker = stale and os.path.isfile(tracker_path)
                if stale:
                    print(
                        f"Removing incomplete torch_dist checkpoint state at {save_path}: "
                        + ", ".join([*stale, *([tracker] if has_tracker else [])])
                    )
                    for name in stale:
                        shutil.rmtree(os.path.join(save_path, name), ignore_errors=True)
                    if has_tracker:
                        os.remove(tracker_path)
                    checkpoints_volume.commit()

            conversion_hf_checkpoint = (
                getattr(miles, "megatron_conversion_hf_checkpoint", None)
                or getattr(miles, "hf_checkpoint", "")
                or model.model_path
                or model.model_name
            )
            return resolve_checkpoint_ref(conversion_hf_checkpoint)
        except BaseException:
            _release_convert_lock(training_run_id, checkpoints_volume_name, save_path)
            raise

    @app.function(
        image=image,
        gpu=convert_gpu,
        volumes=all_volumes,
        timeout=miles.convert_timeout_seconds or 4 * 60 * 60,
        secrets=proxy_auth_secrets() or None,
        # torch_dist stages each rank's shard through host RAM.
        memory=miles.memory,
        ephemeral_disk=miles.convert_ephemeral_disk_mb,
        experimental_options=shared.experimental_options(miles),
        serialized=True,
        name="convert_checkpoint",
    )
    @clustered_if(
        convert_multi_node,
        convert_nnodes,
        gpu_type=miles.gpu_type,
    )
    def convert_checkpoint(
        hf_path: str,
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        save_path = str(miles.ref_load)
        num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(
            miles, model=model
        )

        if num_nodes == 1:
            node_rank, master_addr, nnodes = 0, "127.0.0.1", 1
        else:
            import modal.experimental

            info = modal.experimental.get_cluster_info()
            node_rank = info.rank
            master_addr = info.container_ipv4_ips[0]
            nnodes = len(info.container_ipv4_ips)

        torchrun_args = [f"--nproc-per-node={nproc_per_node}"]
        if nnodes > 1:
            torchrun_args += [
                f"--nnodes={nnodes}",
                f"--node-rank={node_rank}",
                f"--master-addr={master_addr}",
                "--master-port=12355",
            ]

        import importlib.util

        spec = importlib.util.find_spec(
            "modal_dojo.frameworks.miles.modal_helpers.convert_hf_to_torch_dist"
        )
        convert_script = spec.origin if spec is not None else None
        if not convert_script:
            raise RuntimeError(
                "modal_dojo.frameworks.miles.modal_helpers.convert_hf_to_torch_dist not found"
            )

        if miles.miles_model_script:
            cmd = (
                f"source {MILES_ROOT}/{miles.miles_model_script} && "
                f"torchrun {' '.join(torchrun_args)} {convert_script} "
                f"${{MODEL_ARGS[@]}} {' '.join(extra_args)} "
                f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
            )
        elif model_args_cmd := model_args_command(miles, MILES_ROOT):
            cmd = (
                f'MODEL_ARGS_LINE="$({model_args_cmd})" || exit 1; '
                f'read -ra MODEL_ARGS <<< "$MODEL_ARGS_LINE"; '
                f"torchrun {' '.join(torchrun_args)} {convert_script} "
                f"${{MODEL_ARGS[@]}} {' '.join(extra_args)} "
                f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
            )
        else:
            cmd = (
                f"torchrun {' '.join(torchrun_args)} {convert_script} "
                f"{' '.join(extra_args)} "
                f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
            )

        env = {**os.environ, **environment}
        if any(arg.startswith("--pipeline-model-parallel-size ") for arg in extra_args):
            env["CONVERT_KEEP_PP1"] = "1"
        if num_nodes > 1:
            env["SKIP_RELEASE_RENAME"] = "1"
        print(
            f"Conversion layout: nodes={num_nodes}, nproc_per_node={nproc_per_node}, "
            f"node_rank={node_rank}"
        )
        print(f"Running: bash -c {cmd!r}")
        if node_rank == 0:
            _refresh_convert_lock(training_run_id, checkpoints_volume_name, save_path)
        heartbeat = (
            _convert_lock_heartbeat(training_run_id, checkpoints_volume_name, save_path)
            if node_rank == 0
            else contextlib.nullcontext()
        )
        try:
            with heartbeat:
                prewarm_remote_code(
                    hf_path, env, required=miles.model_name == "kimi_k3"
                )
                subprocess.run(["bash", "-c", cmd], check=True, env=env)

                checkpoints_volume.commit()

                if node_rank == 0:
                    print(f"Saved Megatron torch_dist checkpoint to {save_path}")
                    # Fail loudly here rather than leaving a partial checkpoint for a later
                    # run to mistake for a cache hit.
                    if not has_torch_dist_checkpoint(
                        save_path, is_complete=is_complete_torch_dist_checkpoint_dir
                    ):
                        raise RuntimeError(
                            f"Conversion finished but {save_path} holds no complete "
                            "torch_dist checkpoint (.metadata or .distcp shards missing)."
                        )
            if node_rank == 0:
                _release_convert_lock(
                    training_run_id, checkpoints_volume_name, save_path
                )
        except BaseException:
            if node_rank == 0:
                _release_convert_lock(
                    training_run_id, checkpoints_volume_name, save_path
                )
            raise

        flush_status_reporter(timeout_seconds=2.0)

    _multi_node = miles.total_nodes > 1

    train_secrets = [
        *(metric_secrets(miles.metrics) if miles.metrics is not None else []),
        *hf_secrets(),
        *proxy_auth_secrets(),
    ]

    @app.function(
        image=image,
        volumes=all_volumes,
        **shared.training_function_options(
            miles,
            framework="miles",
            secrets=train_secrets,
            experimental_options={},
        ),
    )
    @clustered_if(
        _multi_node,
        miles.total_nodes,
        gpu_type=miles.gpu_type,
    )
    async def train(
        modal_app_id: str = "",
        modal_app_url: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        cluster, modal_app_id, modal_app_url = await start_training_cluster(
            miles,
            (hf_cache_volume, data_volume, checkpoints_volume),
            "MILES_HOST_IP",
            "SGLANG_HOST_IP",
            modal_app_id=modal_app_id,
            modal_app_url=modal_app_url,
            framework_status_url=framework_status_url,
            framework_status_token=framework_status_token,
        )

        prep_id = hashlib.sha1(training_run_id.encode("utf-8")).hexdigest()[:16]
        prep_marker = os.path.join(
            checkpoints_mount_path, f".training_gym_prepared_{prep_id}"
        )
        prep_error = f"{prep_marker}.error"

        run_record: TrainingRun | None = None

        metric_entity = ""
        metric_run_id = ""

        if cluster.is_head:
            metric_entity = preflight_metric(miles.metrics)

            print(f"Training run id: {training_run_id}")
            (
                run_record,
                metric_run_id,
                framework_status_token,
            ) = await init_training_run_record(
                training_run_id=training_run_id,
                modal_app_id=modal_app_id,
                modal_app_url=modal_app_url,
                framework=Framework.MILES,
                initializing_status=MilesStatus.INITIALIZING,
                recipe=miles,
                model=model,
                dataset=dataset,
                eval_dataset=eval_dataset,
                dataset_path=dataset_path,
                eval_dataset_path=eval_dataset_path,
                recipe_metadata=("gpu_type",),
                metric_entity=metric_entity,
                framework_status_token=framework_status_token,
                checkpoint_dir=recorded_checkpoint_dir,
                checkpoints_volume_name=checkpoints_volume_name,
                checkpoints_mount_path=checkpoints_mount_path,
            )

        async def _prepare_shared_inputs() -> None:
            await set_status(MilesStatus.DOWNLOAD_MODEL)
            if model:
                model.download()
                if hasattr(model, "prepare_runtime_cache"):
                    model.prepare_runtime_cache()

            miles.download_model()
            materialize_model_remote_code()
            await set_status(MilesStatus.CONVERT_MODEL)
            miles.post_process_model()
            await hf_cache_volume.commit.aio()
            await checkpoints_volume.commit.aio()

            await set_status(MilesStatus.PREPARE_DATASET)
            if write_datasets(dataset, eval_dataset, dataset_path, eval_dataset_path):
                await data_volume.commit.aio()

        if cluster.is_head:
            try:
                async with shared.training_run_lifecycle(
                    run_record, framework_status_token
                ) as set_status:
                    await _prepare_shared_inputs()
            except BaseException as exc:
                os.makedirs(os.path.dirname(prep_error), exist_ok=True)
                with open(prep_error, "w") as f:
                    f.write(repr(exc))
                await checkpoints_volume.commit.aio()
                raise
            os.makedirs(os.path.dirname(prep_marker), exist_ok=True)
            with open(prep_marker, "w") as f:
                f.write(str(time.time()))
            await checkpoints_volume.commit.aio()
        else:
            deadline = time.time() + download_timeout + dataset_timeout
            while True:
                await asyncio.gather(
                    hf_cache_volume.reload.aio(),
                    data_volume.reload.aio(),
                    checkpoints_volume.reload.aio(),
                )
                if os.path.exists(prep_marker):
                    break
                if os.path.exists(prep_error):
                    with open(prep_error) as f:
                        raise RuntimeError(f"Head preparation failed: {f.read()}")
                if time.time() > deadline:
                    raise RuntimeError("Timed out waiting for head preparation marker")
                await asyncio.sleep(5)

        if model and model.model_name:
            # Warm each node's cache before Ray imports remote code concurrently.
            prewarm_remote_code(
                resolve_checkpoint_ref(model.model_path or model.model_name),
                environment,
                required=miles.model_name == "kimi_k3",
            )

        from modal_dojo.frameworks.miles.kernel_cache import prepare_kernel_cache

        runtime_environment, kernel_cache = await asyncio.to_thread(
            prepare_kernel_cache,
            environment,
            node_rank=cluster.rank,
            gpu_type=miles.gpu_type,
        )
        cluster.start_ray()

        if not cluster.is_head:
            try:
                await cluster.wait_forever()
            finally:
                if kernel_cache is not None:
                    await asyncio.to_thread(kernel_cache.close)
            return
        assert run_record is not None

        async with shared.training_run_lifecycle(
            run_record, framework_status_token
        ) as set_status:
            save_root = checkpoint_dir
            apply_scoped_save(miles, save_root)
            prepare_miles_config(miles, model, tempfile.mkdtemp())

            os.makedirs(save_root, exist_ok=True)

            resume_checkpoint = torch_dist_resume_checkpoint(
                save_root, is_complete=_is_resumable_checkpoint
            )
            record_resume_checkpoint(run_record, resume_checkpoint)
            await run_record.save(is_async=True)

            with shared.resumed_recipe(miles, save_root, resume_checkpoint):
                if resume_checkpoint is None and (
                    unresumable := _unresumable_save_dirs(save_root)
                ):
                    print(
                        f"WARNING: {save_root} holds saves of interrupted writes that cannot be resumed "
                        f"({', '.join(unresumable)}). Resuming into one of these would load a partial save."
                    )
                cmd = build_train_cmd(
                    miles,
                    MILES_ROOT,
                    model=model,
                    dataset=dataset,
                    eval_dataset=eval_dataset,
                    dataset_path=dataset_path,
                    eval_dataset_path=eval_dataset_path,
                )

            runtime_env = build_ray_runtime_env(
                head_addr=cluster.head_addr,
                metric_env=metric_runtime_env(
                    miles.metrics,
                    run_id=metric_run_id,
                    entity=metric_entity,
                ),
                environment=runtime_environment,
                substep_timing=miles.substep_timing,
                extra_env={
                    "MODAL_DOJO_TRAINING_RUN_ID": training_run_id,
                    "MODAL_DOJO_CHECKPOINTS_VOLUME_NAME": checkpoints_volume_name,
                    **shared.training_reporting_env(
                        miles, model, app_name, framework_status_url
                    ),
                },
                framework_status_token=framework_status_token,
            )

            mode = "async" if miles.train_async else "sync"
            print(
                f"Training {app_name} - {miles.total_nodes} node(s) x {gpu_spec} ({mode})"
            )
            print(miles.gpu_allocation.summary())
            print(f"Command: {cmd}")
            print(f"Runtime environment variables: {sorted(runtime_env['env_vars'])}")

            await set_status(MilesStatus.TRAINING)
            try:
                result = await cluster.submit_and_tail(cmd, runtime_env=runtime_env)
            finally:
                if kernel_cache is not None:
                    await asyncio.to_thread(kernel_cache.close)
            shared.check_training_result(result, run_record)
            print(f"Ray job message: {result.message}")

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
