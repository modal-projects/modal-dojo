"""Helper functions for Modal multi-node training infrastructure.

Shared implementations live in :mod:`modal_dojo.common.launcher_utils`;
this module keeps slime's parametrization and the historical import path.
"""

from modal_dojo.common.launcher_utils import (
    build_train_cmd as _build_train_cmd,
    get_checkpoint_conversion_policy as _get_checkpoint_conversion_policy,
    is_local_checkpoint_ref as is_local_checkpoint_ref,
    prepare_launch_config as _prepare_launch_config,
    resolve_checkpoint_ref as resolve_checkpoint_ref,
)


def get_checkpoint_conversion_policy(
    slime_cfg, model=None
) -> tuple[int, int, list[str]]:
    """Return (num_nodes, nproc_per_node, extra_args) for checkpoint conversion."""
    return _get_checkpoint_conversion_policy(
        slime_cfg,
        model=model,
        single_rank_mtp=True,
        extended_arch_args=True,
        arch_args_model_script_attr="slime_model_script",
    )


def get_modal_cluster_context(n_nodes: int) -> tuple[int, str, str, int]:
    """Return (rank, master_addr, my_ip, n_nodes) for the current Modal cluster."""
    if n_nodes == 1:
        return 0, "127.0.0.1", "127.0.0.1", 1

    import modal.experimental

    info = modal.experimental.get_cluster_info()
    actual_nodes = len(info.container_ipv4_ips)
    if actual_nodes != n_nodes:
        raise RuntimeError(
            f"cluster size mismatch: expected {n_nodes} node(s), got {actual_nodes}"
        )
    return (
        info.rank,
        info.container_ipv4_ips[0],
        info.container_ipv4_ips[info.rank],
        actual_nodes,
    )


def prepare_slime_config(slime_cfg, model, tmpdir: str) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs."""
    from modal_dojo.train_recipes.slime_recipe.recipe import YAML_CONFIG_FIELDS

    _prepare_launch_config(
        slime_cfg, model, tmpdir, yaml_config_fields=YAML_CONFIG_FIELDS
    )


def build_train_cmd(
    slime_cfg,
    slime_root: str,
    model=None,
    dataset=None,
    eval_dataset=None,
    dataset_path=None,
    eval_dataset_path=None,
) -> str:
    """Build the Ray job entrypoint, sourcing model arch args if needed."""
    return _build_train_cmd(
        slime_cfg,
        slime_root,
        model=model,
        dataset=dataset,
        eval_dataset=eval_dataset,
        dataset_path=dataset_path,
        eval_dataset_path=eval_dataset_path,
        model_script_attr="slime_model_script",
    )
