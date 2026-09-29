"""Miles launcher helpers.

Shared implementations live in :mod:`modal_dojo.common.launcher_utils`;
this module keeps miles' parametrization and the historical import path.
"""

import shlex

from modal_dojo.common.launcher_utils import (
    build_train_cmd as _build_train_cmd,
    get_checkpoint_conversion_policy as _get_checkpoint_conversion_policy,
    is_local_checkpoint_ref as is_local_checkpoint_ref,
    prepare_launch_config as _prepare_launch_config,
    resolve_checkpoint_ref as resolve_checkpoint_ref,
)


def get_checkpoint_conversion_policy(
    miles_cfg, model=None
) -> tuple[int, int, list[str]]:
    return _get_checkpoint_conversion_policy(
        miles_cfg,
        model=model,
        single_rank_mtp=False,
        extended_arch_args=not bool(miles_cfg.miles_model_script),
        arch_args_model_script_attr="miles_model_name",
    )


def prepare_miles_config(miles_cfg, model, tmpdir: str) -> None:
    from modal_dojo.train_recipes.miles_recipe.recipe import YAML_CONFIG_FIELDS

    _prepare_launch_config(
        miles_cfg, model, tmpdir, yaml_config_fields=YAML_CONFIG_FIELDS
    )


def model_args_command(miles_cfg, miles_root: str) -> str:
    if not miles_cfg.miles_model_name:
        return ""
    utility = f"{miles_root}/miles/utils/external_utils/model_args_utils.py"
    return shlex.join(["python3", utility, miles_cfg.miles_model_name])


def build_train_cmd(
    miles_cfg,
    miles_root: str,
    model=None,
    dataset=None,
    eval_dataset=None,
    dataset_path=None,
    eval_dataset_path=None,
) -> str:
    return _build_train_cmd(
        miles_cfg,
        miles_root,
        model=model,
        dataset=dataset,
        eval_dataset=eval_dataset,
        dataset_path=dataset_path,
        eval_dataset_path=eval_dataset_path,
        model_script_attr="miles_model_script",
        model_args_command=model_args_command(miles_cfg, miles_root),
    )
