"""CPU checks for the candidate Muse Spark recipe; these do not prove 64K fits."""

from modal_dojo import (
    MilesRecipe,
    Muse_Spark_Open,
    Muse_Spark_Open_64k_B300_Recipe,
    Muse_Spark_Open_64k_Recipe,
)
from modal_dojo.common.models.validation import _ValidationConfig
from modal_dojo.frameworks.miles.modal_helpers.utils import (
    build_train_cmd,
    get_checkpoint_conversion_policy,
)


def test_registry_and_conversion_layout():
    model = _ValidationConfig.find("Muse-Spark-Open").model_config()
    assert isinstance(model, Muse_Spark_Open)
    recipe = MilesRecipe.get_base_recipe(model)
    assert isinstance(recipe, Muse_Spark_Open_64k_Recipe)
    assert recipe.gpu_allocation.total_gpus == 64
    nodes, ranks, flags = get_checkpoint_conversion_policy(recipe, model)
    assert (nodes, ranks) == (4, 8)
    assert "--tensor-model-parallel-size 1" in flags
    assert "--pipeline-model-parallel-size 4" in flags
    assert "--expert-model-parallel-size 8" in flags
    assert "--decoder-first-pipeline-num-layers 15" in flags


def test_long_context_offload_and_observability_reach_miles():
    recipe = Muse_Spark_Open_64k_Recipe()
    cmd = build_train_cmd(recipe, "/root/miles", model=Muse_Spark_Open())
    for expected in (
        "model_args_utils.py muse-spark-full",
        "--seq-length 65536",
        "--sglang-context-length 65536",
        "--rollout-max-response-len 61440",
        "--context-parallel-size 4",
        "--expert-model-parallel-size 16",
        "--offload-train ",
        "--offload-rollout ",
        "--offload-train-target disk",
        "--stream-optimizer-state-to-disk",
        "--sglang-kv-cache-dtype bfloat16",
        "modal_dojo.frameworks.miles.phase_reporting.before_train_step_hook",
    ):
        assert expected in cmd
    assert recipe.rollout_max_prompt_len + recipe.rollout_max_response_len == 65536
    assert recipe.save is None
    assert recipe.num_rollout == 1


def test_b300_layout_conversion_and_long_context_command():
    recipe = Muse_Spark_Open_64k_B300_Recipe()
    model = Muse_Spark_Open()
    assert recipe.gpu_allocation.total_gpus == 32
    assert recipe.gpu_type == "B300"
    assert recipe.train_env_vars == {"DOJO_MUSE_TE_FA4": "0"}
    assert Muse_Spark_Open_64k_Recipe().train_env_vars is None
    nodes, ranks, flags = get_checkpoint_conversion_policy(recipe, model)
    assert (nodes, ranks) == (4, 8)
    assert "--tensor-model-parallel-size 1" in flags
    assert "--pipeline-model-parallel-size 4" in flags
    assert "--expert-model-parallel-size 8" in flags
    assert "--decoder-first-pipeline-num-layers 15" in flags
    cmd = build_train_cmd(recipe, "/root/miles", model=model)
    for expected in (
        "--tensor-model-parallel-size 8",
        "--pipeline-model-parallel-size 4",
        "--expert-model-parallel-size 8",
        "--rollout-num-gpus-per-engine 8",
        "--sglang-ep-size 8",
        "--pin-rollout-manager-to-head",
        "--seq-length 65536",
        "--sglang-context-length 65536",
        "--rollout-max-response-len 61440",
        "--sglang-max-running-requests 4",
        "--stream-optimizer-state-to-disk",
        "modal_dojo.frameworks.miles.phase_reporting.before_train_step_hook",
    ):
        assert expected in cmd
    assert "--cp-comm-type" not in cmd
    assert recipe.context_parallel_size == 1
    assert recipe.train_function_kwargs["ephemeral_disk"] == 3 * 1024 * 1024
    assert recipe.num_rollout == 1
    # An explicit hardware variant must not change the default H200 recipe.
    assert type(MilesRecipe.get_base_recipe(model)) is Muse_Spark_Open_64k_Recipe
