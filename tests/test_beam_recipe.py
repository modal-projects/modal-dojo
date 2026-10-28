"""Regression coverage for Beam recipe overrides and checkpoint safety."""

import json
from types import SimpleNamespace

import pytest

from modal_dojo import Beam_1_0_501B_A23B_LoRA_256k_Recipe as Beam
from modal_dojo.common.dataset import HuggingFaceDataset
from modal_dojo.common.launcher_helpers import ship_recipe_callables
from modal_dojo.frameworks.miles.modal_helpers import beam_checkpoint
from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_beam_activation as activation,
    patch_beam_cp as cp,
    patch_beam_lora_memory as memory,
)
from modal_dojo.train_recipes.miles_recipe.beam import beam_reward


@pytest.mark.parametrize("length", [8, 24, 33013, 262144])
def test_padding_preserves_local_sequence_parallel_alignment(length):
    recipe = Beam()
    multiple = recipe.tensor_model_parallel_size * recipe.data_pad_size_multiplier
    padded = (length + multiple - 1) // multiple * multiple
    assert (
        padded // recipe.context_parallel_size % recipe.tensor_model_parallel_size == 0
    )


@pytest.mark.parametrize("extra", [False, True])
def test_rejects_unaligned_padding_before_launch(extra):
    options = {"data_pad_size_multiplier": 1}
    with pytest.raises(ValueError, match="alignment"):
        Beam(**({"extra_config": options} if extra else options))


def test_custom_additions_preserve_required_image_and_environment():
    baseline = Beam()
    recipe = Beam(
        patch_files=["/tmp/user_patch.py"],
        image_run_commands=["python /tmp/user_patch.py"],
        environment={"CUSTOM": "yes", "NCCL_MIN_NCHANNELS": "16"},
        train_env_vars='{"CUSTOM": "train"}',
    )
    assert recipe.patch_files == [*baseline.patch_files, "/tmp/user_patch.py"]
    assert recipe.image_run_commands == [
        *baseline.image_run_commands,
        "python /tmp/user_patch.py",
    ]
    assert recipe.environment["MILES_REINSTALL_RDMA"] == "0"
    assert recipe.environment["NCCL_MIN_NCHANNELS"] == "16"
    assert recipe.train_env_vars == {**baseline.train_env_vars, "CUSTOM": "train"}
    rebuilt = Beam(**recipe.__dict__)
    assert rebuilt.patch_files == recipe.patch_files
    assert rebuilt.image_run_commands == recipe.image_run_commands


def test_extra_config_keeps_allocator_default():
    recipe = Beam(extra_config={"train_env_vars": {"CUSTOM": "train"}})
    assert (
        recipe.extra_config["train_env_vars"]["PYTORCH_ALLOC_CONF"]
        == "expandable_segments:True"
    )


def test_custom_reward_path_survives_callable_shipping():
    recipe = Beam(extra_config={"custom_rm_path": "customer.reward.score"})
    sentinel_image = object()
    assert (
        ship_recipe_callables(
            sentinel_image,
            recipe,
            caller_script=None,
            reward_post_process_in_config=False,
        )
        is sentinel_image
    )
    assert recipe.extra_config["custom_rm_path"] == "customer.reward.score"
    assert recipe.custom_rm_function is None
    assert recipe.reward_key is None
    assert recipe.eval_reward_key is None


def test_custom_scalar_reward_does_not_inherit_dict_keys():
    def scalar(*args):
        return 1.0

    recipe = Beam(custom_rm_function=scalar)
    assert recipe.custom_rm_function is scalar
    assert recipe.reward_key is recipe.eval_reward_key is None
    assert Beam(custom_rm_function=scalar, reward_key="quality").reward_key == "quality"
    assert Beam().custom_rm_function is beam_reward
    assert Beam().reward_key == "score"


def test_eval_requires_separate_data_and_is_enabled_when_supplied():
    dataset = HuggingFaceDataset(
        "zhuzilin/dapo-math-17k", input_column="prompt", output_column="label"
    )
    recipe = Beam()
    assert "--eval-interval" not in recipe.cli_args()
    args = recipe.cli_args(
        dataset=dataset,
        eval_dataset=dataset,
        dataset_path="/train",
        eval_dataset_path="/eval",
    )
    assert args[args.index("--eval-interval") + 1] == "1"
    assert args[
        args.index("--eval-prompt-data") + 1 : args.index("--eval-prompt-data") + 3
    ] == ["eval", "/eval"]
    assert "--skip-eval-before-train" not in args
    assert "--eval-interval" not in Beam(eval_interval=None).cli_args(
        dataset=dataset, eval_dataset=dataset
    )


def test_scheduler_uses_runtime_horizon_unless_explicitly_overridden():
    args = Beam(num_rollout=3).cli_args()
    assert "--lr-decay-iters" not in args
    assert "--lr-warmup-iters" not in args
    assert args[args.index("--lr-warmup-fraction") + 1] == "0.025"
    assert Beam(lr_warmup_iters=2).lr_warmup_fraction is None
    assert Beam(lr_decay_iters=1000).lr_decay_iters == 1000


def test_rollout_defaults_allow_capacity_based_scheduling_and_agent_parsing():
    args = Beam().cli_args()
    assert "--sglang-max-running-requests" not in args
    assert args[args.index("--sglang-attention-backend") + 1] == "trtllm_mha"
    for flag in ("--sglang-tool-call-parser", "--sglang-reasoning-parser"):
        assert args[args.index(flag) + 1] == "beam"
    with pytest.raises(ValueError, match="trtllm_mha"):
        Beam(extra_config={"sglang_attention_backend": "triton"})


@pytest.mark.parametrize(
    "mode,actor,rollout", [("debug_train_only", 16, 0), ("debug_rollout_only", 0, 16)]
)
def test_debug_allocation_matches_miles_placement(mode, actor, rollout):
    allocation = Beam(extra_config={mode: True}).gpu_allocation
    assert (allocation.actor_gpus, allocation.rollout_gpus) == (actor, rollout)
    assert allocation.total_gpus == 16
    assert allocation.total_nodes == 2


def test_debug_train_with_eval_pool_counts_both_pools():
    assert (
        Beam(
            extra_config={"debug_train_only": True, "eval_num_gpus": 16}
        ).gpu_allocation.total_gpus
        == 32
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        Beam(extra_config={"debug_train_only": True, "debug_rollout_only": True})


def test_refills_can_be_smaller_than_training_batch():
    assert Beam(over_sampling_batch_size=1).rollout_batch_size == 32
    assert Beam(extra_config={"over_sampling_batch_size": 1}).global_batch_size == 256
    with pytest.raises(ValueError, match="positive integer"):
        Beam(over_sampling_batch_size=0)


@pytest.mark.parametrize("comm", ["all_gather", ["all_gather"], ("all_gather",)])
def test_cp_patch_propagates_effective_communication(comm):
    source = "def configure(provider, args):\n" + cp.ANCHOR
    patched = cp.patch_source(source)
    namespace = {}
    exec(compile(patched, "provider.py", "exec"), namespace)
    provider = SimpleNamespace()
    namespace["configure"](
        provider, SimpleNamespace(context_parallel_size=2, cp_comm_type=comm)
    )
    assert provider.cp_comm_type == "all_gather"
    assert cp.patch_source(patched) == patched
    with pytest.raises(RuntimeError):
        cp.patch_source("upstream changed")


@pytest.mark.parametrize(
    "patch,source",
    [
        (
            memory.patch_source,
            "def apply():\n    if True:\n" + memory.INPUT + memory.SCALE,
        ),
        (memory.patch_wrapper, "def apply():\n    if True:\n" + memory.WRAPPER),
        (
            activation.patch_source,
            "class TEGroupedMLP(MegatronModule):\n    def forward(self):\n"
            + activation.ANCHOR
            + "            return intermediate_parallel\n",
        ),
    ],
)
def test_memory_patches_compile_are_idempotent_and_fail_on_source_drift(patch, source):
    patched = patch(source)
    compile(patched, "patched.py", "exec")
    assert patch(patched) == patched
    with pytest.raises(RuntimeError):
        patch("upstream changed")


def test_incomplete_native_checkpoint_is_never_resumed(tmp_path):
    root = tmp_path / "iter_0000000"
    (root / "adapter").mkdir(parents=True)
    files = {}
    for rank in range(2):
        for prefix in ("adapter_megatron_rank", "training_state_rank"):
            name = f"adapter/{prefix}{rank}.pt"
            (root / name).write_bytes(b"saved state")
            files[name] = 11
    (root / beam_checkpoint.MARKER).write_text(
        json.dumps({"world_size": 2, "files": files})
    )
    assert beam_checkpoint.complete_checkpoint(str(root))
    (root / "adapter/training_state_rank1.pt").write_bytes(b"partial")
    assert not beam_checkpoint.complete_checkpoint(str(root))
    (root / "adapter/training_state_rank1.pt").unlink()
    assert not beam_checkpoint.complete_checkpoint(str(root))


@pytest.mark.parametrize("mode", ["debug_train_only", "debug_rollout_only"])
def test_dashboard_prices_the_debug_allocation_from_serialized_recipe(mode):
    from dataclasses import asdict
    from modal_dojo.common.derived_metrics import RunPricing

    pricing = RunPricing.from_recipe(
        asdict(Beam(extra_config={mode: True})), {"gpu_hour_cost_h200": 4.0}
    )
    assert pricing.gpus == 16
    assert pricing.usd_per_s == pytest.approx(16 * 4 / 3600)


@pytest.mark.parametrize(
    "target,constant",
    [
        ("group.py", "_CELLS_READY_TIMEOUT_SECONDS"),
        ("inference_controller.py", "CELLS_READY_TIMEOUT_SECONDS"),
    ],
)
def test_startup_timeout_is_configurable_and_rejects_invalid_values(
    monkeypatch, target, constant
):
    from modal_dojo.frameworks.miles.modal_helpers.patches import (
        patch_beam_startup as startup,
    )

    source = startup.patch_source(f"{constant} = 3600.0", target)
    monkeypatch.setenv("MILES_CELL_READY_TIMEOUT_SECONDS", "10800")
    namespace = {}
    exec(compile(source, target, "exec"), namespace)
    assert namespace[constant] == 10800
    assert startup.patch_source(source, target) == source
    for invalid in ("0", "-1", "nan", "inf"):
        monkeypatch.setenv("MILES_CELL_READY_TIMEOUT_SECONDS", invalid)
        with pytest.raises(ValueError):
            exec(compile(source, target, "exec"), {})
    with pytest.raises(RuntimeError):
        startup.patch_source("upstream changed", target)


def test_user_patch_cannot_shadow_a_required_patch_in_remote_tmp():
    with pytest.raises(ValueError, match="Cannot replace required Beam patch"):
        Beam(patch_files=["/tmp/patch_beam_cp.py"])


@pytest.mark.parametrize("flag", ["offload_train", "offload"])
def test_offload_rejects_beams_current_allocator_variable(flag):
    with pytest.raises(ValueError, match="PYTORCH_ALLOC_CONF"):
        Beam(extra_config={flag: True})


def test_disabling_expandable_segments_allows_explicit_offload():
    recipe = Beam(
        extra_config={"offload_train": True},
        train_env_vars={"PYTORCH_ALLOC_CONF": "expandable_segments:False"},
    )
    assert recipe.extra_config["offload_train"] is True
