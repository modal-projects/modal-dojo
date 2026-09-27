"""Ray runtime env construction for miles jobs."""

from __future__ import annotations

import pytest

from modal_training_gym.frameworks.miles import launcher
from modal_training_gym.frameworks.miles.launcher import build_ray_runtime_env
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe


@pytest.fixture
def image_commands(monkeypatch):
    commands = []

    def run_commands(image, *args, **kwargs):
        commands.extend(args)
        return image

    monkeypatch.setattr(launcher.Image, "run_commands", run_commands)
    return commands


def test_multinode_image_reinstalls_matching_rdma_runtime(image_commands):
    recipe = MilesRecipe(colocate=False, actor_num_gpus_per_node=8, rollout_num_gpus=8)

    launcher._build_miles_base_image(recipe)

    assert launcher.RDMA_RUNTIME_INSTALL_COMMAND in image_commands


def test_colocate_multinode_skips_rdma_reinstall(image_commands):
    recipe = MilesRecipe(colocate=True, actor_num_nodes=2, actor_num_gpus_per_node=8)

    launcher._build_miles_base_image(recipe)

    assert launcher.RDMA_RUNTIME_INSTALL_COMMAND not in image_commands


def test_single_node_image_keeps_base_rdma_runtime(image_commands):
    launcher._build_miles_base_image(MilesRecipe())

    assert launcher.RDMA_RUNTIME_INSTALL_COMMAND not in image_commands


def test_ld_library_path_comes_from_the_container(monkeypatch):
    """Workers get the container's linker path, behind the system lib dir."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64:/wheel/nvidia/lib")

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1", metric_env={}, environment={}
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == (
        "/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/wheel/nvidia/lib"
    )
    assert env_vars["MASTER_ADDR"] == "10.0.0.1"
    assert env_vars["no_proxy"] == "127.0.0.1,10.0.0.1"
    assert "MASTER_PORT" not in env_vars


def test_efa_dirs_move_ahead_of_the_system_lib_dir(monkeypatch):
    """Host-mounted libfabric/ofi-nccl must shadow the image's libfabric."""
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        "/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:/usr/local/cuda/lib64"
        ":/wheel/nvidia/lib",
    )

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1", metric_env={}, environment={}
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == (
        "/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:/usr/lib/x86_64-linux-gnu"
        ":/usr/local/cuda/lib64:/wheel/nvidia/lib"
    )


def test_efa_dirs_keep_relative_order_when_not_leading(monkeypatch):
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        "/usr/local/cuda/lib64:/opt/amazon/ofi-nccl/lib:/usr/lib/x86_64-linux-gnu"
        ":/opt/amazon/efa/lib:/opt/amazon/efa/lib",
    )

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1", metric_env={}, environment={}
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == (
        "/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/usr/lib/x86_64-linux-gnu"
        ":/usr/local/cuda/lib64"
    )


def test_recipe_environment_still_wins(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/from/container")

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1",
        metric_env={},
        environment={
            "LD_LIBRARY_PATH": "/from/recipe",
            "PYTHONPATH": "/root/Megatron-LM/",
        },
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == "/from/recipe"
    assert env_vars["PYTHONPATH"] == "/root/Megatron-LM/"


def test_unset_container_path_yields_only_the_system_lib_dir(monkeypatch):
    """No empty entry, which the loader would read as the working directory."""
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1", metric_env={}, environment={}
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == "/usr/lib/x86_64-linux-gnu"


def test_system_lib_dir_is_not_duplicated(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib/x86_64-linux-gnu:/wheel/nvidia/lib")

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1", metric_env={}, environment={}
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == (
        "/usr/lib/x86_64-linux-gnu:/wheel/nvidia/lib"
    )


def test_metric_env_is_preserved(monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1",
        metric_env={
            "WANDB_RUN_ID": "abc",
            "WANDB_RESUME": "allow",
            "WANDB_API_KEY": "validated-key",
        },
        environment={"WANDB_RUN_ID": "other-run", "WANDB_API_KEY": "other-key"},
    )["env_vars"]

    assert env_vars["WANDB_RUN_ID"] == "abc"
    assert env_vars["WANDB_RESUME"] == "allow"
    assert env_vars["WANDB_API_KEY"] == "validated-key"
