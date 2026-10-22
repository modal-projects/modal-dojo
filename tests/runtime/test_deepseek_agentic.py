"""CPU tests against the pinned native Miles image; no GPUs or Tau package required."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("miles.utils.dp_schedule")
torch = pytest.importorskip("torch")

from modal_dojo.frameworks.miles.lora_diagnostics import _layers, effective_squares  # noqa: E402


def args():
    return SimpleNamespace(
        model_name="deepseekv41",
        use_dynamic_batch_size=False,
        micro_batch_size=1,
        qkv_format="bshd",
        actor_num_nodes=4,
        actor_num_gpus_per_node=8,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        global_batch_size=32,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        reward_key=None,
    )


@pytest.mark.parametrize(
    ("model_name", "lora_rank", "train_only", "weight_backups"),
    [
        ("deepseekv41", 32, False, 0),
        ("deepseekv41", 0, False, 1),
        ("deepseekv41", 32, True, 1),
        ("glm5", 32, False, 1),
    ],
)
def test_initial_offload_preserves_resident_regions_without_lora_weight_roundtrip(
    monkeypatch,
    model_name,
    lora_rank,
    train_only,
    weight_backups,
):
    from miles.ray.rollout import server_cell
    from sglang.srt.constants import (
        GPU_MEMORY_TYPE_CUDA_GRAPH,
        GPU_MEMORY_TYPE_KV_CACHE,
        GPU_MEMORY_TYPE_WEIGHTS,
    )

    all_regions = {
        GPU_MEMORY_TYPE_CUDA_GRAPH,
        GPU_MEMORY_TYPE_KV_CACHE,
        GPU_MEMORY_TYPE_WEIGHTS,
    }
    resident = set(all_regions)
    backups, transitions = [], []

    class API:
        def __init__(self, **kwargs):
            pass

        async def release_memory_occupation(self, tags=None):
            tags = set(tags) if tags is not None else all_regions
            assert tags <= resident
            if GPU_MEMORY_TYPE_WEIGHTS in tags:
                backups.append("weights")
            resident.difference_update(tags)

        async def resume_memory_occupation(self, tags):
            assert not resident.intersection(tags)
            resident.update(tags)

    async def healthy(**kwargs):
        return True

    monkeypatch.setattr(server_cell, "SGLangApiClient", API)
    monkeypatch.setattr(server_cell, "probe_server_healthy", healthy)
    cell = SimpleNamespace(
        args=SimpleNamespace(
            model_name=model_name,
            lora_rank=lora_rank,
            lora_train_only=train_only,
            check_weight_update_equal=False,
            debug_rollout_only=False,
        ),
        meta=SimpleNamespace(
            needs_offload=True, update_weights=True, sglang_api_key=None
        ),
        _state=SimpleNamespace(
            addr_info=server_cell.CellAddrInfo(
                server_url="http://test",
                bootstrap_port=None,
                gate_url=None,
            )
        ),
        _change_state=lambda *a, **kw: transitions.append(a[0]),
    )
    asyncio.run(server_cell.ServerCell._tick_when_initializing(cell))
    # Same serving state as release-all followed by restore-weights; the normal
    # training transition must still be able to back up and restore all regions.
    assert resident == {GPU_MEMORY_TYPE_WEIGHTS}
    assert len(backups) == weight_backups
    assert transitions == ["mark_pending_weights"]
    api = API()
    asyncio.run(
        api.resume_memory_occupation(
            [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]
        )
    )
    asyncio.run(api.release_memory_occupation())
    assert not resident and len(backups) == weight_backups + 1
    asyncio.run(api.resume_memory_occupation(all_regions))
    assert resident == all_regions


def test_native_microbatch_padding_preserves_tokens_masks_and_replay(monkeypatch):
    import copy

    import numpy as np
    from miles.backends.training_utils.data import context_parallel, rollout
    from miles.backends.training_utils.replay import data as replay

    parallel = SimpleNamespace(
        tp=SimpleNamespace(rank=0, size=4),
        cp=SimpleNamespace(rank=0, size=1),
        effective_dp=SimpleNamespace(rank=0),
    )
    for module in (rollout, context_parallel, replay):
        monkeypatch.setattr(module, "get_parallel_state", lambda: parallel)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    lengths = [2, 4101, 27023]
    source = {
        "tokens": [list(range(n)) for n in lengths],
        "loss_masks": [[0], [1, 0, 1], [1, 1, 1]],
        "response_lengths": [1, 3, 3],
        "total_lengths": lengths,
        "rollout_routed_experts": [
            np.broadcast_to(np.arange(6, dtype=np.int32) + 12, (n - 1, 1, 6)).copy()
            for n in lengths
        ],
    }
    monkeypatch.setattr(
        rollout, "process_rollout_data", lambda *a, **kw: (copy.deepcopy(source), None)
    )
    a = args()
    a.data_pad_size_multiplier, a.compress_ratios, a.enable_witness = (
        128,
        [0, 1, 2],
        False,
    )
    a.allgather_cp, a.sequence_parallel = False, True
    data, _ = rollout.get_rollout_data(a, None, train_parallel_config=None)
    assert data["max_seq_lens"] == [512, 4608, 27136]
    # Native fallback still covers multi-sample microbatches and other models.
    a.micro_batch_size = 2
    old, _ = rollout.get_rollout_data(a, None, train_parallel_config=None)
    assert old["max_seq_lens"] == [27136] * 3
    a.micro_batch_size = 1
    keys = ["tokens", "loss_masks", "total_lengths", "response_lengths", "max_seq_lens"]
    iterator = rollout.DataIterator(data, micro_batch_size=1)
    for i, length in enumerate(lengths):
        batch = rollout.get_batch(iterator, keys, qkv_format="bshd")
        assert batch["tokens"].shape == (1, data["max_seq_lens"][i])
        torch.testing.assert_close(
            batch["tokens"][0, :length], torch.tensor(source["tokens"][i])
        )
        assert not batch["tokens"][0, length:].any()
        assert batch["input_loss_masks"].sum() == sum(source["loss_masks"][i])
    captured = []
    replay.fill_replay_data(
        args=a,
        models=[],
        data_iterator=[iterator],
        num_microbatches=[3],
        rollout_data=data,
        data_key="rollout_routed_experts",
        replay_list=[],
        register_replay_list_func=lambda _, value, **kw: captured.append(value.clone()),
    )
    for i, tensor in enumerate(captured):
        assert tensor.shape == (data["max_seq_lens"][i] // 4, 1, 6)
        real = min(lengths[i] - 1, len(tensor))
        torch.testing.assert_close(
            tensor[:real], torch.from_numpy(source["rollout_routed_experts"][i][:real])
        )
        assert (tensor[real:] == -1).all()
    # A masked two-token stub no longer dispatches ~27k padding tokens through the MoE.
    assert data["max_seq_lens"][0] < old["max_seq_lens"][0] / 50


def test_native_linear_diagnostics_cover_factors_and_use_effective_weight_delta():
    from miles.backends.megatron_utils.lora.native.config import LoRAConfig
    from miles.backends.megatron_utils.lora.native.modules.linear import LoRALinear
    from miles.backends.megatron_utils.lora.native.spec.base import (
        ProjectionSpec,
        ShardLayout,
    )

    host = torch.nn.Linear(5, 7, bias=False)
    host.weight.requires_grad_(False)
    host.weight.tensor_model_parallel = False
    context = SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        sequence_parallel=False,
        lora=LoRAConfig(rank=2, alpha=2, dropout=0, target_modules=None),
    )
    projection = ProjectionSpec(
        hf_leaf="wq_a", param_prefix="lora", layout=ShardLayout.REPLICATED
    )
    adapter = LoRALinear(
        hf_prefix="layers.0.attn.",
        projection=projection,
        reference=host.weight,
        context=context,
        in_features=5,
        out_features=7,
    )
    adapter.bind_host(host)
    model = torch.nn.Module()
    model.host, model.adapter = host, adapter
    parallel = SimpleNamespace(
        effective_dp=SimpleNamespace(size=1), tp=SimpleNamespace(size=1)
    )
    rows = list(_layers([model], parallel))
    assert len(rows) == 1
    _, base, a, b, scale, replicas = rows[0]
    old_a, old_b = a.detach().clone(), b.detach().clone()
    with torch.no_grad():
        b.fill_(0.01)
    values = effective_squares(base, old_a, old_b, a, b, scale)
    expected = (
        ((b.double() @ a.double() - old_b.double() @ old_a.double()) * scale)
        .square()
        .sum()
    )
    torch.testing.assert_close(values[0], expected)
    assert values[0] > 0 and replicas == 1


def test_native_rng_round_trip(monkeypatch):
    from modal_dojo.frameworks.miles import native_lora_state as native_state

    # This CPU preflight has no initialized Megatron training process.
    monkeypatch.setattr(
        native_state, "_layout", lambda: {"tensor_model_parallel_size": 4}
    )
    state = native_state.capture_resume_state()
    expected = torch.rand(4)
    native_state.restore_resume_state(copy.deepcopy(state))
    torch.testing.assert_close(torch.rand(4), expected)


def test_native_checkpoint_load_keeps_optimizer_rng_and_scheduler_intent(
    monkeypatch, tmp_path
):
    from miles.backends.megatron_utils.lora import utils
    from miles.backends.megatron_utils.megatron_config import (
        resolve_args_checkpoint_load,
    )

    from modal_dojo.frameworks.miles import native_lora_state as native_state

    monkeypatch.setattr(
        native_state, "_layout", lambda: {"tensor_model_parallel_size": 4}
    )
    adapter = tmp_path / "iter_0000000" / "adapter"
    adapter.mkdir(parents=True)
    a = SimpleNamespace(
        load=str(tmp_path / "fresh-run"),
        ref_load="/converted-base",
        ref_ckpt_step=None,
        megatron_to_hf_mode="raw",
        lora_adapter_path=str(adapter),
        no_load_optim=False,
        no_load_rng=False,
    )
    resolve_args_checkpoint_load(a)
    assert a.load == "/converted-base" and a.no_load_optim and a.no_load_rng
    assert a.dojo_native_resume_flags == (False, False)
    native_state.prepare_native_resume(a)
    state = {
        "iteration": 0,
        "optimizer": {"step": 1},
        "parameter_state": [{"exp_avg": torch.tensor([0.25])}],
        "opt_param_scheduler": {"num_steps": 32},
        "tau_resume_state": native_state.capture_resume_state(),
    }
    expected_rng = torch.rand(4)
    restored = {}
    optimizer = SimpleNamespace(
        load_state_dict=lambda value: restored.update(optimizer=value)
    )
    part = SimpleNamespace(
        load_parameter_state_from_dp_reshardable=lambda value: restored.update(
            moments=value
        )
    )
    scheduler = SimpleNamespace(
        load_state_dict=lambda value: restored.update(scheduler=value)
    )
    monkeypatch.setattr(
        utils,
        "_native_adapter_shard",
        lambda directory: directory / "adapter_megatron_rank0.pt",
    )
    monkeypatch.setattr(utils, "_load_native_adapter_shard", lambda *args: None)
    monkeypatch.setattr(
        utils, "_load_training_state", lambda directory: copy.deepcopy(state)
    )
    monkeypatch.setattr(utils, "_distributed_optimizers", lambda optimizer: [part])
    iteration = utils.restore_lora_adapter(
        [], a, optimizer=optimizer, opt_param_scheduler=scheduler
    )
    assert iteration == 0 and restored["optimizer"] == {"step": 1}
    torch.testing.assert_close(restored["moments"]["exp_avg"], torch.tensor([0.25]))
    assert (
        restored["scheduler"] == {"num_steps": 32} and a.dojo_native_scheduler_restored
    )
    torch.testing.assert_close(torch.rand(4), expected_rng)


@pytest.mark.parametrize("restored_index", [1, 6])
def test_native_resume_coordinator_restores_sampler_and_runs_one_remaining_update(
    monkeypatch, restored_index
):
    from miles.ray import placement_group as pg

    # The real base-checkpoint resolver resets this to zero even when the
    # native adapter returns a later index. Exercise the coordinator, not just
    # the tensor loader: this is where sampler loading and loop bounds diverge.
    a = SimpleNamespace(
        start_rollout_id=0,
        lora_adapter_path=f"/parent/iter_{restored_index - 1:07d}/adapter",
        use_critic=False,
        num_rollout=restored_index + 1,
    )
    config = SimpleNamespace(role=pg.ACTOR_ROLE, trainer_id="actor")
    parallel = {"tensor_model_parallel_size": 4}
    handle = SimpleNamespace(get_train_parallel_config=AsyncMock(return_value=parallel))
    executor = SimpleNamespace(set_train_parallel_config=AsyncMock(), load=AsyncMock())
    monkeypatch.setattr(pg, "compute_trainer_configs", lambda args: [config])
    monkeypatch.setattr(
        pg, "create_trainer_handles", lambda *args, **kwargs: {"actor": handle}
    )
    monkeypatch.setattr(pg, "take_over_trainers", AsyncMock(return_value=False))
    monkeypatch.setattr(pg, "compute_trainer_args", lambda args, config: args)
    monkeypatch.setattr(
        pg, "trainer_init_or_load_state", AsyncMock(return_value=[restored_index] * 32)
    )

    actor, critic = asyncio.run(pg.create_training_models(a, executor))
    assert actor is handle and critic is None
    assert a.start_rollout_id == restored_index
    executor.load.assert_awaited_once_with(restored_index - 1)
    executor.set_train_parallel_config.assert_awaited_once_with(parallel)
    assert list(range(a.start_rollout_id, a.num_rollout)) == [restored_index]


def test_native_coordinator_keeps_explicit_start_without_adapter(monkeypatch):
    from miles.ray import placement_group as pg

    monkeypatch.setattr(
        pg, "trainer_init_or_load_state", AsyncMock(return_value=[0] * 32)
    )
    a = SimpleNamespace(start_rollout_id=3, lora_adapter_path=None)
    result = asyncio.run(
        pg.create_training_model(a, handle=object(), trainer_id="actor", resumed=False)
    )
    assert result.start_rollout_id == 3


def test_native_executor_loads_parent_sampler_instead_of_frozen_base(tmp_path):
    from miles.ray.rollout.rollout_executor import (
        RolloutExecutor,
        compute_rollout_checkpoint_dir,
    )
    from miles.rollout.data_source import RolloutDataSource
    from miles.utils.simple_checkpointer import save_simple_checkpoint

    parent = tmp_path / "parent"
    directory = compute_rollout_checkpoint_dir(parent, rollout_id=0)
    state = {
        "sample_offset": 8,
        "epoch_id": 0,
        "sample_group_index": 8,
        "sample_index": 32,
    }
    save_simple_checkpoint(directory=directory / "data_source", data=state)
    a = SimpleNamespace(
        load=str(tmp_path / "frozen-base"),
        lora_adapter_path=str(parent / "iter_0000000" / "adapter"),
        rollout_global_dataset=True,
        rollout_shuffle=False,
    )
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = a
    snapshotter = SimpleNamespace(load=Mock(), discard_completed=Mock())
    executor = SimpleNamespace(
        args=a,
        data_source=source,
        _output_snapshotter=snapshotter,
        use_legacy_rollout_v1=True,
    )
    asyncio.run(RolloutExecutor.load(executor, 0))
    assert {key: getattr(source, key) for key in state} == state
    snapshotter.load.assert_called_once_with(directory / "executor")
    snapshotter.discard_completed.assert_called_once_with(
        completed_rollout_ids={None: 0}
    )


def test_masked_fragments_do_not_dilute_the_token_weighted_gradient():
    from miles.backends.training_utils.loss.objective import count_loss_tokens

    real = [torch.tensor([1, 1]), torch.tensor([1, 0, 1])]
    padded = real + [torch.tensor([0])] * 7
    assert count_loss_tokens(padded) == count_loss_tokens(real) == 4
    assert count_loss_tokens([torch.tensor([0])]) == 0
    parameter = torch.tensor(0.3, requires_grad=True)
    loss = parameter.square() * 2 + (parameter * 3).square() * 2
    (expected,) = torch.autograd.grad(
        loss / count_loss_tokens(real), parameter, retain_graph=True
    )
    (observed,) = torch.autograd.grad(loss / count_loss_tokens(padded), parameter)
    torch.testing.assert_close(expected, observed)
