"""CPU contracts against the pinned training and serving sources (no model download)."""

import asyncio
import ast
import importlib.util
import re
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip("miles.utils.dp_schedule")
torch = pytest.importorskip("torch")


@pytest.fixture
def native_recipe_model():
    from transformers import AutoConfig

    from miles.utils.hf_utils.config import register_hf_config_aliases
    from miles.utils.lora.arguments import _resolve_lora_targets
    from modal_dojo import DeepSeek_V4_1_Flash_LoRA_Recipe

    recipe = DeepSeek_V4_1_Flash_LoRA_Recipe()
    register_hf_config_aliases()
    targets, served = _resolve_lora_targets(
        SimpleNamespace(
            target_modules=recipe.target_modules,
            exclude_modules=recipe.exclude_modules,
            multi_lora=False,
            megatron_to_hf_mode="raw",
            lora_type="lora",
            experts_shared_outer_loras=False,
        ),
        AutoConfig.for_model("deepseek_v41"),
    )
    assert set(targets) == set(served) == set(recipe.target_modules.split(","))
    # Use upstream's small architectural fixture, with the recipe's actual targets.
    path = "/root/miles/tests/fast/backends/megatron_utils/lora/native/test_deepseek_v4_1.py"
    spec = importlib.util.spec_from_file_location("v41_upstream_test", path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    return upstream, targets, upstream._model(targets)


def test_every_recipe_target_exports_and_is_recognized_by_serving(native_recipe_model):
    from miles.backends.megatron_utils.lora.native.hf_adapter import (
        export_lora_hf_named,
    )
    from miles.utils.lora.utils import get_adapter_target_modules

    _, targets, model = native_recipe_model
    exported = dict(export_lora_hf_named([model]))
    assert set(get_adapter_target_modules(exported)) == {
        target.replace("*", "0") for target in targets
    }
    # Read the actual SGLang predicate without importing its GPU-only kernels.
    path = Path("/root/sglang/python/sglang/srt/models/deepseek_v4.py")
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4ForCausalLM"
    )
    selected = [
        n
        for n in cls.body
        if (
            isinstance(n, ast.FunctionDef)
            and n.name in {"should_apply_lora", "get_lora_target_module_name"}
        )
        or (
            isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_lora_module_pattern"
                for t in n.targets
            )
        )
    ]
    cls.bases, cls.decorator_list, cls.keywords, cls.body = [], [], [], selected
    namespace = {"re": re}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    serving = namespace[cls.name]()
    for target in targets:
        name = target.replace("layers.*.attn.", "model.layers.0.self_attn.")
        if ".ffn.shared_experts." in target:
            leaf = serving.get_lora_target_module_name(target)
            name = "model.layers.0.mlp.shared_experts." + (
                "gate_up_proj" if leaf in {"gate_proj", "up_proj"} else leaf
            )
        assert serving.should_apply_lora(name), name
    for name in (
        "model.layers.0.self_attn.indexer.wq_b",
        "model.layers.0.self_attn.compressor.wkv",
        "model.layers.0.mlp.experts.0.down_proj",
        "model.layers.0.engram.wkv",
    ):
        assert not serving.should_apply_lora(name)


def test_shared_expert_export_preserves_outputs_and_gradients(native_recipe_model):
    from miles.backends.megatron_utils.lora.native.hf_adapter import (
        export_lora_hf_named,
    )

    upstream, _, model = native_recipe_model
    exported = dict(export_lora_hf_named([model]))
    shared = model.layers[0].mlp.shared_experts

    def delta(leaf):
        prefix = f"layers.0.ffn.shared_experts.{leaf}"
        return (
            upstream.ALPHA
            / upstream.RANK
            * (
                exported[prefix + ".lora_B.weight"].float()
                @ exported[prefix + ".lora_A.weight"].float()
            )
        )

    x = torch.randn(2, 3, upstream.HIDDEN)
    actual, _ = shared.linear_fc1(x)
    merged = shared.linear_fc1.weight + torch.cat([delta("w1"), delta("w3")])
    torch.testing.assert_close(actual, x @ merged.t())
    gate, up = actual.chunk(2, dim=-1)
    hidden = torch.nn.functional.silu(gate) * up
    actual, _ = shared.linear_fc2(hidden)
    torch.testing.assert_close(
        actual, hidden @ (shared.linear_fc2.weight + delta("w2")).t()
    )
    actual.square().mean().backward()
    adapters = [p for name, p in shared.named_parameters() if "lora" in name]
    assert len(adapters) == 6
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in adapters
    )


def args():
    return SimpleNamespace(
        model_name="deepseekv41",
        lora_rank=32,
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
        _post_launch_release=None,
        _change_state=lambda *a, **kw: transitions.append(a[0]),
    )
    cell._release_memory_after_launch = MethodType(
        server_cell.ServerCell._release_memory_after_launch, cell
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
