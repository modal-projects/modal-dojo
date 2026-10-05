from __future__ import annotations

import warnings
from dataclasses import replace

import pytest

from modal_dojo.common.framework import Framework
from modal_dojo.common.memory_estimate import (
    _arch_from_hf,
    _peak_gib,
    maybe_warn_gpu_oom,
)
from modal_dojo.common.models.base import ModelArchitecture
from modal_dojo.common.models.qwen3_4b import Qwen3_4B
from modal_dojo.common.models.validation import VALIDATION_CONFIGS
from modal_dojo.train_recipes.miles_recipe import MilesRecipe
from modal_dojo.train_recipes.slime_recipe import SlimeRecipe


@pytest.mark.parametrize(
    "entry",
    [e for e in VALIDATION_CONFIGS if e.model_config.architecture is not None],
    ids=lambda e: e.name,
)
def test_presets_do_not_warn(entry) -> None:
    model = entry.model_config()
    recipe = (
        SlimeRecipe if entry.framework is Framework.SLIME else MilesRecipe
    ).get_base_recipe(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maybe_warn_gpu_oom(recipe, model)
    assert not [w for w in caught if "Estimated peak" in str(w.message)]


def test_known_mistake_warns() -> None:
    model, recipe = Qwen3_4B(), SlimeRecipe.get_base_recipe(Qwen3_4B())
    recipe.max_tokens_per_gpu = 65536
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maybe_warn_gpu_oom(recipe, model)
    msg = next(str(w.message) for w in caught if "Estimated peak" in str(w.message))
    assert "max_tokens_per_gpu=65536" in msg
    assert "recompute_granularity" not in msg
    assert "tensor_model_parallel_size" not in msg


def test_invalid_model_name_without_arch_does_not_raise() -> None:
    model = Qwen3_4B()
    model.architecture, model.model_name = None, ""
    maybe_warn_gpu_oom(SlimeRecipe.get_base_recipe(Qwen3_4B()), model)


def test_hf_cfg_detects_moe(tmp_path, monkeypatch) -> None:
    p = tmp_path / "config.json"
    p.write_text(
        '{"num_hidden_layers":4,"hidden_size":256,"num_attention_heads":4,'
        '"intermediate_size":512,"num_experts":8,"moe_intermediate_size":128,'
        '"vocab_size":1000}'
    )
    monkeypatch.setattr(
        "modal_dojo.common.memory_estimate.hf_hub_download",
        lambda **_: str(p),
    )
    assert _arch_from_hf("org/moe").num_experts == 8


def test_hf_bad_json_returns_none(tmp_path, monkeypatch) -> None:
    p = tmp_path / "config.json"
    p.write_text("not-json")
    monkeypatch.setattr(
        "modal_dojo.common.memory_estimate.hf_hub_download",
        lambda **_: str(p),
    )
    assert _arch_from_hf("org/bad") is None


def test_periodic_moe_freq_matches_explicit_pattern() -> None:
    base = ModelArchitecture(
        num_layers=12,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        num_experts=8,
        moe_ffn_hidden_size=128,
        vocab_size=1000,
    )
    knobs = {
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 1,
        "use_dynamic_batch_size": True,
        "max_tokens_per_gpu": 1024,
        "rollout_max_response_len": 128,
        "recompute_granularity": "full",
    }
    peak_int, _ = _peak_gib(replace(base, moe_layer_freq="2"), knobs, 80.0)
    peak_list, _ = _peak_gib(replace(base, moe_layer_freq="[1,0]*6"), knobs, 80.0)
    assert peak_int == peak_list


def _qwen3_5_knobs(**overrides) -> dict[str, object]:
    return {
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 1,
        "use_distributed_optimizer": True,
        "use_dynamic_batch_size": True,
        "max_tokens_per_gpu": 32768,
        "rollout_max_response_len": 32768,
        "recompute_granularity": "full",
        "optimizer_cpu_offload": True,
        **overrides,
    }


_QWEN3_5_4B = ModelArchitecture(
    num_layers=32,
    hidden_size=2560,
    ffn_hidden_size=9216,
    num_attention_heads=16,
    num_query_groups=4,
    kv_channels=256,
    vocab_size=248320,
)


def test_logits_backward_counted() -> None:
    # 32k tokens x 248k vocab OOMed in the logits backward on an 80 GB H100.
    peak, _ = _peak_gib(_QWEN3_5_4B, _qwen3_5_knobs(), 79.2)
    assert peak > 79.2


def test_lora_recipe_estimated() -> None:
    full, _ = _peak_gib(_QWEN3_5_4B, _qwen3_5_knobs(), 79.2)
    lora, raised = _peak_gib(_QWEN3_5_4B, _qwen3_5_knobs(lora_rank=16), 79.2)
    assert 0 < lora < full
    assert "optimizer_cpu_offload" not in raised


def test_multi_turn_without_cap_warns() -> None:
    model, recipe = Qwen3_4B(), SlimeRecipe.get_base_recipe(Qwen3_4B())
    recipe.extra_config = {
        "custom_generate_function_path": "miles.rollout.generate_hub.agentic_tool_call.generate"
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maybe_warn_gpu_oom(recipe, model)
    assert any("max_seq_len" in str(w.message) for w in caught)


def test_max_seq_len_capped_multi_turn_does_not_warn() -> None:
    model, recipe = Qwen3_4B(), SlimeRecipe.get_base_recipe(Qwen3_4B())
    recipe.extra_config = {
        "custom_generate_function_path": "miles.rollout.generate_hub.agentic_tool_call.generate",
        "max_seq_len": 16384,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maybe_warn_gpu_oom(recipe, model)
    assert not [w for w in caught if "lower bound" in str(w.message)]


def test_context_len_alone_does_not_cap_agentic() -> None:
    # rollout_max_context_len bounds each request, not the assembled session.
    model, recipe = Qwen3_4B(), SlimeRecipe.get_base_recipe(Qwen3_4B())
    recipe.extra_config = {
        "custom_generate_function_path": "miles.rollout.generate_hub.agentic_tool_call.generate",
        "rollout_max_context_len": 16384,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maybe_warn_gpu_oom(recipe, model)
    assert any("lower bound" in str(w.message) for w in caught)


def test_agentic_rl_path_warns() -> None:
    model, recipe = Qwen3_4B(), SlimeRecipe.get_base_recipe(Qwen3_4B())
    recipe.extra_config = {
        "custom_generate_function_path": "agentic_rl.generate.generate"
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maybe_warn_gpu_oom(recipe, model)
    assert any("lower bound" in str(w.message) for w in caught)


def test_max_seq_len_bounds_sample() -> None:
    knobs = {
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 1,
        "use_dynamic_batch_size": False,
        "micro_batch_size": 1,
        "rollout_max_prompt_len": 512,
        "rollout_max_response_len": 128,
        "max_seq_len": 1024,
        "recompute_granularity": "full",
        "custom_generate_function_path": "agentic_rl.generate.generate",
    }
    bounded, raised = _peak_gib(_QWEN3_5_4B, knobs, 80.0)
    uncapped, _ = _peak_gib(_QWEN3_5_4B, {**knobs, "max_seq_len": None}, 80.0)
    assert bounded > uncapped
    assert raised.get("max_seq_len") == 1024


def test_max_seq_len_ignored_for_single_turn() -> None:
    knobs = {
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 1,
        "use_dynamic_batch_size": False,
        "micro_batch_size": 1,
        "rollout_max_prompt_len": 512,
        "rollout_max_response_len": 128,
        "max_seq_len": 16,  # smaller than prompt+response; not enforced
        "recompute_granularity": "full",
    }
    peak, raised = _peak_gib(_QWEN3_5_4B, knobs, 80.0)
    assert "max_seq_len" not in raised
    assert raised.get("rollout_max_prompt_len") == 512


def test_multi_turn_path_warns_despite_max_seq_len() -> None:
    # miles multi_turn bounds each request, not the assembled session.
    model, recipe = Qwen3_4B(), SlimeRecipe.get_base_recipe(Qwen3_4B())
    recipe.extra_config = {
        "custom_generate_function_path": "miles.rollout.generate_hub.multi_turn.generate",
        "max_seq_len": 16384,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maybe_warn_gpu_oom(recipe, model)
    assert any("lower bound" in str(w.message) for w in caught)


def test_multi_turn_cap_scales_with_turns() -> None:
    knobs = {
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 1,
        "use_dynamic_batch_size": False,
        "micro_batch_size": 1,
        "rollout_max_context_len": 512,
        "generate_max_turns": 8,
        "recompute_granularity": "full",
        "custom_generate_function_path": "miles.rollout.generate_hub.multi_turn.generate",
    }
    with_turns, _ = _peak_gib(_QWEN3_5_4B, knobs, 80.0)
    one_turn, _ = _peak_gib(_QWEN3_5_4B, {**knobs, "generate_max_turns": 1}, 80.0)
    assert with_turns > one_turn


def test_single_turn_generate_path_does_not_warn() -> None:
    model, recipe = Qwen3_4B(), SlimeRecipe.get_base_recipe(Qwen3_4B())
    recipe.extra_config = {
        "custom_generate_function_path": "miles.rollout.generate_hub.single_turn.generate"
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maybe_warn_gpu_oom(recipe, model)
    assert not [w for w in caught if "lower bound" in str(w.message)]
