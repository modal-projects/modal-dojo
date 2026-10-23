"""Unit tests for the Flyte 2 backend (no cluster required).

These exercise the deterministic object-store layout, recipe transport
serialization, GPU/resource mapping, mode resolution, and the ``TrainConfig``
backend dispatch — the pure logic that drives ``modal_dojo.flyte``. They do not
launch anything on a cluster.
"""

from __future__ import annotations

import os

import pytest

from modal_dojo.common.models import Qwen3_5_4B
from modal_dojo.flyte import fields
from modal_dojo.flyte.launch import _recipe_as_plain
from modal_dojo.train_recipes.slime_recipe import SlimeRecipe


# ── Object-store layout ──────────────────────────────────────────────────────


def test_object_key_prefixes_are_deterministic() -> None:
    assert fields.data_prefix("run-1") == "dojo/runs/run-1/data"
    assert fields.checkpoints_prefix("run-1") == "dojo/runs/run-1/checkpoints"
    assert fields.hf_cache_prefix("Qwen/Qwen3") == "dojo/models/Qwen/Qwen3/hf-cache"
    assert fields.hf_cache_prefix(None) == "dojo/models/unknown-model/hf-cache"
    # run-scoped prefixes must differ; shared hf cache must not be run-scoped
    assert fields.data_prefix("a") != fields.data_prefix("b")
    assert fields.hf_cache_prefix("m") == fields.hf_cache_prefix("m")


def test_join_normalizes_slashes() -> None:
    assert fields.join("dojo", "/runs/", "x/") == "dojo/runs/x"


# ── Resource mapping ─────────────────────────────────────────────────────────


def test_gpu_spec_maps_recipe_topology() -> None:
    from modal_dojo.flyte.env import gpu_spec

    recipe = SlimeRecipe(actor_num_gpus_per_node=8, gpu_type="H100")
    assert gpu_spec(recipe) == "H100:8"


def test_resource_kwargs_minimal_mode_ignores_recipe_gpu() -> None:
    from modal_dojo.flyte.env import resource_kwargs

    recipe = SlimeRecipe(actor_num_gpus_per_node=8, gpu_type="H100", cpu=4)
    minimal = resource_kwargs(recipe, minimal_cpu=True)
    assert "gpu" not in minimal

    gpu = resource_kwargs(recipe, minimal_cpu=False)
    assert gpu["gpu"] == "H100:8"
    assert gpu["cpu"] == str(recipe.cpu)


# ── Recipe transport serialization ───────────────────────────────────────────


def test_recipe_as_plain_is_json_safe_and_drops_callables() -> None:
    recipe = SlimeRecipe(
        actor_num_gpus_per_node=1,
        gpu_type="H100",
        lr=1e-4,
        num_rollout=3,
    )
    plain = _recipe_as_plain(recipe)
    # core scalar fields survive
    assert plain["gpu_type"] == "H100"
    assert plain["lr"] == 1e-4
    assert plain["num_rollout"] == 3
    # everything is JSON-serializable (no callables / modal objects)
    import json

    json.dumps(plain)
    for value in plain.values():
        assert isinstance(value, (str, int, float, bool, list, dict, type(None)))


# ── Mode resolution ──────────────────────────────────────────────────────────


def test_resolve_mode_defaults_to_framework_or_minimal() -> None:
    from modal_dojo import TrainConfig
    from modal_dojo.flyte.launch import _resolve_mode
    from modal_dojo.common.dataset import HuggingFaceDataset
    from modal_dojo.train_recipes.miles_recipe import MilesRecipe

    slime_cfg = TrainConfig(
        dataset=HuggingFaceDataset(
            hf_repo="x/y", input_column="p", output_column="a", input_format="text"
        ),
        model=Qwen3_5_4B(),
        recipe=SlimeRecipe(),
    )
    assert _resolve_mode(slime_cfg, mode="minimal") == "minimal"
    assert _resolve_mode(slime_cfg, mode=None) == "slime"

    miles_cfg = TrainConfig(
        dataset=HuggingFaceDataset(
            hf_repo="x/y", input_column="p", output_column="a", input_format="text"
        ),
        model=Qwen3_5_4B(),
        recipe=MilesRecipe(),
    )
    assert _resolve_mode(miles_cfg, mode=None) == "miles"


# ── Pipeline module-level tasks ──────────────────────────────────────────────


def test_pipeline_exports_module_level_main_task() -> None:
    """Flyte's remote resolver needs tasks resolvable as module attributes."""
    from modal_dojo import TrainConfig
    from modal_dojo.common.dataset import HuggingFaceDataset
    from modal_dojo.flyte import pipeline as P

    cfg = TrainConfig(
        dataset=HuggingFaceDataset(
            hf_repo="x/y", input_column="p", output_column="a", input_format="text"
        ),
        model=Qwen3_5_4B(),
        recipe=SlimeRecipe(),
    )
    task = P.build_pipeline(cfg, mode="minimal")
    # The task the resolver will look up by name:
    assert P.main is task
    assert type(P.main).__name__ == "AsyncFunctionTaskTemplate"
    for name in ("download_model", "prepare_dataset", "train_stage"):
        assert hasattr(P, name)


# ── Materialization helper (no torch needed) ─────────────────────────────────


def test_materialize_dataset_writes_jsonl(tmp_path) -> None:
    from modal_dojo.flyte.smoke import SmokeDataset
    from modal_dojo.flyte.pipeline import _materialize_dataset

    dst = str(tmp_path / "data")
    os.makedirs(dst, exist_ok=True)
    path = _materialize_dataset(SmokeDataset(), dst)
    assert path.endswith("data-0.jsonl")
    with open(path) as f:
        rows = [line for line in f if line.strip()]
    assert len(rows) == 3


# ── Minimal trainer (requires torch) ─────────────────────────────────────────


def test_train_minimal_writes_checkpoint(tmp_path) -> None:
    pytest.importorskip("torch")
    from modal_dojo.flyte.smoke import SmokeDataset
    from modal_dojo.flyte.pipeline import _train_minimal

    data = str(tmp_path / "data")
    os.makedirs(data, exist_ok=True)
    from modal_dojo.flyte.pipeline import _materialize_dataset

    _materialize_dataset(SmokeDataset(), data)

    ckpt = str(tmp_path / "ckpt")
    result = _train_minimal(SmokeDataset(), data, ckpt)
    assert {f for f in os.listdir(ckpt)} >= {
        "config.json",
        "run_summary.json",
    }
    assert result["num_texts"] == 3


# ── TrainConfig backend dispatch ─────────────────────────────────────────────


def test_train_config_backend_dispatch() -> None:
    from modal_dojo.common.train import TrainConfig
    from modal_dojo.common.dataset import HuggingFaceDataset

    cfg = TrainConfig(
        dataset=HuggingFaceDataset(
            hf_repo="x/y", input_column="p", output_column="a", input_format="text"
        ),
        model=Qwen3_5_4B(),
        recipe=SlimeRecipe(),
    )
    assert cfg.backend == "flyte"
    assert cfg._flyte_backend() is True

    cfg.backend = "modal"
    assert cfg._flyte_backend() is False

    # env override wins
    old = os.environ.get("MODAL_DOJO_BACKEND")
    try:
        os.environ["MODAL_DOJO_BACKEND"] = "modal"
        assert cfg._flyte_backend() is False
        cfg.backend = "flyte"
        assert cfg._flyte_backend() is False
    finally:
        if old is None:
            os.environ.pop("MODAL_DOJO_BACKEND", None)
        else:
            os.environ["MODAL_DOJO_BACKEND"] = old
