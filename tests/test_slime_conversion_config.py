from modal_dojo import Qwen3_6_35B, Qwen3_6_35B_Recipe
from modal_dojo.frameworks.slime.launcher import (
    _build_conversion_config,
    _conversion_config_matches,
)


def _config(**overrides):
    return _build_conversion_config(
        Qwen3_6_35B_Recipe(**overrides), model=Qwen3_6_35B()
    )


def test_conversions_at_different_parallelism_are_interchangeable() -> None:
    single = _config(
        actor_num_gpus_per_node=1,
        rollout_num_gpus=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        expert_model_parallel_size=1,
    )
    node_wide = _config(
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        expert_model_parallel_size=4,
    )
    tp2 = _config()

    assert single != node_wide != tp2
    for stored, current in [(single, node_wide), (node_wide, tp2), (tp2, single)]:
        assert _conversion_config_matches(stored, current)


def test_different_weights_are_stale() -> None:
    current = {
        "num_nodes": 1,
        "nproc_per_node": 2,
        "extra_args": [
            "--tensor-model-parallel-size 2",
            "--make-vocab-size-divisible-by 128",
        ],
        "model_name": "Qwen/Qwen3.6-35B-A3B",
        "slime_model_script": "scripts/models/qwen3.5-35B-A3B.sh",
    }
    for key, value in [
        ("model_name", "Qwen/Qwen3.6-27B"),
        ("slime_model_script", "scripts/models/qwen3.5-27B.sh"),
        ("extra_args", ["--tensor-model-parallel-size 2"]),
    ]:
        assert not _conversion_config_matches({**current, key: value}, current)


def test_mtp_conversions_must_match_exactly() -> None:
    current = {
        "num_nodes": 1,
        "nproc_per_node": 1,
        "extra_args": ["--mtp-num-layers 1"],
        "model_name": "m",
        "slime_model_script": "",
    }
    sharded = {
        **current,
        "nproc_per_node": 8,
        "extra_args": ["--tensor-model-parallel-size 8", "--mtp-num-layers 1"],
    }

    assert _conversion_config_matches(current, current)
    assert not _conversion_config_matches(sharded, current)
