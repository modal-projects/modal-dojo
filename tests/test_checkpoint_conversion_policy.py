from types import SimpleNamespace

import pytest

from modal_dojo.common.launcher_utils import get_checkpoint_conversion_policy


def test_implicit_single_rank_layout_keeps_automatic_conversion():
    assert get_checkpoint_conversion_policy(SimpleNamespace())[:2] == (1, 8)


@pytest.mark.parametrize("override", ["tensor", "pipeline"])
def test_explicit_one_disables_automatic_conversion_parallelism(override):
    cfg = SimpleNamespace(**{f"conversion_{override}_model_parallel_size": 1})
    nodes, processes, args = get_checkpoint_conversion_policy(cfg)
    assert (nodes, processes) == (1, 1)
    assert args == [
        "--tensor-model-parallel-size 1",
        "--pipeline-model-parallel-size 1",
    ]


def test_explicit_tp1_pp1_overrides_sharded_training_and_drops_pipeline_split():
    cfg = SimpleNamespace(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        conversion_tensor_model_parallel_size=1,
        conversion_pipeline_model_parallel_size=1,
        conversion_expert_model_parallel_size=1,
        conversion_expert_tensor_parallel_size=1,
        decoder_first_pipeline_num_layers=10,
        mtp_num_layers=0,
    )
    nodes, processes, args = get_checkpoint_conversion_policy(cfg)
    assert (nodes, processes) == (1, 1)
    assert "--expert-model-parallel-size 1" in args
    assert "--expert-tensor-parallel-size 1" in args
    assert "--mtp-num-layers 0" in args
    assert not any("pipeline-num-layers" in arg for arg in args)


@pytest.mark.parametrize("mtp", [None, 0, 1])
def test_mtp_none_is_omitted_but_zero_is_an_override(mtp):
    args = get_checkpoint_conversion_policy(SimpleNamespace(mtp_num_layers=mtp))[2]
    assert [arg for arg in args if arg.startswith("--mtp-num-layers")] == (
        [] if mtp is None else [f"--mtp-num-layers {mtp}"]
    )


def test_expert_parallelism_wider_than_tp_pp_widens_the_world():
    cfg = SimpleNamespace(
        conversion_tensor_model_parallel_size=1,
        conversion_expert_model_parallel_size=2,
        conversion_expert_tensor_parallel_size=1,
    )
    nodes, processes, args = get_checkpoint_conversion_policy(cfg)
    assert (nodes, processes) == (1, 2)
    assert "--expert-model-parallel-size 2" in args
