"""Parallelism sweep harness: grid presets, planning, and metric derivation."""

from __future__ import annotations

import pytest

from modal_dojo import TrainConfig, TrainingGroup
from modal_dojo.common.dataset import HuggingFaceDataset
from modal_dojo.common.framework import Framework
from modal_dojo.common.models import Qwen3_6_35B
from modal_dojo.common.run import TrainingRun, TrainingRunStatus
from modal_dojo.train_recipes.gpu_allocation import GpuAllocation
from modal_dojo.train_recipes.slime_recipe.qwen3_6_35b import Qwen3_6_35B_Recipe
from scripts.parallelism_sweep import (
    EP_FIELD,
    PP_FIELD,
    SGLANG_DP_FIELD,
    SGLANG_EP_FIELD,
    apply_fixed_overrides,
    classify_run,
    derive_throughput,
    derive_timing,
    parse_grid_arg,
    plan_sweep,
    preset_grid,
    render_markdown,
    rollout_tokens,
)


def _base(**recipe_kwargs) -> TrainConfig:
    return TrainConfig(
        model=Qwen3_6_35B(),
        dataset=HuggingFaceDataset(
            hf_repo="openai/gsm8k",
            input_column="question",
            output_column="answer",
            input_format="text",
        ),
        recipe=Qwen3_6_35B_Recipe(
            **{
                "num_rollout": 3,
                "actor_num_gpus_per_node": 8,
                "rollout_num_gpus": 8,
                **recipe_kwargs,
            }
        ),
    )


def test_ep_pp_preset_drops_axes_the_recipe_lacks():
    base = _base()
    grid, dropped = preset_grid("ep-pp", base.recipe, base.model)
    assert grid == {EP_FIELD: [1, 2, 4, 8]}
    assert PP_FIELD in dropped and "pipeline_model_parallel_size" in dropped[PP_FIELD]


def test_ep_preset_limits_ep_to_expert_divisors_within_world_size():
    base = _base(actor_num_gpus_per_node=4, rollout_num_gpus=4)
    grid, _ = preset_grid("ep-pp", base.recipe, base.model)
    assert grid[EP_FIELD] == [1, 2, 4]


def test_rollout_preset_uses_engine_divisors():
    base = _base(rollout_num_gpus_per_engine=4)
    grid, dropped = preset_grid("rollout", base.recipe, base.model)
    assert grid == {SGLANG_EP_FIELD: [1, 2, 4], SGLANG_DP_FIELD: [1, 2, 4]}
    assert dropped == {}


def test_unknown_preset_raises():
    base = _base()
    with pytest.raises(ValueError, match="unknown preset"):
        preset_grid("nope", base.recipe, base.model)


def test_parse_grid_arg_coerces_scalars():
    assert parse_grid_arg("recipe.expert_model_parallel_size=1, 2,4") == (
        "recipe.expert_model_parallel_size",
        [1, 2, 4],
    )
    assert parse_grid_arg("recipe.lr=1e-6") == ("recipe.lr", [1e-6])
    assert parse_grid_arg("recipe.colocate=true,false") == (
        "recipe.colocate",
        [True, False],
    )


def test_apply_fixed_overrides_rebuilds_base_without_group_tags():
    cfg = apply_fixed_overrides(
        _base(),
        {"recipe.actor_num_gpus_per_node": 4, "recipe.rollout_num_gpus": 4},
    )
    assert cfg.recipe.actor_num_gpus_per_node == 4
    assert cfg.recipe.rollout_num_gpus == 4
    assert cfg.group_id is None and cfg.group_overrides is None


def test_plan_sweep_reports_invalid_gpu_capped_and_truncated_points():
    group = TrainingGroup(
        _base(),
        grid={EP_FIELD: [1, 3, 8], SGLANG_EP_FIELD: [1, 8]},
        skip_invalid=True,
    )
    points, dropped = plan_sweep(group, max_gpus=None, max_points=3)
    assert [p.overrides for p in points] == [
        {EP_FIELD: 1, SGLANG_EP_FIELD: 1},
        {EP_FIELD: 1, SGLANG_EP_FIELD: 8},
        {EP_FIELD: 8, SGLANG_EP_FIELD: 1},
    ]
    assert all(p.allocation.total_gpus == 16 for p in points)
    reasons = {tuple(sorted(o.items())): r for o, r in dropped}
    assert "not divisible" in reasons[((EP_FIELD, 3), (SGLANG_EP_FIELD, 1))]
    assert "not divisible" in reasons[((EP_FIELD, 3), (SGLANG_EP_FIELD, 8))]
    assert "--max-points 3" in reasons[((EP_FIELD, 8), (SGLANG_EP_FIELD, 8))]

    _, capped = plan_sweep(group, max_gpus=8, max_points=None)
    assert any("--max-gpus 8" in r for _, r in capped)


# ── Metric derivation ───────────────────────────────────────────────────────


def _step(duration: float, *, partial: bool = False) -> dict:
    return {"duration_s": duration, "partial": partial}


def _substeps(**durations: float) -> dict:
    return {
        name: {"duration_s": d, "wall_duration_s": d, "invocation_count": 1}
        for name, d in durations.items()
    }


STEP_TIMES = {
    "1": _step(100.0),
    "2": _step(40.0),
    "3": _step(50.0),
    "4": _step(5.0, partial=True),
}
SUBSTEP_TIMES = {
    "1": _substeps(generate_rollouts=80.0, train_models=20.0),
    "2": _substeps(generate_rollouts=30.0, train_models=8.0, onload_train=1.0),
    "3": _substeps(generate_rollouts=40.0, train_models=8.0, onload_train=1.0),
    "4": _substeps(generate_rollouts=5.0),
}


def test_derive_timing_skips_warmup_and_partial_steps():
    timing = derive_timing(STEP_TIMES, SUBSTEP_TIMES)
    assert timing["measured_steps"] == [2, 3]
    assert timing["step_time_s"] == 45.0
    assert timing["phase_times_s"]["generate_rollouts"] == 35.0
    assert timing["phase_times_s"]["train_models"] == 8.0
    assert timing["phase_times_s"]["onload_train"] == 1.0
    assert timing["phase_times_s"]["compute_log_probs"] is None
    assert timing["other_s"] == 1.0


def test_derive_timing_with_no_complete_steps_is_empty():
    timing = derive_timing({"1": _step(10.0)}, {"1": _substeps(train_models=1.0)})
    assert timing["measured_steps"] == []
    assert timing["step_time_s"] is None
    assert timing["other_s"] is None


def test_rollout_tokens_sums_inference_metadata():
    payload = {
        "samples": [
            {"metadata": {"inference": {"tokens_in": 100, "tokens_out": 50}}},
            {"metadata": {"inference": {"tokens_in": 20, "tokens_out": 30}}},
            {"metadata": {}},
        ]
    }
    assert rollout_tokens(payload) == (120, 80)
    assert rollout_tokens({"samples": [{"metadata": {}}]}) is None


def _allocation(total: int, rollout: int) -> GpuAllocation:
    return GpuAllocation(
        actor_gpus=total - rollout,
        critic_gpus=0,
        rollout_gpus=rollout,
        total_gpus=total,
        total_nodes=1,
        gpus_per_node=total,
        rollout_num_gpus_per_engine=rollout,
        rollout_engines=1,
        colocate=False,
    )


def test_derive_throughput_normalizes_by_gpus_and_step_time():
    timing = derive_timing(STEP_TIMES, SUBSTEP_TIMES)
    tokens = {2: (1000, 800), 3: (1000, 800)}
    out = derive_throughput(tokens, timing, _allocation(total=16, rollout=8))
    assert out["tokens_in_per_step"] == 1000
    assert out["tokens_out_per_step"] == 800
    assert out["tokens_per_s_per_gpu"] == pytest.approx(1800 / 45.0 / 16, abs=0.01)
    assert out["gen_tokens_per_s_per_rollout_gpu"] == pytest.approx(
        800 / 35.0 / 8, abs=0.01
    )


def test_derive_throughput_is_none_without_tokens_or_allocation():
    timing = derive_timing(STEP_TIMES, SUBSTEP_TIMES)
    assert (
        derive_throughput({}, timing, _allocation(16, 8))["tokens_per_s_per_gpu"]
        is None
    )
    assert derive_throughput({2: (1, 1)}, timing, None)["tokens_per_s_per_gpu"] is None


def _run(
    status: TrainingRunStatus, error: str | None = None, **metadata
) -> TrainingRun:
    return TrainingRun(
        training_run_id="run-1",
        framework=Framework.SLIME,
        config={},
        status=status,
        error_message=error,
        metadata=metadata or None,
    )


def test_classify_run_distinguishes_oom_preemption_and_errors():
    assert classify_run(_run(TrainingRunStatus.COMPLETED))["failure_class"] is None
    oom = classify_run(
        _run(TrainingRunStatus.FAILED, "torch.OutOfMemoryError: CUDA out of memory")
    )
    assert oom["failure_class"] == "oom"
    preempted = classify_run(
        _run(TrainingRunStatus.FAILED, "worker lost", attempt_count=2)
    )
    assert preempted["failure_class"] == "preempted" and preempted["attempts"] == 2
    assert (
        classify_run(_run(TrainingRunStatus.FAILED, "boom"))["failure_class"] == "error"
    )
    assert classify_run(_run(TrainingRunStatus.RUNNING))["status"] == "running"


def test_render_markdown_ranks_by_step_time_and_shows_axes():
    def point(run_id: str, ep: int, step: float | None, status: str = "completed"):
        return {
            "training_run_id": run_id,
            "overrides": {EP_FIELD: ep},
            "status": status,
            "failure_class": "oom" if status == "failed" else None,
            "total_gpus": 16,
            "measured_steps": [2, 3] if step else [],
            "step_time_s": step,
            "phase_times_s": {"generate_rollouts": step, "train_models": None},
            "other_s": None,
            "tokens_per_s_per_gpu": None,
            "gen_tokens_per_s_per_rollout_gpu": None,
        }

    md = render_markdown(
        "g",
        [point("a", 1, 50.0), point("b", 8, 40.0), point("c", 4, None, "failed")],
    )
    rows = [line for line in md.splitlines() if line.startswith("| ")]
    assert rows[0].startswith("| expert_model_parallel_size | status |")
    assert [r.split("|")[1].strip() for r in rows[1:]] == ["8", "1", "4"]
    assert "failed (oom)" in rows[3]
