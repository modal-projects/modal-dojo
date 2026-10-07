from __future__ import annotations

import asyncio
import copy
import os
import tempfile
from dataclasses import dataclass, field

import pytest
import yaml

from modal_dojo.common import launcher_helpers
from modal_dojo.common.batch_size_inference import (
    METADATA_KEY,
    BatchSizeInference,
    OOMDetector,
    inferred_batch_size_result,
    looks_like_oom,
    resolve_batch_size_knob,
    set_recipe_value,
    validate_batch_size_inference,
)
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.launcher_utils import prepare_launch_config
from modal_dojo.common.models.qwen3_4b import Qwen3_4B
from modal_dojo.common.ray_cluster import ModalRayJobResult
from modal_dojo.train_recipes.miles_recipe import MilesRecipe
from modal_dojo.train_recipes.slime_recipe import SlimeRecipe


@pytest.mark.parametrize(
    "text",
    [
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "RuntimeError: CUDA error: out of memory",
        "CUDA_ERROR_OUT_OF_MEMORY: out of memory",
        "cuBLAS error: CUBLAS_STATUS_ALLOC_FAILED",
    ],
)
def test_looks_like_oom_matches_cuda_failures(text):
    assert looks_like_oom(text)


@pytest.mark.parametrize(
    "text",
    [None, "", "ValueError: dataset is empty", "Job finished with status: FAILED"],
)
def test_looks_like_oom_ignores_other_failures(text):
    assert not looks_like_oom(text)


def test_oom_detector_keeps_first_evidence():
    detector = OOMDetector()
    detector.observe("loading model\n")
    detector.observe("torch.OutOfMemoryError: CUDA out of memory\n")
    detector.observe("RuntimeError: out of memory (again)\n")
    assert detector.evidence == "torch.OutOfMemoryError: CUDA out of memory"


def test_resolve_knob_dynamic_batching_floors_at_longest_sample():
    recipe = SlimeRecipe.get_base_recipe(Qwen3_4B())
    recipe.infer_batch_size = True
    recipe.max_tokens_per_gpu = 262144
    knob, value, floor = resolve_batch_size_knob(recipe)
    assert knob == "max_tokens_per_gpu"
    assert value == 262144
    assert floor == recipe.rollout_max_response_len
    recipe.extra_config = {"rollout_max_prompt_len": 2048}
    assert resolve_batch_size_knob(recipe)[2] == 2048 + recipe.rollout_max_response_len


def test_resolve_knob_respects_context_len_and_cp():
    recipe = MilesRecipe(
        infer_batch_size=True,
        max_tokens_per_gpu=65536,
        context_parallel_size=2,
        extra_config={"rollout_max_context_len": 9000},
    )
    assert resolve_batch_size_knob(recipe) == ("max_tokens_per_gpu", 65536, 4500)


def test_resolve_knob_fixed_micro_batch_uses_field_or_hatch():
    recipe = MilesRecipe(infer_batch_size=True, use_dynamic_batch_size=False)
    with pytest.raises(DojoConfigError, match="micro_batch_size"):
        resolve_batch_size_knob(recipe)
    recipe.micro_batch_size = 64
    assert resolve_batch_size_knob(recipe) == ("micro_batch_size", 64, 1)

    slime = SlimeRecipe.get_base_recipe(Qwen3_4B())
    slime.infer_batch_size = True
    slime.extra_config = {"use_dynamic_batch_size": False, "micro_batch_size": 32}
    assert resolve_batch_size_knob(slime) == ("micro_batch_size", 32, 1)


def test_validate_rejects_value_already_at_floor():
    recipe = SlimeRecipe.get_base_recipe(Qwen3_4B())
    recipe.infer_batch_size = True
    recipe.max_tokens_per_gpu = recipe.rollout_max_response_len
    with pytest.raises(DojoConfigError, match="Start much higher"):
        validate_batch_size_inference(recipe)
    recipe.infer_batch_size = False
    validate_batch_size_inference(recipe)


def test_set_recipe_value_updates_field_then_materialized_hatch():
    recipe = SlimeRecipe.get_base_recipe(Qwen3_4B())
    set_recipe_value(recipe, "max_tokens_per_gpu", 1234)
    assert recipe.max_tokens_per_gpu == 1234

    recipe.extra_config = {"micro_batch_size": 16, "use_dynamic_batch_size": False}
    set_recipe_value(recipe, "micro_batch_size", 8)
    assert recipe.extra_config["micro_batch_size"] == 8

    tmpdir = tempfile.mkdtemp()
    prepare_launch_config(recipe, None, tmpdir, yaml_config_fields=("extra_config",))
    assert isinstance(recipe.extra_config, str)
    set_recipe_value(recipe, "micro_batch_size", 4)
    with open(recipe.extra_config) as f:
        assert yaml.safe_load(f)["micro_batch_size"] == 4
    assert recipe._escape_hatch_values()["micro_batch_size"] == 4
    assert "micro_batch_size" in recipe._escape_hatch_keys()
    assert "--micro-batch-size" not in " ".join(recipe.cli_args(None))
    os.remove(recipe.extra_config)


def test_shrink_halves_clamps_and_stops_at_floor():
    state = BatchSizeInference(
        knob="max_tokens_per_gpu", initial=20000, floor=6000, current=20000
    )
    assert state.shrink() == 10000
    assert state.shrink() == 6000
    assert state.shrink() is None
    assert state.result is None
    state.finish_attempt("succeeded")
    assert state.result == 6000
    assert state.to_metadata()["inferred_batch_size_result"] == 6000


@dataclass
class _FakeRun:
    training_run_id: str = "run-1"
    metadata: dict = field(default_factory=dict)
    error_message: str = ""
    saves: int = 0
    persisted: list[dict] = field(default_factory=list)

    async def save(self, *, is_async: bool = False) -> None:
        self.saves += 1
        self.persisted.append(copy.deepcopy(self.metadata))


@dataclass
class _Recipe:
    infer_batch_size: bool = True
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 40000
    rollout_max_prompt_len: int = 1000
    rollout_max_response_len: int = 4000


class _FakeCluster:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.commands: list[str] = []

    async def submit_and_tail(self, cmd, *, runtime_env=None, on_log_line=None):
        self.commands.append(cmd)
        lines, result = self.outcomes.pop(0)
        for line in lines:
            on_log_line(line)
        return result


_OOM = ModalRayJobResult(status="FAILED", is_success=False, message="Job failed")
_OK = ModalRayJobResult(status="SUCCEEDED", is_success=True)


def _run(cluster, recipe, run):
    return asyncio.run(
        launcher_helpers.run_training_attempts(
            cluster=cluster,
            recipe=recipe,
            run_record=run,
            runtime_env={"env_vars": {}},
            build_cmd=lambda: f"train --max-tokens-per-gpu {recipe.max_tokens_per_gpu}",
            retry_delay_seconds=0,
        )
    )


def test_start_restores_previous_container_state():
    recipe = _Recipe()
    run = _FakeRun(
        metadata={
            METADATA_KEY: {
                "knob": "max_tokens_per_gpu",
                "current": 10000,
                "attempts": [
                    {"value": 20000, "outcome": "oom"},
                    {"value": 10000, "outcome": "running"},
                ],
            }
        }
    )
    state = BatchSizeInference.start(recipe, run)
    assert state is not None
    assert (state.initial, state.current, state.floor) == (40000, 10000, 5000)
    assert [a["outcome"] for a in state.attempts] == ["oom", "interrupted"]
    assert BatchSizeInference.start(_Recipe(infer_batch_size=False), run) is None


def test_run_training_attempts_halves_on_oom_until_it_fits():
    recipe = _Recipe()
    run = _FakeRun()
    cluster = _FakeCluster(
        [
            (["loading\n", "torch.OutOfMemoryError: CUDA out of memory\n"], _OOM),
            (
                [],
                ModalRayJobResult(
                    status="FAILED",
                    is_success=False,
                    message="RuntimeError: CUDA error: out of memory",
                ),
            ),
            (["step 1\n"], _OK),
        ]
    )
    result = _run(cluster, recipe, run)
    assert result.is_success
    assert cluster.commands == [
        "train --max-tokens-per-gpu 40000",
        "train --max-tokens-per-gpu 20000",
        "train --max-tokens-per-gpu 10000",
    ]
    assert recipe.max_tokens_per_gpu == 10000
    stored = run.metadata[METADATA_KEY]
    assert stored["settled"] is True
    assert stored["inferred_batch_size_result"] == 10000
    assert [a["outcome"] for a in stored["attempts"]] == ["oom", "oom", "succeeded"]
    assert "CUDA out of memory" in stored["attempts"][0]["evidence"]
    assert inferred_batch_size_result(run.metadata) == 10000
    # The settled state must reach the volume, not just the in-memory record.
    assert run.persisted[-1][METADATA_KEY]["settled"] is True
    assert run.persisted[-1][METADATA_KEY]["inferred_batch_size_result"] == 10000


def test_run_training_attempts_does_not_retry_non_oom_failures():
    recipe = _Recipe()
    run = _FakeRun()
    cluster = _FakeCluster(
        [
            (
                ["KeyError: 'prompt'\n"],
                ModalRayJobResult(
                    status="FAILED", is_success=False, message="KeyError"
                ),
            )
        ]
    )
    with pytest.raises(RuntimeError, match="KeyError"):
        _run(cluster, recipe, run)
    assert run.persisted[-1][METADATA_KEY]["attempts"][-1]["outcome"] == "failed"
    assert len(cluster.commands) == 1
    assert run.metadata[METADATA_KEY]["attempts"][-1]["outcome"] == "failed"
    assert run.metadata[METADATA_KEY]["inferred_batch_size_result"] is None
    assert inferred_batch_size_result(run.metadata) is None


def test_run_training_attempts_gives_up_at_floor():
    recipe = _Recipe(max_tokens_per_gpu=8000)
    run = _FakeRun()
    oom = (["CUDA out of memory\n"], _OOM)
    cluster = _FakeCluster([oom, oom, oom])
    with pytest.raises(RuntimeError, match="already the floor"):
        _run(cluster, recipe, run)
    assert cluster.commands == [
        "train --max-tokens-per-gpu 8000",
        "train --max-tokens-per-gpu 5000",
    ]
    assert "already the floor" in run.error_message


def test_run_training_attempts_without_flag_is_single_submit():
    recipe = _Recipe(infer_batch_size=False)
    run = _FakeRun()
    cluster = _FakeCluster([(["CUDA out of memory\n"], _OOM)])
    with pytest.raises(RuntimeError, match="Job failed"):
        _run(cluster, recipe, run)
    assert len(cluster.commands) == 1
    assert METADATA_KEY not in run.metadata
