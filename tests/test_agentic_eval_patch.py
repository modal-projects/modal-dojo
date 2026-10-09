from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import logging
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from modal_dojo.frameworks.slime.modal_helpers.patches import (
    patch_agentic_eval as patcher,
)

FIXTURES = Path(__file__).parent / "testdata/slime/agentic_eval"
FILES = {
    "model": "agentic_rl/core/model.py",
    "generate": "agentic_rl/core/generate.py",
    "env": "agentic_rl/envs/harbor/env.py",
    "rollout": "slime/ray/rollout.py",
}


@pytest.fixture
def sources(tmp_path):
    for name, relative in FILES.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((FIXTURES / f"{name}.input").read_text())
    return tmp_path


def load_functions(source, names, namespace, class_name=None):
    tree = ast.parse(source)
    if class_name:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    for node in nodes:
        node.decorator_list = []
    assert len(nodes) == len(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "pinned_slime", "exec"), namespace)


def test_patch_is_idempotent_and_fails_closed(sources):
    patcher.patch(sources)
    first = {name: (sources / path).read_text() for name, path in FILES.items()}
    patcher.patch(sources)
    assert first == {name: (sources / path).read_text() for name, path in FILES.items()}
    model = sources / FILES["model"]
    model.write_text(
        model.read_text().replace(
            "        self.gen_time = 0.0", "        self.gen_time = 1.0"
        )
    )
    before = {name: (sources / path).read_bytes() for name, path in FILES.items()}
    with pytest.raises(RuntimeError, match="anchor changed"):
        patcher.patch(sources)
    assert before == {
        name: (sources / path).read_bytes() for name, path in FILES.items()
    }


@pytest.fixture
def runtime(sources):
    patcher.patch(sources)
    ns = {
        "Path": Path,
        "time": time,
        "math": math,
        "logger": logging.getLogger(__name__),
        "RewardResult": NS,
        "_meets_min_reward": lambda *a: True,
    }
    exec(patcher.ERROR_SOURCE, ns)
    ns["PhaseTimer"] = lambda: NS(
        phase=lambda _: contextlib.nullcontext(),
        record=lambda *a: None,
        as_dict=lambda: {},
    )
    ns["rewards_mod"] = NS(
        signal_from_reward_dict=lambda r: NS(
            score_raw=r["reward"] * 100, is_solved=bool(r["reward"])
        ),
        shape=lambda s, _: float(s.is_solved),
        resolve_shape=lambda _: "binary",
        episode_outcome_from_artifacts=lambda reward, *a: reward,
        shape_outcome=lambda outcome, _: (outcome, {}),
        resolve_outcome=lambda _: "final",
    )
    load_functions(
        (sources / FILES["env"]).read_text(),
        {"_episode", "_aggregate"},
        ns,
        "HarborEnv",
    )
    load_functions((sources / FILES["generate"]).read_text(), {"_run_episode"}, ns)
    return ns


class Sandbox:
    exec_count = exec_time = exec_timeouts = 0
    exec_durations = []

    def __enter__(self):
        self.closed = False
        return self

    def __exit__(self, *args):
        self.closed = True


def episode(
    ns, *, recover=True, solved=True, failure=None, verifier_failure=None, steps=1
):
    sb = Sandbox()
    verified = []
    specs = [
        {"name": str(i), "instruction": "fix", "tests_path": "tests"}
        for i in range(steps)
    ]

    def verify(*args, **kwargs):
        verified.append(True)
        if verifier_failure:
            raise verifier_failure
        return {"reward": float(solved)}

    env = NS(
        _step_specs=lambda _: specs,
        _prepare_sandbox=lambda *a: (sb, "/repo"),
        _verify=verify,
        _aggregate=ns["_aggregate"],
        _collect_artifacts=lambda *a: {},
    )

    def leg(*args):
        if failure:
            raise failure

    md = dict(
        task_dir="/task",
        verifier={},
        reward_strategy="final",
        instance_id="task",
        grade_after_generation_error=recover,
    )
    limits = NS(eval_timeout=1800)
    env.rollout = lambda md, **kw: ns["_episode"](
        env, md, run_leg=leg, agent_budget_sec=1800, limits=limits
    )
    result = ns["_run_episode"](env, md, None, limits)
    return result, verified, sb


@pytest.mark.parametrize("solved", [True, False])
def test_transport_failure_grades_existing_artifact(runtime, solved):
    error = runtime["GenerationRequestError"](
        TimeoutError("timed out"), {"phase": "generation", "seconds": 600}
    )
    result, verified, sb = episode(runtime, solved=solved, failure=error)
    assert result.reward == float(solved)
    assert result.is_solved == solved
    assert result.grading_status == "valid"
    assert result.extra["agent_error"]["seconds"] == 600
    assert result.extra["graded_after_generation_error"]
    assert result.extra["failure_phase"] == "generation"
    assert verified == [True] and sb.closed


def test_training_and_nonrecoverable_errors_are_not_salvaged(runtime):
    timeout = runtime["GenerationRequestError"](TimeoutError("timed out"), {})
    bad_input = runtime["GenerationRequestError"](ValueError("bad response"), {})
    for recover, error in [(False, timeout), (True, bad_input)]:
        result, verified, sb = episode(runtime, recover=recover, failure=error)
        assert result.reward == 0 and not verified and sb.closed
        assert result.extra["failure_phase"] == "generation"
    assert (
        episode(runtime, recover=False, failure=timeout)[0].grading_status == "timeout"
    )


def test_grading_failure_and_incomplete_multistep_never_get_credit(runtime):
    error = runtime["GenerationRequestError"](
        TimeoutError("timed out"), {"seconds": 600}
    )
    result, verified, sb = episode(
        runtime, failure=error, verifier_failure=TimeoutError("verifier hung")
    )
    assert result.reward == 0 and result.grading_status == "timeout"
    assert result.extra["failure_phase"] == "verifier"
    assert result.extra["agent_error"] and verified and sb.closed
    result, verified, _ = episode(runtime, failure=error, steps=2)
    assert result.reward == 0 and not result.is_solved and len(verified) == 1


def test_successful_episode_unchanged(runtime):
    result, verified, _ = episode(runtime)
    assert result.reward == 1 and result.is_solved and verified
    assert result.extra["failure_phase"] is None
    assert not result.extra["graded_after_generation_error"]


def test_request_failure_records_duration_and_budget(sources, monkeypatch):
    patcher.patch(sources)
    ns = {"json": json, "time": time, "urllib": urllib}
    exec(patcher.ERROR_SOURCE, ns)
    load_functions(
        (sources / FILES["model"]).read_text(), {"_generate"}, ns, "RecordingModel"
    )
    model = NS(
        sampling_params={"max_new_tokens": 8192},
        max_context_len=0,
        url="http://example.invalid/generate",
        headers={},
        query_timeout=600,
        gen_time=0.0,
        request_errors=[],
    )

    def fail(req, timeout):
        assert timeout == 600
        assert json.loads(req.data)["input_ids"] == [11, 12]
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(ns["GenerationRequestError"]) as caught:
        ns["_generate"](model, [11, 12])
    assert caught.value.recoverable and caught.value.is_timeout
    assert model.gen_time > 0
    assert model.request_errors[0]["input_tokens"] == 2
    assert model.request_errors[0]["max_new_tokens"] == 8192
    assert model.request_errors[0]["error_type"] == "TimeoutError"
    assert model.request_errors[0]["started_at"] <= model.request_errors[0]["ended_at"]


def test_only_eval_concurrency_is_limited(sources):
    patcher.patch(sources)
    ns = {"asyncio": asyncio, "_eval_limit": None}
    active = peak = 0

    async def fake_episode(*args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.005)
        active -= 1

    ns["_generate_episode"] = fake_episode
    load_functions((sources / FILES["generate"]).read_text(), {"generate"}, ns)

    async def run(evaluation):
        await asyncio.gather(
            *(
                ns["generate"](NS(agentic_eval_concurrency=2), None, {}, evaluation)
                for _ in range(7)
            )
        )

    asyncio.run(run(True))
    assert peak == 2
    peak = 0
    asyncio.run(run(False))
    assert peak == 7


def test_diagnostics_do_not_hide_ungraded_samples(runtime):
    records = [
        dict(
            grading_status="valid",
            is_solved=True,
            graded_after_generation_error=True,
            generation_request_errors=[{}],
        ),
        dict(grading_status="timeout", failure_phase="generation"),
        dict(grading_status="valid", is_solved=False),
    ]
    metrics = runtime["evaluation_diagnostics"](
        [NS(metadata={"agentic": r}) for r in records]
    )
    assert metrics == dict(
        graded_count=2,
        ungraded_count=1,
        generation_error_count=2,
        recovered_grade_count=1,
        solve_rate_on_graded=0.5,
    )


def test_evaluation_runner_preserves_protocol_without_training():
    from tutorials.coding_agent.evaluate import build_config
    from tutorials.coding_agent.main import config

    base = build_config()
    trained = build_config(checkpoint="/checkpoints/test-run", checkpoint_step=29)
    for cfg in (base, trained):
        assert cfg.recipe.num_rollout == 0
        assert cfg.recipe.save is None and cfg.recipe.save_interval is None
        assert cfg.recipe.max_retries == 0 and cfg.recipe.no_load_optim
        assert cfg.recipe.extra_config["agentic_eval_concurrency"] == 128
        assert cfg.recipe.extra_config["agentic_eval_recover_generation_errors"]
        assert cfg.recipe.extra_config["agentic_max_steps"] == 75
        assert cfg.recipe.extra_config["agentic_episode_timeout"] == 1800
        assert cfg.recipe.eval_max_response_len == 8192
        assert cfg.recipe.eval_config["defaults"] == {
            "n_samples_per_eval_prompt": 1,
            "temperature": 0.6,
            "top_p": 1.0,
        }
    assert base.recipe.eval_config == trained.recipe.eval_config
    assert trained.recipe.extra_config["ckpt_step"] == 29
    assert build_config(proof=True).recipe.extra_config["agentic_eval_concurrency"] == 4
    assert config.recipe.num_rollout == 500


def test_evaluation_dataset_loads_without_tutorial_package():
    import cloudpickle

    from tutorials.coding_agent.evaluate import build_config

    child = '''import builtins, cloudpickle, sys
original = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.startswith("tutorials"):
        raise ModuleNotFoundError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = restricted
dataset = cloudpickle.loads(sys.stdin.buffer.read())
assert dataset.input_key() == "prompt"
assert dataset.label_key() == "label"
assert not dataset.apply_chat_template()
'''
    subprocess.run(
        [sys.executable, "-c", child],
        input=cloudpickle.dumps(build_config().dataset),
        check=True,
        timeout=30,
    )
