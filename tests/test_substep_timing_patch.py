"""Golden-file test for the substep-timing patcher.

Anchors are literal source lines, so the test worth having is one that runs them
over the sources they were written for: ``tests/testdata`` holds slime's
``train.py`` after the rollout-status patcher (the state the timing patcher sees)
and miles' two entrypoints as shipped in the pinned image. Regenerate the
expected output with ``uv run pytest tests/test_substep_timing_patch.py
--rewrite``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TESTDATA = Path(__file__).parent / "testdata"
FRAMEWORKS = Path(__file__).parents[1] / "modal_dojo" / "frameworks"


def patcher_path(framework: str) -> Path:
    return (
        FRAMEWORKS / framework / "modal_helpers" / "patches" / "patch_substep_timing.py"
    )


# framework, fixture, entrypoint, golden output, driver phases the loop records.
DRIVERS = [
    (
        "slime",
        "slime/train.py.status.output",
        "train.py",
        "slime/train.py.timing.output",
        {
            "evaluate_rollouts",
            "generate_rollouts",
            "offload_rollout",
            "train_models",
            "checkpoint_save",
            "offload_train",
            "weight_sync",
            "initial_weight_sync",
            "evaluate_rollouts_end",
        },
    ),
    (
        "miles",
        "miles/train.py.status.output",
        "train.py",
        "miles/train.py.timing.output",
        {
            "evaluate_rollouts",
            "generate_rollouts",
            "offload_rollout",
            "train_models",
            "checkpoint_save",
            "offload_train",
            "weight_sync",
            "initial_weight_sync",
            "evaluate_rollouts_end",
        },
    ),
    (
        "miles",
        "miles/train_async.py.status.output",
        "train_async.py",
        "miles/train_async.py.timing.output",
        {
            "evaluate_rollouts",
            "wait_for_rollout",
            "wait_for_next_rollout",
            "train_models",
            "offload_train",
            "checkpoint_save",
            "weight_sync",
            "initial_weight_sync",
            "evaluate_rollouts_end",
        },
    ),
    (
        "slime",
        "slime/train_async.py.status.output",
        "train_async.py",
        "slime/train_async.py.timing.output",
        {
            "wait_for_rollout",
            "wait_for_next_rollout",
            "train_models",
            "checkpoint_save",
            "weight_sync",
            "initial_weight_sync",
            "evaluate_rollouts_end",
        },
    ),
]

MILES_PACKAGE_TARGETS = [
    (
        "miles/ray/rollout/rollout_manager.py",
        "miles/rollout_manager.py.input",
        "miles/rollout_manager.py.timing.output",
    ),
    (
        "miles/rollout/rm_hub/__init__.py",
        "miles/rm_hub_init.py.input",
        "miles/rm_hub_init.py.timing.output",
    ),
    (
        "miles/rollout/sglang_rollout.py",
        "miles/sglang_rollout.py.input",
        "miles/sglang_rollout.py.timing.output",
    ),
    (
        "miles/backends/megatron_utils/actor.py",
        "miles/actor.py.input",
        "miles/actor.py.timing.output",
    ),
    (
        "miles/backends/megatron_utils/model.py",
        "miles/model.py.input",
        "miles/model.py.timing.output",
    ),
]

SLIME_PACKAGE_TARGETS = [
    (
        "slime/ray/rollout.py",
        "slime/rollout.py.input",
        "slime/rollout.py.timing.output",
    ),
    (
        "slime/rollout/rm_hub/__init__.py",
        "slime/rm_hub_init.py.input",
        "slime/rm_hub_init.py.timing.output",
    ),
    (
        "slime/rollout/sglang_rollout.py",
        "slime/sglang_rollout.py.input",
        "slime/sglang_rollout.py.timing.output",
    ),
    (
        "slime/backends/megatron_utils/actor.py",
        "slime/actor.py.input",
        "slime/actor.py.timing.output",
    ),
    (
        "slime/backends/megatron_utils/model.py",
        "slime/model.py.input",
        "slime/model.py.timing.output",
    ),
]


@pytest.fixture(scope="session")
def patchers() -> dict[str, object]:
    """Load each framework's patch script by path: they run standalone in the
    image, and are deliberately not importable as part of the package."""
    loaded = {}
    for framework in ("slime", "miles"):
        name = f"patch_substep_timing_{framework}"
        spec = importlib.util.spec_from_file_location(name, patcher_path(framework))
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # its dataclass resolves annotations here
        spec.loader.exec_module(module)
        loaded[framework] = module
    return loaded


@pytest.fixture(scope="session")
def miles(patchers):
    return patchers["miles"]


def _patched(patcher, tmp_path, fixture: str, entrypoint: str) -> str:
    work = tmp_path / entrypoint
    work.write_text((TESTDATA / fixture).read_text())
    patcher._patch_file(work, patcher.ENTRYPOINTS[entrypoint])
    return work.read_text()


@pytest.mark.parametrize(
    "framework, fixture, entrypoint, golden, expected_phases", DRIVERS
)
def test_patch_matches_golden(
    patchers, tmp_path, request, framework, fixture, entrypoint, golden, expected_phases
):
    patched = _patched(patchers[framework], tmp_path, fixture, entrypoint)
    golden_path = TESTDATA / golden

    if request.config.getoption("--rewrite"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(patched)
        return

    assert patched == golden_path.read_text(), (
        f"golden mismatch for {golden}; rerun with --rewrite to accept"
    )
    assert "with _tg_role('driver', rollout_id) as _tg_rec:" in patched
    for phase in expected_phases:
        assert f"with _tg_rec.phase('{phase}'):" in patched
    compile(patched, entrypoint, "exec")


@pytest.mark.parametrize("framework", ("slime", "miles"))
def test_replace_once_rejects_mid_line_anchor(patchers, tmp_path, framework):
    with pytest.raises(RuntimeError, match="line boundary"):
        patchers[framework].replace_once(
            "prefix    anchor\n",
            "    anchor",
            "replacement",
            tmp_path / "source.py",
        )


@pytest.mark.parametrize("path, fixture, golden", MILES_PACKAGE_TARGETS)
def test_miles_package_patch_matches_golden(
    miles, tmp_path, request, path, fixture, golden
):
    target = next(target for target in miles.PACKAGE_TARGETS if target.path == path)
    work = tmp_path / target.path
    work.parent.mkdir(parents=True)
    work.write_text((TESTDATA / fixture).read_text())
    miles.patch_package_file(tmp_path, target)
    patched = work.read_text()
    golden_path = TESTDATA / golden

    if request.config.getoption("--rewrite"):
        golden_path.write_text(patched)
        return

    assert patched == golden_path.read_text(), (
        f"golden mismatch for {golden}; rerun with --rewrite to accept"
    )
    for phase, _ in target.blocks:
        assert f"with _tg_time_phase('{phase}'):" in patched
    if target.scope is not None:
        assert "_tg_role('rollout', rollout_id)" in patched or "_tg_mrec(" in patched
    compile(patched, path, "exec")


@pytest.mark.parametrize("path, fixture, golden", SLIME_PACKAGE_TARGETS)
def test_slime_package_patch_matches_golden(
    patchers, tmp_path, request, path, fixture, golden
):
    patcher = patchers["slime"]
    target = next(target for target in patcher.PACKAGE_TARGETS if target.path == path)
    work = tmp_path / target.path
    work.parent.mkdir(parents=True)
    work.write_text((TESTDATA / fixture).read_text())
    patcher.patch_package_file(tmp_path, target)
    patched = work.read_text()
    golden_path = TESTDATA / golden

    if request.config.getoption("--rewrite"):
        golden_path.write_text(patched)
        return

    assert patched == golden_path.read_text(), (
        f"golden mismatch for {golden}; rerun with --rewrite to accept"
    )
    for phase, _ in target.blocks:
        assert f"with _tg_time_phase('{phase}'):" in patched
    if target.scope is not None:
        assert "_tg_role('rollout', rollout_id)" in patched or "_tg_mrec(" in patched
    compile(patched, path, "exec")


def test_a_conditional_phase_is_timed_inside_its_branch(patchers, tmp_path):
    """A skipped save must record nothing, not a 0s bar on every rollout.

    Its condition spans three lines, so the closing ``):`` sits at the ``if``'s
    own indent and must not be read as the start of another clause.
    """
    patched = _patched(
        patchers["slime"], tmp_path, "slime/train.py.status.output", "train.py"
    )
    save = patched.split("if release_train or should_run_periodic_action(")[1]
    before_wrap = save.split("with _tg_rec.phase('checkpoint_save'):")[0]
    assert before_wrap.split("#")[0].rstrip().endswith("):")


def test_patching_twice_is_a_no_op(miles, tmp_path, capsys):
    work = tmp_path / "train.py"
    work.write_text((TESTDATA / "miles/train.py.status.output").read_text())
    miles._patch_file(work, miles.ENTRYPOINTS["train.py"])
    once = work.read_text()
    miles._patch_file(work, miles.ENTRYPOINTS["train.py"])
    assert work.read_text() == once
    assert "already patched" in capsys.readouterr().out


def test_preamble_stays_below_future_imports(patchers):
    source = '"""module docs"""\nfrom __future__ import annotations\n\nvalue = 1\n'
    for patcher in patchers.values():
        patched = patcher._inject_preamble(source)
        assert patched.index("from __future__ import annotations") < patched.index(
            patcher.PREAMBLE_MARKER
        )
        compile(patched, "future_module.py", "exec")


def test_a_moved_anchor_fails_the_build(miles, tmp_path):
    """Half-instrumented timing is worse than none: a lane would just be absent."""
    work = tmp_path / "train.py"
    source = (TESTDATA / "miles/train.py.status.output").read_text()
    work.write_text(
        source.replace("await offload_train()", "await offload_train(args)")
    )
    with pytest.raises(RuntimeError, match="expected 1 occurrence"):
        miles._patch_file(work, miles.ENTRYPOINTS["train.py"])


def test_a_duplicate_anchor_fails_the_build(miles, tmp_path):
    work = tmp_path / "train_async.py"
    source = (TESTDATA / "miles/train_async.py.status.output").read_text()
    source = source.replace(
        "await critic_model.offload()",
        "await critic_model.offload()\n                await critic_model.offload()",
    )
    work.write_text(source)
    with pytest.raises(RuntimeError, match="expected 1 occurrence"):
        miles._patch_file(work, miles.ENTRYPOINTS["train_async.py"])


@pytest.mark.parametrize("mode", ("auto", "off"))
def test_package_patch_failure_is_best_effort(
    miles, tmp_path, monkeypatch, capsys, mode
):
    target = miles.PackageTarget(
        path="missing.py",
        scope=None,
        blocks=(("missing", "missing\n"),),
    )
    monkeypatch.setenv("MODAL_DOJO_SUBSTEP_TIMING", mode)
    miles.patch_package_file(tmp_path, target)
    assert "substep timing patch skipped" in capsys.readouterr().out


def test_async_training_offloads_are_separate_from_train(miles, tmp_path):
    patched = _patched(
        miles, tmp_path, "miles/train_async.py.status.output", "train_async.py"
    )
    critic_train = patched.index("with _tg_rec.phase('train_models'):")
    critic_offload = patched.index("with _tg_rec.phase('offload_train'):")
    actor_train = patched.index("with _tg_rec.phase('train_models'):", critic_offload)
    actor_offload = patched.index("with _tg_rec.phase('offload_train'):", actor_train)
    assert critic_train < critic_offload < actor_train < actor_offload
    assert "with _tg_rec.phase('generate_rollouts'):" not in patched
    assert "with _tg_rec.phase('offload_rollout'):" not in patched
    for phase in (
        "initial_weight_sync",
        "evaluate_rollouts",
        "evaluate_rollouts_end",
    ):
        assert f"with _tg_rec.phase('{phase}'):" in patched
    assert (
        "if not args.eval_uses_snapshots:\n"
        "                    # PATCHED_MODAL_DOJO_TIMING_EVALUATE_ROLLOUTS_END"
    ) in patched
    assert (
        "with _tg_rec.phase('evaluate_rollouts_end'):\n"
        "            await eval_dispatcher.drain()"
    ) in patched


@pytest.mark.parametrize("framework", ["miles", "slime"])
def test_per_sample_generation_target_wraps_only_generation_branch(
    patchers, tmp_path, framework
):
    patcher = patchers[framework]
    target = next(
        target
        for target in patcher.PACKAGE_TARGETS
        if target.path.endswith("sglang_rollout.py")
    )
    blocks = dict(target.blocks)
    source = (
        "async def generate_and_rm(args, sample, sampling_params, evaluation=False):\n"
        "    async with state.semaphore:\n"
        "        with state.dp_rank_context() as _:\n"
        f"{blocks['sample_generation']}"
        f"{blocks['reward']}"
        "    return sample\n"
    )
    work = tmp_path / target.path
    work.parent.mkdir(parents=True)
    work.write_text(source)
    patcher.patch_package_file(tmp_path, target)
    patched = work.read_text()
    assert patched.count("with _tg_time_phase('sample_generation'):") == 1
    assert patched.count("with _tg_time_phase('reward'):") == 1
    assert patched.index("with _tg_time_phase('sample_generation'):") < patched.index(
        "with _tg_time_phase('reward'):"
    )
    compile(patched, str(work), "exec")


def test_missing_package_file_warns_and_continues(miles, tmp_path, capsys):
    miles.patch_package_file(tmp_path, miles.PACKAGE_TARGETS[0])
    assert "substep timing patch skipped" in capsys.readouterr().out


_EXECUTOR_TRAIN = """
def train(args):
    create_rollout_components()
    create_training_models()
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        inference_controller.prepare_rollout()
        rollout_executor.get.remote()
        inference_controller.offload_kv()
        inference_controller.offload_weights()
        inference_controller.offload()
        actor_model.onload()
        actor_model.train()
        critic_model.train()
        actor_model.offload()
        critic_model.offload()
        offload_train()
        actor_model.clear_memory()
        actor_model.offload_grad_buffer()
        inference_controller.onload_weights()
        save()
        inference_controller.onload_kv()
        inference_controller.prepare_eval()
        eval_dispatcher.dispatch()
"""


def test_executor_driver_tolerates_missing_grad_buffer_offload(miles, tmp_path, capsys):
    """The K3-only gradient offload is optional on other executor layouts."""
    src = _EXECUTOR_TRAIN.replace(
        "        actor_model.offload_grad_buffer()\n",
        "        # offload_grad_buffer handled elsewhere\n",
    )
    patched = miles._patch_executor_driver(src, tmp_path / "train.py")
    assert "_tg_time_phase('generate_rollouts')" in patched
    assert "offload_train_gradients" not in patched
    assert "offload_train_gradients skipped" in capsys.readouterr().out
    compile(patched, "train.py", "exec")


def test_executor_driver_still_requires_required_calls(miles, tmp_path):
    src = _EXECUTOR_TRAIN.replace("        save()\n", "")
    with pytest.raises(RuntimeError, match="timing calls missing"):
        miles._patch_executor_driver(src, tmp_path / "train.py")


def test_executor_package_skips_unpatchable_files(miles, tmp_path, capsys):
    """One executor file that drifts must not lose the others, or the build."""
    files = {
        "miles/ray/rollout/rollout_executor.py": (
            "class R:\n"
            "    def get(self, rollout_id):\n"
            "        self._get_rollout_data()\n"
            "        convert_samples_to_train_data()\n"
        ),
        "miles/backends/megatron_utils/actor.py": (
            "class A:\n"
            "    def compute_log_prob(self, rollout_id):\n"
            "        pass\n"
            "    def train(self, rollout_id):\n"
            "        pass\n"
        ),
        "miles/backends/megatron_utils/model.py": (
            "class M:\n    def train_one_step(self):\n        optimizer.step()\n"
        ),
        "miles/ray/train/group.py": (
            "class G:\n"
            "    def update_weights(self, rollout_id):\n"
            "        self._inference_controller.start_update_weights()\n"
            "        retry()\n"
            "        self._inference_controller.end_update_weights()\n"
            "        self._maybe_log_inference_engine_weight_checksums()\n"
        ),
    }
    for rel, src in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(src)

    miles.patch_executor_package(tmp_path)

    out = capsys.readouterr().out
    assert "model.py" in out and "skipped" in out
    assert (
        miles.PREAMBLE_MARKER
        not in (tmp_path / "miles/backends/megatron_utils/model.py").read_text()
    )
    for rel in files:
        if "model.py" not in rel:
            assert miles.PREAMBLE_MARKER in (tmp_path / rel).read_text()
