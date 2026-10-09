"""Golden-file tests for the miles rollout-status and advantage patchers."""

from __future__ import annotations

from pathlib import Path

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_advantage_distribution as advantage_patcher,
    patch_qkvr_cpu_merge as qkvr_patcher,
    patch_rollout_status_reporting as rollout_patcher,
)

TESTDATA = Path(__file__).parent / "testdata" / "miles"

STATUS_PATCH_FILES = ("train.py", "train_async.py", "log_utils.py")


@pytest.fixture(scope="session")
def miles_inputs() -> dict[str, str]:
    inputs = [TESTDATA / f"{name}.input" for name in STATUS_PATCH_FILES]
    assert inputs
    return {path.name.removesuffix(".input"): path.read_text() for path in inputs}


def test_all_miles_goldens_compile():
    for golden in TESTDATA.glob("*.output"):
        compile(golden.read_text(), str(golden), "exec")


def _apply_patcher(name: str, work: Path) -> None:
    if name == "log_utils.py":
        advantage_patcher._patch_file(work)
    else:
        rollout_patcher._patch_file(work)


def test_missing_patch_targets_are_skipped(tmp_path, capsys):
    missing = tmp_path / "train.py"
    rollout_patcher._patch_file(missing)
    assert not missing.exists()
    assert "not found, skipping rollout-status patch" in capsys.readouterr().out

    advantage_patcher._patch_file(missing)
    assert not missing.exists()
    assert "not found, skipping advantage-distribution patch" in capsys.readouterr().out


def test_patch_matches_golden(miles_inputs, tmp_path, request):
    rewrite_goldens = request.config.getoption("--rewrite")
    for name, source in miles_inputs.items():
        golden_path = TESTDATA / (
            f"{name}.advantage.output"
            if name == "log_utils.py"
            else f"{name}.status.output"
        )
        work = tmp_path / name
        work.write_text(source)
        _apply_patcher(name, work)
        actual = work.read_text()

        if rewrite_goldens:
            golden_path.write_text(actual)
            continue

        assert golden_path.exists(), (
            f"Golden output file does not exist: {golden_path}. "
            "Regenerate and review the expected patch output with "
            "uv run pytest tests/test_miles_patches.py --rewrite."
        )
        expected = golden_path.read_text()
        assert actual == expected, (
            f"golden mismatch for {name}; rerun with --rewrite to accept"
        )


def test_qkvr_cpu_merge_rewrites_factory(tmp_path, capsys):
    # Indent matches the pinned miles image (nested under `if qkey in sd`).
    work = tmp_path / "layers.py"
    work.write_text(
        "class InklingSelfAttention:\n"
        "    def sharded_state_dict(self):\n"
        "        if True:\n"
        "            def _qkvr_merge(sub):\n"
        "                return torch.cat(list(sub), dim=0)\n"
        "\n"
        "            return _qkvr_merge\n"
    )

    assert qkvr_patcher.apply(work) == 1
    patched = work.read_text()
    assert qkvr_patcher.MARKER in patched
    assert "return torch.cat(list(sub), dim=0)" not in patched
    assert "resize_" not in patched
    assert ".to(device)" not in patched
    compile(patched, str(work), "exec")
    assert qkvr_patcher.apply(work) == 0
    assert "already applied" in capsys.readouterr().out
