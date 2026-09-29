"""Golden-file test for the Slime rollout-status patcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from modal_dojo.frameworks.slime.modal_helpers.patches import (
    patch_rollout_status_reporting as patcher,
)

TESTDATA = Path(__file__).parent / "testdata" / "slime"


@pytest.fixture(scope="session")
def slime_inputs() -> dict[str, str]:
    inputs = sorted(
        path
        for path in TESTDATA.glob("*.input")
        if path.name in {"train.py.input", "train_async.py.input"}
    )
    assert inputs
    return {path.name.removesuffix(".input"): path.read_text() for path in inputs}


def test_missing_patch_target_is_skipped(tmp_path, capsys):
    missing = tmp_path / "train_async.py"
    patcher._patch_file(missing)
    assert not missing.exists()
    assert "not found, skipping rollout-status patch" in capsys.readouterr().out


# The async loop has no generation/offload/eval step of its own to report
# around, so those anchors are absent there and the patcher says so.
UNREPORTED = {
    "train_async.py": (
        "generate rollout, step start, offload rollout, offload train, "
        "substep start, eval begin"
    )
}


def test_patch_matches_golden(slime_inputs, tmp_path, request, capsys):
    rewrite_goldens = request.config.getoption("--rewrite")
    for name, source in slime_inputs.items():
        golden_name = f"{name}.status.output"
        golden_path = TESTDATA / golden_name
        work = tmp_path / name
        work.write_text(source)
        patcher._patch_file(work)
        actual = work.read_text()
        warned = capsys.readouterr().out
        if name in UNREPORTED:
            assert f"Could not patch {name} for: {UNREPORTED[name]}" in warned
        else:
            assert "WARNING: Could not patch" not in warned

        if rewrite_goldens:
            golden_path.write_text(actual)
            continue

        assert golden_path.exists(), (
            f"Golden output file does not exist: {golden_path}. "
            "Regenerate and review the expected patch output with "
            "uv run pytest tests/test_substep_times.py --rewrite."
        )
        expected = golden_path.read_text()
        assert actual == expected, (
            f"golden mismatch for {name}; rerun with --rewrite to accept"
        )
