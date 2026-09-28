from pathlib import Path
import runpy
from unittest.mock import patch

import pytest

FIXTURES = Path(__file__).parent / "testdata/miles/kimi_k3"
PATCH = str(
    Path(__file__).resolve().parents[1]
    / "modal_training_gym/frameworks/miles/modal_helpers/patches/patch_cell_tick_timeout.py"
)


def test_cell_timeout_snapshot_and_idempotence(tmp_path):
    target = tmp_path / "inference_controller.py"
    target.write_text((FIXTURES / "inference_controller.py.input").read_text())
    with patch("pathlib.Path", return_value=target):
        runpy.run_path(PATCH, run_name="__main__")
    expected = (FIXTURES / "inference_controller.py.output").read_text()
    assert target.read_text() == expected
    compile(expected, str(target), "exec")
    with patch("pathlib.Path", return_value=target), pytest.raises(SystemExit) as exc:
        runpy.run_path(PATCH, run_name="__main__")
    assert exc.value.code == 0
    assert target.read_text() == expected


def test_cell_timeout_rejects_source_drift(tmp_path):
    target = tmp_path / "inference_controller.py"
    target.write_text("CELL_TIMEOUT = 120.0\n")
    with (
        patch("pathlib.Path", return_value=target),
        pytest.raises(SystemExit, match="did not match"),
    ):
        runpy.run_path(PATCH, run_name="__main__")
    assert target.read_text() == "CELL_TIMEOUT = 120.0\n"
