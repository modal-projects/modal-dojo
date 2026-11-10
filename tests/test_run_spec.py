"""``RunSpec`` is how a run's configuration reaches its Modal containers.

The spec is the single place Modal Dojo pickles user objects, so these tests
pin down what has to survive the trip: instances of classes the user defined
in their own source tree, which the image cannot import.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import pathlib
import subprocess
import sys
import textwrap

import pytest

from modal_dojo.common.dataset import HuggingFaceDataset, _SftDataset
from modal_dojo.common.errors import DojoError
from modal_dojo.common.framework import is_user_module
from modal_dojo.common.run_spec import RunSpec, _user_modules
from modal_dojo.train_recipes.slime_recipe import SlimeRecipe

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

USER_PACKAGE = {
    "__init__.py": "",
    "text.py": 'SUFFIX = "!"\n\n\ndef shout(text: str) -> str:\n    return text.upper() + SUFFIX\n',
    "data.py": textwrap.dedent(
        """
        from modal_dojo.common.dataset import DatasetConfig

        from .text import shout


        class UserDataset(DatasetConfig):
            def __init__(self, n_rows: int) -> None:
                self.n_rows = n_rows

            def input_key(self) -> str:
                return "prompt"

            def label_key(self) -> str:
                return "label"

            def rows(self):
                for index in range(self.n_rows):
                    yield {"prompt": shout(f"row {index}"), "label": str(index)}
        """
    ),
}


@pytest.fixture
def user_dataset_cls(tmp_path, monkeypatch):
    """A dataset class in a package from the user's source tree, whose
    ``rows()`` depends on a sibling module."""
    package = tmp_path / "user_project_pkg"
    package.mkdir()
    for name, source in USER_PACKAGE.items():
        (package / name).write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("user_project_pkg.data")
    yield module.UserDataset
    for name in [n for n in sys.modules if n.split(".")[0] == "user_project_pkg"]:
        del sys.modules[name]


def _spec(**overrides) -> RunSpec:
    fields = dict(
        training_run_id="run-1",
        app_name="app",
        recipe=None,
        model=None,
        dataset=None,
        eval_dataset=None,
        dataset_path="/data/train.jsonl",
        eval_dataset_path=None,
        checkpoint_dir="/checkpoints/run-1",
        hf_cache_volume_name="huggingface-cache",
        data_volume_name="data",
        checkpoints_volume_name="checkpoints",
        checkpoints_mount_path="/checkpoints",
    )
    return RunSpec(**{**fields, **overrides})


def _unpack_in_container(payload: dict, expression: str, cwd: pathlib.Path) -> object:
    """Unpack ``payload`` in a process that, like a training image, can import
    Modal Dojo but nothing from the user's source tree."""
    probe = textwrap.dedent(
        f"""
        import base64, json, sys

        class NoUserCode:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("user_project_pkg", "scripts"):
                    raise ModuleNotFoundError(f"No module named {{name!r}}")
                return None

        sys.meta_path.insert(0, NoUserCode())

        from modal_dojo.common.run_spec import RunSpec

        payload = json.loads(sys.argv[1])
        payload["spec"] = base64.b64decode(payload["spec"])
        spec = RunSpec.unpack(payload)
        print(json.dumps({expression}))
        """
    )
    wire = {**payload, "spec": base64.b64encode(payload["spec"]).decode()}
    result = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(wire)],
        capture_output=True,
        text=True,
        cwd=cwd,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": ""},
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_user_class_and_its_sibling_module_travel_by_value(
    user_dataset_cls, tmp_path
) -> None:
    payload = _spec(dataset=user_dataset_cls(n_rows=2)).pack()

    rows = _unpack_in_container(payload, "list(spec.dataset.rows())", tmp_path)

    assert rows == [
        {"prompt": "ROW 0!", "label": "0"},
        {"prompt": "ROW 1!", "label": "1"},
    ]


def test_user_class_wrapped_by_the_launcher_is_still_found(
    user_dataset_cls, tmp_path
) -> None:
    # sft_loss runs wrap the user's dataset before it reaches the spec.
    payload = _spec(dataset=_SftDataset(user_dataset_cls(n_rows=1))).pack()

    labels = _unpack_in_container(
        payload, "[row['label'] for row in spec.dataset._inner.rows()]", tmp_path
    )

    assert labels == ["0"]


def test_plain_fields_round_trip() -> None:
    spec = _spec(group_id="group-7", eval_dataset_path="/data/eval.jsonl")

    assert RunSpec.unpack(spec.pack()) == spec


def test_user_modules_are_registered_by_top_level_package(user_dataset_cls) -> None:
    modules = _user_modules(_spec(dataset=user_dataset_cls(n_rows=1)))

    assert {module.__name__ for module in modules} == {"user_project_pkg"}


def test_package_and_installed_classes_are_not_shipped_by_value() -> None:
    import pydantic

    dataset = HuggingFaceDataset("org/repo", input_column="q", output_column="a")

    assert _user_modules(_spec(dataset=dataset, recipe=SlimeRecipe())) == set()
    assert not is_user_module(sys.modules["modal_dojo.common.dataset"])
    assert not is_user_module(pydantic)
    assert not is_user_module(json)
    assert not is_user_module(sys)


def test_unpack_explains_a_python_version_mismatch() -> None:
    payload = {"python": "3.9", "spec": b"not a pickle"}

    with pytest.raises(DojoError, match=r"launched from Python 3\.9.*image runs"):
        RunSpec.unpack(payload)


def test_unpack_reraises_other_failures_unchanged() -> None:
    payload = _spec().pack()
    payload["spec"] = b"not a pickle"

    with pytest.raises(Exception) as excinfo:
        RunSpec.unpack(payload)

    assert not isinstance(excinfo.value, DojoError)
