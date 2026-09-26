"""Invariants of the cross-framework model validation registry."""

from __future__ import annotations

import json

import pytest

from modal_training_gym.common.models.qwen3_0_6b import Qwen3_0_6B
from modal_training_gym.common.models.validation import (
    VALIDATION_CONFIGS,
    Framework,
    _ValidationConfig,
)
from scripts.diff_impact import (
    FRAMEWORK_VALIDATION_HARNESS_PATHS,
    REPO_ROOT,
    affected_models,
)
from scripts.validation_backends import build_recipe_and_dataset


def _diff_touching(*repo_relative_paths: str) -> str:
    """A minimal unified diff naming the given paths."""
    return "\n".join(
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}"
        for path in repo_relative_paths
    )


ALL_CONFIGS = _ValidationConfig.select()


def test_registry_has_both_frameworks_represented():
    frameworks = {config.framework for config in VALIDATION_CONFIGS}
    assert frameworks == set(Framework)


@pytest.mark.parametrize(
    "name",
    ["Qwen3.5-4B-SFT", "Moonlight-16B-A3B-Instruct-SFT"],
)
def test_sft_validation_entry_builds_sft_recipe(name):
    config = _ValidationConfig.find(name)
    recipe, dataset = build_recipe_and_dataset(
        config.framework,
        config.model_config(),
        step_count=1,
        loss_type=config.loss_type,
    )
    assert recipe.loss_type == "sft_loss"
    assert dataset.hf_repo == "HuggingFaceH4/no_robots"


def test_registry_uses_the_packages_one_framework_enum():
    """A second same-valued enum would fail every dispatch branch.

    ``build_recipe_and_dataset`` and ``diff_impact._base_recipe_for`` both
    dispatch on identity, so a caller holding ``common.framework.Framework``
    would fall through to "no validation backend" if the registry declared its
    own copy — equal by value, a different class.
    """
    from modal_training_gym.common.framework import Framework as CanonicalFramework

    assert Framework is CanonicalFramework
    recipe, dataset = build_recipe_and_dataset(
        CanonicalFramework.SLIME, Qwen3_0_6B(), step_count=1
    )
    assert recipe is not None and dataset is not None


def test_registry_names_are_unique():
    """Every model/framework pair needs an unambiguous dispatch name."""
    seen: dict[str, _ValidationConfig] = {}
    for config in ALL_CONFIGS:
        key = config.name.lower()
        other = seen.setdefault(key, config)
        assert other is config, f"duplicate validation name: {config.name!r}"


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.name)
def test_config_resolves_by_name_case_insensitively(config):
    assert _ValidationConfig.find(config.name) is config
    assert _ValidationConfig.find(config.name.upper()) is config
    assert _ValidationConfig.find(f"  {config.name.lower()} ") is config


POLICY_CONFIGS = [c for c in ALL_CONFIGS if c.loss_type == "policy_loss"]


@pytest.mark.parametrize("config", POLICY_CONFIGS, ids=lambda c: c.name)
def test_config_resolves_by_hf_repo_id(config):
    """``check -m Qwen/Qwen3-4B`` must keep working, not just the short name."""
    matches = [c for c in POLICY_CONFIGS if c.model_name == config.model_name]
    if len(matches) == 1:
        assert _ValidationConfig.find(config.model_name) is config
        assert _ValidationConfig.find(config.model_name.upper()) is config
    else:
        with pytest.raises(ValueError, match="ambiguous model"):
            _ValidationConfig.find(config.model_name)


def test_unknown_model_names_are_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        _ValidationConfig.find("not-a-real-model")


@pytest.mark.parametrize(
    ("name", "recipe_cls_name"),
    [
        ("Inkling-Small", "Inkling_Small_Recipe"),
        ("Inkling-Small-LoRA", "Inkling_Small_LoRA_Recipe"),
        ("Qwen3.6-35B-A3B", "Qwen3_6_35B_Recipe"),
    ],
)
def test_named_validation_resolves_expected_recipe(name, recipe_cls_name):
    """``check -m`` must pick the registry variant, not only the model default."""
    config = _ValidationConfig.find(name)
    recipe, _ = build_recipe_and_dataset(config.framework, config.model_config(), 1)
    assert type(recipe).__name__ == recipe_cls_name


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.name)
def test_every_config_builds_a_recipe_on_its_declared_framework(config):
    """The declared framework must actually have a base recipe for the model.

    This is the check that a wrong ``framework`` value would otherwise only
    fail on a GPU, minutes into a run.
    """
    recipe, dataset = build_recipe_and_dataset(
        config.framework, config.model_config(), step_count=1
    )
    assert recipe is not None
    assert dataset is not None
    assert recipe.rm_type, f"{config.name} validation recipe has no rm_type"


def test_list_prints_every_registered_model():
    """``list`` and ``list --names-only`` both emit the whole registry."""
    from scripts.validate_model_configs import available_model_names, available_models

    names = {config.name for config in VALIDATION_CONFIGS}
    listed = available_models()

    assert set(available_model_names()) == names
    assert {row["name"] for row in listed} == names


def test_blank_dispatch_lists_every_registered_model():
    """The workflow's blank-models branch must emit the full registry."""
    import re

    workflow = (REPO_ROOT / ".github/workflows/validate-models.yml").read_text()
    unnarrowed = re.findall(
        r"validate_model_configs\.py list(?! --names-only)", workflow
    )

    assert "validate_model_configs.py list --names-only" in workflow
    assert not unnarrowed, "a `list` in the workflow is missing --names-only"


def test_baselines_are_fetched_only_for_models_that_ran(tmp_path):
    """Baseline lookup follows the results directory, not the registry.

    Widening ``available_model_names`` to the whole registry would otherwise
    have the PR comment job scan artifacts for models that did not run.
    """
    from scripts.download_perf_baseline import models_with_results

    (tmp_path / "validate-result-Qwen3-4B.json").write_text(
        json.dumps({"base_model_name": "Qwen3-4B", "succeeded": True})
    )

    assert models_with_results(tmp_path) == ["Qwen3-4B"]
    assert models_with_results(tmp_path / "missing") == []


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.name)
def test_validation_dataset_unpickles_without_the_scripts_directory(config, tmp_path):
    """Every backend dataset must survive the trip into a training container.

    Backend modules live under ``scripts/``, which is absent from the training
    image, so a dataset pickled by reference crashes remotely during data
    preparation. Unpickling in a subprocess whose ``sys.path`` has no
    ``scripts/`` entry reproduces that container exactly.

    Pickled with ``modal._serialization``, which is what actually ships the
    dataset: it uses Modal's vendored cloudpickle, whose by-value registry is
    separate from the installed cloudpickle's.
    """
    import base64
    import subprocess
    import sys
    import textwrap

    from modal._serialization import serialize

    from scripts.validate_model_configs import _ship_dataset_definition

    _, dataset = build_recipe_and_dataset(config.framework, config.model_config(), 1)

    _ship_dataset_definition(dataset)
    payload = serialize(dataset)

    # Blocking the import is a truer stand-in for the image than trimming
    # sys.path: modal_training_gym is installed from this same tree, so the
    # entry that makes the backends importable is the one it needs too.
    probe = textwrap.dedent("""
        import base64, pickle, sys

        class Blocked:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("scripts", "validation_backends"):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        sys.meta_path.insert(0, Blocked())
        obj = pickle.loads(base64.b64decode(sys.argv[1]))
        print(type(obj).__name__)
    """)
    result = subprocess.run(
        [sys.executable, "-c", probe, base64.b64encode(payload).decode()],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, (
        f"{config.name} dataset does not survive into a training container:\n"
        f"{result.stderr}"
    )
    assert result.stdout.strip() == type(dataset).__name__


def test_framework_harness_change_only_revalidates_that_framework():
    """A miles-only change must not re-run the slime set, and vice versa."""
    slime_models = {c.name for c in _ValidationConfig.select(Framework.SLIME)}
    miles_models = {c.name for c in _ValidationConfig.select(Framework.MILES)}

    slime_paths = FRAMEWORK_VALIDATION_HARNESS_PATHS["slime"]
    diff = _diff_touching(*(str(p.relative_to(REPO_ROOT)) for p in slime_paths))
    selected = set(affected_models(diff))

    assert selected == slime_models
    assert not selected & (miles_models - slime_models)


def test_shared_harness_change_revalidates_every_registered_model():
    names = {c.name for c in VALIDATION_CONFIGS}
    diff = _diff_touching("scripts/validate_model_configs.py")

    assert set(affected_models(diff)) == names


def test_framework_change_does_not_narrow_shared_harness_impact():
    names = {c.name for c in VALIDATION_CONFIGS}
    miles_launcher = next(iter(FRAMEWORK_VALIDATION_HARNESS_PATHS["miles"]))
    diff = _diff_touching(
        "scripts/validate_model_configs.py",
        str(miles_launcher.relative_to(REPO_ROOT)),
    )

    assert set(affected_models(diff)) == names


def test_framework_change_preserves_shared_class_impact():
    names = {c.name for c in VALIDATION_CONFIGS}
    miles_launcher = next(iter(FRAMEWORK_VALIDATION_HARNESS_PATHS["miles"]))
    diff = _diff_touching(
        "modal_training_gym/common/models/base.py",
        str(miles_launcher.relative_to(REPO_ROOT)),
    )

    assert set(affected_models(diff)) == names
