"""Exercise the actual Miles 41c5e38 rollout metric function without GPUs."""

import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches.patch_zero_std_metrics import (
    patch_source,
)

FIXTURE = Path(__file__).parent / "testdata/miles/k3_metrics/metrics.py.input"


def group_by(items, key=lambda item: item):
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return groups


def compute(groups, estimator="grpo"):
    patched = patch_source(FIXTURE.read_text())
    fn = next(
        n
        for n in ast.parse(patched).body
        if isinstance(n, ast.FunctionDef) and n.name == "_compute_zero_std_metrics"
    )
    namespace = {"Sample": object, "group_by": group_by}
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), str(FIXTURE), "exec"), namespace
    )
    samples = [
        SimpleNamespace(group_index=i, get_reward_value=lambda args, r=reward: r)
        for i, rewards in enumerate(groups)
        for reward in rewards
    ]
    return namespace[fn.name](SimpleNamespace(advantage_estimator=estimator), samples)


def test_patch_matches_pinned_golden_is_idempotent_and_rejects_drift():
    patched = patch_source(FIXTURE.read_text())
    assert patched == FIXTURE.with_suffix(".output").read_text()
    assert patch_source(patched) == patched
    compile(patched, str(FIXTURE), "exec")
    with pytest.raises(ValueError, match="anchor changed"):
        patch_source("def _compute_zero_std_metrics(): pass")


@pytest.mark.parametrize("cast", [int, float])
def test_acute_group_counts_produce_actual_percentages(cast):
    groups = [[cast(0)] * 8] * 3 + [[cast(1)] * 8] * 2 + [[cast(0), cast(1)] * 4] * 3
    result = compute(groups)
    assert result == {
        "zero_std/count_0.0": 3,
        "zero_std/count_1.0": 2,
        "zero_std/all_zero_percentage": 3 / 8,
        "zero_std/all_one_percentage": 2 / 8,
    }


def test_mixed_numeric_types_share_buckets_and_continuous_rewards_remain_visible():
    result = compute([[0, 0.0], [0.0, 0.0], [1, 1.0], [0.5, 0.5], [0, 1]])
    assert result["zero_std/count_0.0"] == 2
    assert result["zero_std/count_0.5"] == 1
    assert result["zero_std/all_zero_percentage"] == 2 / 5
    assert result["zero_std/all_one_percentage"] == 1 / 5
    assert compute([]) == {}
    assert compute([[0, 0]], estimator="ppo") == {}


def test_patch_is_applied_by_shared_miles_launcher():
    from modal_dojo.frameworks.miles.launcher import (
        _PATCH_ZERO_STD_B64,
        _REPORTING_PATCH_COMMANDS,
    )

    assert any(_PATCH_ZERO_STD_B64 in command for command in _REPORTING_PATCH_COMMANDS)
