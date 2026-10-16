import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from modal.call_graph import InputStatus

source = ast.parse(Path(__file__).with_name("launch_h200_validation.py").read_text())
namespace = {}
exec(
    compile(
        ast.Module(
            body=[
                n
                for n in source.body
                if isinstance(n, ast.FunctionDef)
                and n.name in {"pending_calls", "checkpoint_wait_reason"}
            ],
            type_ignores=[],
        ),
        "<gate>",
        "exec",
    ),
    namespace,
)
pending_calls = namespace["pending_calls"]
checkpoint_wait_reason = namespace["checkpoint_wait_reason"]


def node(call, status, *children):
    return SimpleNamespace(
        function_call_id=call, status=status, children=list(children)
    )


def test_failed_parent_does_not_hide_pending_retried_conversion():
    conversion = node("new-conversion", InputStatus.PENDING)
    graph = [
        node(
            "launch",
            InputStatus.FAILURE,
            node("old-conversion", InputStatus.TERMINATED),
            conversion,
        )
    ]
    assert pending_calls(graph, "new-conversion") == [conversion]


def test_completed_conversion_does_not_hide_nested_pending_training():
    training = node("train", InputStatus.PENDING)
    graph = [
        node(
            "launch",
            InputStatus.SUCCESS,
            node("conversion", InputStatus.SUCCESS),
            node("other-launch", InputStatus.SUCCESS, training),
        )
    ]
    assert pending_calls(graph, "conversion") == [training]


def test_all_terminal_calls_allow_progress():
    assert (
        pending_calls(
            [
                node(
                    "launch",
                    InputStatus.FAILURE,
                    node("conversion", InputStatus.FAILURE),
                )
            ],
            "conversion",
        )
        == []
    )


def test_missing_conversion_is_not_assumed_complete():
    with pytest.raises(RuntimeError, match="absent"):
        pending_calls([node("launch", InputStatus.SUCCESS)], "conversion")


@pytest.mark.parametrize(
    "holder,expected_wait",
    [
        (None, False),
        ({"run_id": "owner", "claimed_at": 0}, False),
        ({"run_id": "owner", "claimed_at": 950}, True),
        ({"run_id": "other", "claimed_at": 0}, True),
        ({"run_id": "owner"}, True),
    ],
)
def test_lock_is_preserved_until_safe_to_proceed(holder, expected_wait):
    assert bool(checkpoint_wait_reason(holder, "owner", 1000, 900)) is expected_wait
