"""
image: radixark/miles:glm53next
commit: https://github.com/radixark/miles/commit/cc76e23915b2132ecfff65b97331fd83b02637e6
file: train.py, miles/backends/megatron_utils/{actor,model}.py

Adapt Dojo step and substep timing to this driver; the shared patch targets
an older Miles API.
"""

from __future__ import annotations

import ast
from pathlib import Path

MARKER = "PATCHED_MODAL_DOJO_GLM53_TIMING"
PREAMBLE = f"""# {MARKER}
import sys as _dojo_glm_sys
if '/root' not in _dojo_glm_sys.path:
    _dojo_glm_sys.path.insert(0, '/root')
from modal_dojo.common.timing_recorder import (
    recording_lane as _dojo_glm_lane,
    recording_lane_on_reporting_rank as _dojo_glm_actor,
    time_phase as _dojo_glm_phase,
)
from modal_dojo.frameworks.miles.phase_reporting import report_step_event as _dojo_glm_report
"""


def _wrap(source: str, spans: list[tuple[int, int, str]]) -> str:
    lines = source.splitlines(keepends=True)
    for start, end, header in sorted(spans, reverse=True):
        indent = lines[start - 1][
            : len(lines[start - 1]) - len(lines[start - 1].lstrip())
        ]
        body = [
            "    " + line if line.strip() else line for line in lines[start - 1 : end]
        ]
        lines[start - 1 : end] = [
            *(indent + line + "\n" for line in header.splitlines()),
            *body,
        ]
    return "".join(lines)


def _function(source: str, name: str):
    matches = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {name} function, found {len(matches)}")
    return matches[0]


def _wrap_function(source: str, name: str, header: str) -> str:
    node = _function(source, name)
    body = node.body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return _wrap(source, [(body[0].lineno, body[-1].end_lineno, header)])


def _wrap_calls(source: str, mapping: dict[str, str]) -> str:
    spans = []
    found = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Expr, ast.Assign, ast.Return)):
            continue
        value = node.value
        if isinstance(value, ast.Await):
            value = value.value
        if isinstance(value, ast.Call) and (name := ast.unparse(value.func)) in mapping:
            found.add(name)
            header = mapping[name]
            if name == "update_weights" and not value.keywords:
                header = "with _dojo_glm_lane('driver', None), _dojo_glm_phase('initial_weight_sync'):"
            spans.append((node.lineno, node.end_lineno, header))
    if missing := mapping.keys() - found:
        raise ValueError(f"missing GLM timing calls: {sorted(missing)}")
    return _wrap(source, spans)


def patch_source(source: str, target: str) -> str:
    if MARKER in source:
        return source
    if target == "train.py":
        source = _wrap_calls(
            source,
            {
                "create_rollout_components": "with _dojo_glm_lane('driver', None), _dojo_glm_phase('initialize_rollouts'):",
                "create_training_models": "with _dojo_glm_lane('driver', None), _dojo_glm_phase('initialize_train'):",
                "rollout_executor.get.remote": "with _dojo_glm_phase('generate_rollouts'):\n    _dojo_glm_report('generate_rollouts', args, rollout_id)",
                "actor_model.train": "with _dojo_glm_phase('train_models'):",
                "inference_controller.offload": "with _dojo_glm_phase('offload_rollout'):",
                "offload_train": "with _dojo_glm_phase('offload_train'):",
                "save": "with _dojo_glm_phase('checkpoint_save'):",
                "update_weights": "with _dojo_glm_phase('weight_sync'):",
                "eval_dispatcher.dispatch": "with _dojo_glm_phase('evaluate_rollouts'):",
            },
        )
        tree = ast.parse(source)
        loops = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For) and ast.unparse(node.target) == "rollout_id"
        ]
        if len(loops) != 1:
            raise ValueError("expected one driver rollout loop")
        loop = loops[0]
        source = _wrap(
            source,
            [
                (
                    loop.body[0].lineno,
                    loop.body[-1].end_lineno,
                    "with _dojo_glm_lane('driver', rollout_id):",
                )
            ],
        )
    elif target == "actor.py":
        source = _wrap_function(
            source, "compute_log_prob", "with _dojo_glm_phase('compute_log_probs'):"
        )
        source = _wrap_function(
            source,
            "train",
            "with _dojo_glm_actor(rollout_id, 'critic' if self.role == 'critic' else 'actor'):",
        )
    elif target == "model.py":
        source = _wrap_calls(
            source,
            {
                "run_forward_backward_pass": "with _dojo_glm_phase('forward_backward'):",
                "optimizer.step": "with _dojo_glm_phase('optimizer_step'):",
            },
        )
    else:
        raise ValueError(f"unknown GLM timing target: {target}")
    lines = source.splitlines(keepends=True)
    # Keep module docstrings and future imports at the top.
    insertion = 0
    for node in ast.parse(source).body:
        if (isinstance(node, ast.ImportFrom) and node.module == "__future__") or (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            insertion = node.end_lineno
        else:
            break
    lines.insert(insertion, PREAMBLE)
    source = "".join(lines)
    compile(source, target, "exec")
    return source


def main(root: Path = Path("/root/miles")) -> None:
    for relative in (
        "train.py",
        "miles/backends/megatron_utils/actor.py",
        "miles/backends/megatron_utils/model.py",
    ):
        path = root / relative
        path.write_text(patch_source(path.read_text(), path.name))
        print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
