"""Patch slime/Miles ``train.py`` to save and stop when a ``Budget`` runs out.

Before the save check, a spent budget makes the current rollout the last one
(forcing a synchronous save); the loop then breaks. The loop also iterates
``run_steps.rollout_ids`` instead of ``range``, so ``modal-dojo run set-steps``
can change ``num_rollout`` while the run trains. Apply after other patches.
"""

from __future__ import annotations

import ast
import base64
from pathlib import Path

MARKER = "PATCHED_MODAL_DOJO_BUDGET_STOP"

PREAMBLE = (
    f"# {MARKER}: stop on a modal_dojo Budget\n"
    "import sys as _md_sys\n"
    "if '/root' not in _md_sys.path:\n"
    "    _md_sys.path.insert(0, '/root')\n"
    "try:\n"
    "    from modal_dojo.common.budget import budget_final as _md_budget_final\n"
    "    from modal_dojo.common.run_steps import rollout_ids as _md_rollout_ids\n"
    "except ImportError:\n"
    "    print('WARNING: modal_dojo not importable; training budgets off')\n"
    "\n"
    "    def _md_budget_final(args, rollout_id):\n"
    "        return False\n"
    "\n"
    "    def _md_rollout_ids(args, start):\n"
    "        return range(start, args.num_rollout)\n"
)


def _is_rollout_loop(node: ast.AST) -> bool:
    if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Call):
        return False
    call = node.iter
    return (
        isinstance(call.func, ast.Name)
        and call.func.id == "range"
        and any(
            isinstance(arg, ast.Attribute) and arg.attr == "num_rollout"
            for arg in call.args
        )
    )


def _is_save_check(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    for call in ast.walk(node.test):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "should_run_periodic_action"
            and len(call.args) > 1
            and isinstance(call.args[1], ast.Attribute)
            and call.args[1].attr == "save_interval"
        ):
            return True
    return False


def patch_source(src: str) -> str:
    if MARKER in src:
        return src
    tree = ast.parse(src)
    loop = next((n for n in ast.walk(tree) if _is_rollout_loop(n)), None)
    if loop is None:
        raise RuntimeError("budget patch: no `for rollout_id in range(...)` loop")
    save = next((n for n in ast.walk(loop) if _is_save_check(n)), None)
    if save is None:
        raise RuntimeError("budget patch: no save_interval check in the loop")
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not imports:
        raise RuntimeError("budget patch: no top-level imports to follow")

    call = loop.iter
    if (
        len(call.args) != 2
        or not isinstance(call.args[1], ast.Attribute)
        or call.lineno != call.end_lineno
    ):
        raise RuntimeError("budget patch: expected range(start, args.num_rollout)")
    start = ast.get_source_segment(src, call.args[0])
    args_name = ast.get_source_segment(src, call.args[1].value)

    lines = src.splitlines(keepends=True)
    loop_header = lines[call.lineno - 1]
    lines[call.lineno - 1] = (
        loop_header[: call.col_offset]
        + f"_md_rollout_ids({args_name}, {start})"
        + loop_header[call.end_col_offset :]
    )
    body = " " * loop.body[0].col_offset
    indent = " " * save.col_offset
    header = lines[save.lineno - 1]
    if not header[save.col_offset :].startswith("if "):
        raise RuntimeError("budget patch: unexpected save check layout")

    lines.insert(
        loop.body[-1].end_lineno,
        f"{body}if _md_final:  # {MARKER}\n{body}    break\n",
    )
    lines[save.lineno - 1] = (
        f"{header[: save.col_offset]}if _md_final or {header[save.col_offset + 3 :]}"
    )
    lines.insert(
        save.lineno - 1,
        f"{indent}_md_final = _md_budget_final(args, rollout_id)  # {MARKER}\n"
        f"{indent}if _md_final:\n"
        f"{indent}    args.num_rollout = rollout_id + 1\n",
    )
    lines.insert(loop.body[0].lineno - 1, f"{body}_md_final = False  # {MARKER}\n")
    lines.insert(max(n.end_lineno for n in imports), PREAMBLE)
    patched = "".join(lines)
    ast.parse(patched)
    return patched


def main(root: str) -> None:
    for name in ("train.py", "train_async.py"):
        path = Path(root) / name
        if not path.exists():
            continue
        path.write_text(patch_source(path.read_text()))
        print(f"Patched {path} to stop on a training budget")


def encoded_patch(root: str) -> str:
    script = Path(__file__).read_text() + f"\n\nmain({root!r})\n"
    return base64.b64encode(script.encode()).decode()


def patch_command(root: str, *, required: bool) -> str:
    """Image build command applying this patch; fatal on failure only if ``required``.

    A run with a ``Budget`` needs the patch to stop; any other run only loses
    ``modal-dojo run set-steps``, so a source layout the patch does not recognize
    warns instead of failing the build.
    """
    command = f"echo {encoded_patch(root)} | base64 -d | python3"
    if required:
        return command
    return (
        f"{command} || echo 'WARNING: rollout loop patch did not apply;"
        " modal-dojo run set-steps will not affect this run'"
    )
