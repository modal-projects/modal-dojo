"""Separate inference readiness from transfer time in the pinned Miles sync path.

Source: miles@41c5e38b94ea23677de93b01a4a77d55677a8f09.
Preserve Dojo's existing dashboard timing contexts around these operations.
"""

import ast
from pathlib import Path

TARGET = Path("/root/miles/miles/ray/train/group.py")
MARKER = "DOJO_WEIGHT_SYNC"
OPERATIONS = (
    ("prepare_engines", "self._inference_controller.start_update_weights"),
    ("transfer", "retry"),
    ("finish_engines", "self._inference_controller.end_update_weights"),
)


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        return
    functions = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "update_weights"
    ]
    if len(functions) != 1:
        raise RuntimeError(f"{target}: expected one weight-sync function")
    statements = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, (ast.Assign, ast.Expr))
        and isinstance(node.value, ast.Await)
        and isinstance(node.value.value, ast.Call)
    ]
    edits = []
    lines = source.splitlines(keepends=True)
    for phase, call in OPERATIONS:
        matches = [
            node for node in statements if ast.unparse(node.value.value.func) == call
        ]
        if len(matches) != 1:
            raise RuntimeError(f"{target}: weight-sync source changed at {phase}")
        node = matches[0]
        start, end = node.lineno - 1, node.end_lineno
        indent = " " * node.col_offset
        old = lines[start:end]
        new = [
            indent + "import time as _dojo_time\n",
            indent + "_dojo_started = _dojo_time.monotonic()\n",
            indent + "_dojo_ok = False\n",
            indent + "try:\n",
            *("    " + line for line in old),
            indent + "    _dojo_ok = True\n",
            indent + "finally:\n",
            indent + f'    logger.info("{MARKER} phase={phase} ok=%s elapsed_s=%.3f", '
            "_dojo_ok, _dojo_time.monotonic() - _dojo_started)\n",
        ]
        edits.append((start, end, new))
    for start, end, new in sorted(edits, reverse=True):
        lines[start:end] = new
    result = "".join(lines)
    compile(result, str(target), "exec")
    target.write_text(result)


if __name__ == "__main__":
    apply()
