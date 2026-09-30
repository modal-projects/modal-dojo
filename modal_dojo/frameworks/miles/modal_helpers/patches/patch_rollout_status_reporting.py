"""Patch miles train entrypoints to report training-loop phase status.

Executed at image-build time via ``python3 <this file>``.
"""

from __future__ import annotations

import re
from pathlib import Path

PREAMBLE_MARKER = "PATCHED_MODAL_DOJO_PREAMBLE"

PREAMBLE = (
    f"# {PREAMBLE_MARKER}: bootstrap phase reporter (runs once per process)\n"
    "import sys as _tg_sys\n"
    "if '/root' not in _tg_sys.path:\n"
    "    _tg_sys.path.insert(0, '/root')\n"
    "try:\n"
    "    from modal_dojo.frameworks.miles.phase_reporting import (\n"
    "        report_step_event as _tg_report,\n"
    "    )\n"
    "except ImportError:\n"
    "    def _tg_report(status, args=None, rollout_id=None): pass\n"
    "\n"
)

# Each entry: (marker, phase, rollout-id expression, line regex). The matched
# line is prefixed with a marker comment + a ``_tg_report`` call at the same
# indent. Anchors target miles' await-style driver loop.
_LINE_INJECTIONS: list[tuple[str, str, str, re.Pattern[str]]] = [
    (
        "PATCHED_MODAL_DOJO_ROLLOUT_STATUS",
        "initialize_rollouts",
        "None",
        re.compile(
            r"^(?P<indent>[ \t]*)(?P<line>rollout_manager, num_rollout_per_epoch = "
            r"create_rollout_manager\(args, pgs\[\"rollout\"\]\))[ \t]*$",
            re.M,
        ),
    ),
    (
        "PATCHED_MODAL_DOJO_GENERATE_ROLLOUT_STATUS",
        "generate_rollouts",
        "rollout_id",
        re.compile(
            r"^(?P<indent>[ \t]*)(?P<line>rollout_data\w* = await "
            r"rollout_manager\.generate\.remote\(rollout_id\))[ \t]*$",
            re.M,
        ),
    ),
    (
        "PATCHED_MODAL_DOJO_COMPUTE_LOG_PROBS_STATUS",
        # actor_model.train() first recomputes log probs in the train actor
        # (which has no rollout_id), so report the phase from the driver loop.
        "compute_log_probs",
        "rollout_id",
        re.compile(
            r"^(?P<indent>[ \t]*)(?P<line>await actor_model\.train\("
            r"rollout_id, rollout_data[^\n]*\))[ \t]*$",
            re.M,
        ),
    ),
    (
        "PATCHED_MODAL_DOJO_OFFLOAD_ROLLOUT_STATUS",
        "offload_rollout",
        "rollout_id",
        re.compile(
            r"^(?P<indent>[ \t]*)(?P<line>await rollout_manager\.offload\.remote\("
            r"[^\n]*\))[ \t]*$",
            re.M,
        ),
    ),
    (
        "PATCHED_MODAL_DOJO_OFFLOAD_TRAIN_STATUS",
        "offload_train",
        "rollout_id",
        re.compile(
            r"^(?P<indent>[ \t]*)(?P<line>await offload_train\(\))[ \t]*$",
            re.M,
        ),
    ),
    (
        "PATCHED_MODAL_DOJO_WEIGHT_SYNC_STATUS",
        "weight_sync",
        "rollout_id",
        re.compile(
            r"^(?P<indent>[ \t]*)(?P<line>await actor_model\.update_weights\("
            r"rollout_id=rollout_id\))[ \t]*$",
            re.M,
        ),
    ),
]

CHECKPOINT_SAVE_MARKER = "PATCHED_MODAL_DOJO_CHECKPOINT_SAVE_STATUS"
_CHECKPOINT_SAVE_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)(?P<guard>if "
    r"(?:external_save or )?should_run_periodic_action\("
    r"[ \t\r\n]*rollout_id,[ \t\r\n]*args\.save_interval,"
    r"[ \t\r\n]*num_rollout_per_epoch,[ \t\r\n]*args\.num_rollout"
    r"[ \t\r\n]*\):)",
    re.M,
)


def _patch_file(path: Path) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping rollout-status patch")
        return

    src = path.read_text()
    failed: list[str] = []

    if PREAMBLE_MARKER not in src:
        src = PREAMBLE + src

    for marker, phase, rollout_id_expr, pattern in _LINE_INJECTIONS:
        if marker in src:
            continue

        def _replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            line = match.group("line")
            return (
                f"{indent}# {marker}: {phase} state\n"
                f"{indent}_tg_report('{phase}', args, {rollout_id_expr})\n"
                f"{indent}{line}"
            )

        src, count = pattern.subn(_replacement, src)
        if count == 0:
            failed.append(phase)

    if CHECKPOINT_SAVE_MARKER not in src:

        def _checkpoint_save_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            body_indent = f"{indent}    "
            return (
                f"{indent}{match.group('guard')}\n"
                f"{body_indent}# {CHECKPOINT_SAVE_MARKER}: checkpoint save state\n"
                f"{body_indent}_tg_report('checkpoint_save', args, rollout_id)"
            )

        src, count = _CHECKPOINT_SAVE_PATTERN.subn(
            _checkpoint_save_replacement, src, count=1
        )
        if count == 0:
            failed.append("checkpoint save")

    if failed:
        print(f"WARNING: Could not patch {path.name} for: {', '.join(failed)}")

    path.write_text(src)
    print(f"Patched {path.name} with rollout status reporting")


def main() -> None:
    _patch_file(Path("/root/miles/train.py"))
    _patch_file(Path("/root/miles/train_async.py"))


if __name__ == "__main__":
    main()
