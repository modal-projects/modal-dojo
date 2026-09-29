"""Patch slime's ``log_rollout_data`` to also emit per-group advantages.

slime logs only the *mean* advantage per training step (in
``log_rollout_data``, which reduces ``rollout_data["advantages"]`` to a single
number). This patch injects a one-line call to our phase-reporter at the top of
that function so we additionally capture the full per-sample advantage
distribution, tagged with each sample's GRPO prompt group, and ship it to the
dashboard's ``/api/advantage-distributions`` endpoint.

The reporter self-guards (TP-rank-0 / last-PP-stage / CP-rank-0) and is a no-op
when the dashboard status URL is unset, so the injected call is safe on every
rank. Executed at image-build time via ``python3 <this file>``.
"""

from __future__ import annotations

from pathlib import Path

PREAMBLE_MARKER = "PATCHED_MODAL_DOJO_ADVANTAGE_PREAMBLE"
CALL_MARKER = "PATCHED_MODAL_DOJO_ADVANTAGE_DIST"

PREAMBLE = (
    f"# {PREAMBLE_MARKER}: bootstrap advantage-distribution reporter\n"
    "import sys as _tg_sys\n"
    "if '/root' not in _tg_sys.path:\n"
    "    _tg_sys.path.insert(0, '/root')\n"
    "try:\n"
    "    from modal_dojo.frameworks.slime.phase_reporting import (\n"
    "        report_advantage_distribution as _tg_report_advantage_distribution,\n"
    "    )\n"
    "except ImportError:\n"
    "    def _tg_report_advantage_distribution(rollout_id, args, rollout_data): pass\n"
    "\n"
)

# Anchor on the *first* body statement of ``log_rollout_data`` (4-space indent).
# The same guard line recurs deeper inside the function under
# ``if args.log_correct_samples:`` at 8-space indent, so the leading newline +
# 4-space indentation keeps this match unique to the function's top.
ANCHOR = (
    "\n    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():"
    "\n        cp_size = mpu.get_context_parallel_world_size()"
    "\n        log_dict = {}"
)

INJECTION = (
    f"\n    # {CALL_MARKER}: capture the per-group advantage distribution"
    "\n    _tg_report_advantage_distribution(rollout_id, args, rollout_data)"
) + ANCHOR


def _patch_file(path: Path) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping advantage-distribution patch")
        return

    src = path.read_text()
    if CALL_MARKER in src:
        print(f"{path.name} already patched for advantage distribution")
        return

    if ANCHOR not in src:
        print(
            f"WARNING: Could not find log_rollout_data anchor in {path.name}; "
            "skipping advantage-distribution patch"
        )
        return

    src = src.replace(ANCHOR, INJECTION, 1)
    if PREAMBLE_MARKER not in src:
        src = PREAMBLE + src

    path.write_text(src)
    print(f"Patched {path.name} with advantage-distribution reporting")


def main() -> None:
    _patch_file(Path("/root/slime/slime/backends/megatron_utils/data.py"))


if __name__ == "__main__":
    main()
