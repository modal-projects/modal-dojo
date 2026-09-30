"""Patch slime train entrypoints to report rollout-engine startup status."""

from __future__ import annotations

import re
from pathlib import Path


PREAMBLE_MARKER = "PATCHED_MODAL_DOJO_PREAMBLE"
ROLLOUT_MARKER = "PATCHED_MODAL_DOJO_ROLLOUT_STATUS"
WEIGHT_SYNC_MARKER = "PATCHED_MODAL_DOJO_WEIGHT_SYNC_STATUS"
STEP_START_MARKER = "PATCHED_MODAL_DOJO_STEP_START"
STEP_FINISH_MARKER = "PATCHED_MODAL_DOJO_STEP_FINISH"
GENERATE_ROLLOUT_MARKER = "PATCHED_MODAL_DOJO_GENERATE_ROLLOUT_STATUS"
COMPUTE_LOG_PROBS_MARKER = "PATCHED_MODAL_DOJO_COMPUTE_LOG_PROBS_STATUS"
OFFLOAD_ROLLOUT_MARKER = "PATCHED_MODAL_DOJO_OFFLOAD_ROLLOUT_STATUS"
OFFLOAD_TRAIN_MARKER = "PATCHED_MODAL_DOJO_OFFLOAD_TRAIN_STATUS"
CHECKPOINT_SAVE_MARKER = "PATCHED_MODAL_DOJO_CHECKPOINT_SAVE_STATUS"
SUBSTEP_FINISH_MARKER = "PATCHED_MODAL_DOJO_SUBSTEP_FINISH"
SUBSTEP_START_MARKER = "PATCHED_MODAL_DOJO_SUBSTEP_START"
EVAL_BEGIN_MARKER = "PATCHED_MODAL_DOJO_EVAL_BEGIN"
EVAL_END_MARKER = "PATCHED_MODAL_DOJO_EVAL_END"

PREAMBLE = (
    f"# {PREAMBLE_MARKER}: bootstrap phase reporter (runs once per process)\n"
    "import sys as _tg_sys\n"
    "if '/root' not in _tg_sys.path:\n"
    "    _tg_sys.path.insert(0, '/root')\n"
    "try:\n"
    "    from modal_dojo.frameworks.slime.phase_reporting import (\n"
    "        report_step_event as _tg_report,\n"
    "    )\n"
    "except ImportError:\n"
    "    def _tg_report(status, args=None, rollout_id=None, step_event=''): pass\n"
    "\n"
)


def _patch_file(path: Path) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping rollout-status patch")
        return

    src = path.read_text()
    needs_preamble = PREAMBLE_MARKER not in src
    needs_rollout = ROLLOUT_MARKER not in src
    needs_weight_sync = WEIGHT_SYNC_MARKER not in src
    needs_step_start = STEP_START_MARKER not in src
    needs_step_finish = STEP_FINISH_MARKER not in src
    needs_generate_rollout = GENERATE_ROLLOUT_MARKER not in src
    needs_compute_log_probs = COMPUTE_LOG_PROBS_MARKER not in src
    needs_offload_rollout = OFFLOAD_ROLLOUT_MARKER not in src
    needs_offload_train = OFFLOAD_TRAIN_MARKER not in src
    needs_checkpoint_save = CHECKPOINT_SAVE_MARKER not in src
    needs_substep_finish = SUBSTEP_FINISH_MARKER not in src
    needs_substep_start = SUBSTEP_START_MARKER not in src
    needs_eval_begin = EVAL_BEGIN_MARKER not in src
    needs_eval_end = EVAL_END_MARKER not in src

    if not (
        needs_preamble
        or needs_rollout
        or needs_weight_sync
        or needs_step_start
        or needs_step_finish
        or needs_generate_rollout
        or needs_compute_log_probs
        or needs_offload_rollout
        or needs_offload_train
        or needs_checkpoint_save
        or needs_substep_finish
        or needs_substep_start
        or needs_eval_begin
        or needs_eval_end
    ):
        print(f"{path.name} already patched for rollout status reporting")
        return

    if needs_preamble:
        src = PREAMBLE + src
    elif "report_step_event" not in src:
        src = src.replace(
            "    from modal_dojo.frameworks.slime.phase_reporting import (\n",
            "    from modal_dojo.frameworks.slime.phase_reporting import (\n"
            "        report_step_event as _tg_report,\n",
            1,
        )
        src = src.replace(
            "except ImportError:\n",
            "except ImportError:\n"
            "    def _tg_report(status, args=None, rollout_id=None, step_event=''): pass\n",
            1,
        )

    rollout_count = 0
    if needs_rollout:
        rollout_pattern = re.compile(
            r"^(?P<indent>[ \t]*)rollout_manager, num_rollout_per_epoch = create_rollout_manager\(args, pgs\[\"rollout\"\]\)",
            re.M,
        )

        def _rollout_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return (
                f"{indent}# {ROLLOUT_MARKER}: rollout engine startup state\n"
                f"{indent}_tg_report('initialize_rollouts', args)\n"
                f'{indent}rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])'
            )

        src, rollout_count = rollout_pattern.subn(_rollout_replacement, src, count=1)

    step_start_count = 0
    if needs_step_start:
        step_start_pattern = re.compile(
            r"^(?P<indent>[ \t]*)(?P<line>rollout_data_ref = "
            r"ray\.get\(rollout_manager\.generate\.remote\("
            r"(?P<rollout_id>rollout_id)\)\))[ \t]*$",
            re.M,
        )

        def _step_start_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            line = match.group("line")
            rollout_id = match.group("rollout_id").strip()
            return "\n".join(
                [
                    f"{indent}# {STEP_START_MARKER}: training step start",
                    f"{indent}_tg_report('generate_rollouts', args, {rollout_id}, 'start')",
                    f"{indent}{line}",
                ]
            )

        src, step_start_count = step_start_pattern.subn(_step_start_replacement, src)

    compute_log_probs_count = 0
    if needs_compute_log_probs:
        # compute_log_probs runs inside actor_model.async_train() (in the train
        # actor, which has no rollout_id), so report it from the driver loop —
        # which knows rollout_id — right before the blocking train call.
        compute_log_probs_pattern = re.compile(
            r"^(?P<indent>[ \t]*)(?P<call>ray\.get\(actor_model\.async_train\(.*\))[ \t]*$",
            re.M,
        )

        def _compute_log_probs_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            call = match.group("call")
            return (
                f"{indent}# {COMPUTE_LOG_PROBS_MARKER}: compute log probs state\n"
                f"{indent}_tg_report('compute_log_probs', args, rollout_id)\n"
                f"{indent}{call}"
            )

        src, compute_log_probs_count = compute_log_probs_pattern.subn(
            _compute_log_probs_replacement, src
        )

    offload_rollout_count = 0
    if needs_offload_rollout:
        offload_rollout_pattern = re.compile(
            r"^(?P<indent>[ \t]*)ray\.get\(rollout_manager\.offload\.remote\(\)\)",
            re.M,
        )

        def _offload_rollout_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return (
                f"{indent}# {OFFLOAD_ROLLOUT_MARKER}: rollout offload state\n"
                f"{indent}_tg_report('offload_rollout', args, rollout_id)\n"
                f"{indent}ray.get(rollout_manager.offload.remote())"
            )

        src, offload_rollout_count = offload_rollout_pattern.subn(
            _offload_rollout_replacement, src, count=1
        )

    offload_train_count = 0
    if needs_offload_train:
        offload_train_pattern = re.compile(
            r"^(?P<indent>[ \t]*)(?P<call>offload_train\("
            r"actor_trains(?:_this_step)?\))[ \t]*$",
            re.M,
        )

        def _offload_train_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            call = match.group("call")
            return (
                f"{indent}# {OFFLOAD_TRAIN_MARKER}: train offload state\n"
                f"{indent}_tg_report('offload_train', args, rollout_id)\n"
                f"{indent}{call}"
            )

        src, offload_train_count = offload_train_pattern.subn(
            _offload_train_replacement, src, count=1
        )

    checkpoint_save_count = 0
    if needs_checkpoint_save:
        checkpoint_save_pattern = re.compile(
            r"^(?P<indent>[ \t]*)(?P<guard>if "
            r"(?:release_train or )?should_run_periodic_action\("
            r"[ \t\r\n]*rollout_id,[ \t\r\n]*args\.save_interval,"
            r"[ \t\r\n]*num_rollout_per_epoch,[ \t\r\n]*args\.num_rollout"
            r"[ \t\r\n]*\):)",
            re.M,
        )

        def _checkpoint_save_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            body_indent = f"{indent}    "
            return (
                f"{indent}{match.group('guard')}\n"
                f"{body_indent}# {CHECKPOINT_SAVE_MARKER}: checkpoint save state\n"
                f"{body_indent}_tg_report('checkpoint_save', args, rollout_id)"
            )

        src, checkpoint_save_count = checkpoint_save_pattern.subn(
            _checkpoint_save_replacement, src, count=1
        )

    substep_finish_count = 0
    if needs_substep_finish:
        substep_finish_pattern = re.compile(
            r"^(?P<indent>[ \t]*)if should_run_periodic_action\("
            r"rollout_id, args\.eval_interval, num_rollout_per_epoch\):[ \t]*\n"
            r"(?P<body>[ \t]+ray\.get\(rollout_manager\.eval\.remote\(rollout_id\)\)[ \t]*\n)",
            re.M,
        )

        def _substep_finish_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return (
                f"{match.group(0)}"
                f"{indent}# {SUBSTEP_FINISH_MARKER}: end-of-iteration substep boundary\n"
                f"{indent}_tg_report('weight_sync', args, rollout_id, 'substep_finish')\n"
            )

        src, substep_finish_count = substep_finish_pattern.subn(
            _substep_finish_replacement, src, count=1
        )

    substep_start_count = 0
    if needs_substep_start:
        substep_start_pattern = re.compile(
            r"^(?P<indent>[ \t]*)if args\.eval_interval is not None and "
            r"rollout_id == 0 and not args\.skip_eval_before_train:[ \t]*$",
            re.M,
        )

        def _substep_start_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return (
                f"{indent}# {SUBSTEP_START_MARKER}: substep window start (before eval)\n"
                f"{indent}_tg_report('generate_rollouts', args, rollout_id, 'substep_start')\n"
                f"{match.group(0)}"
            )

        src, substep_start_count = substep_start_pattern.subn(
            _substep_start_replacement, src, count=1
        )

    eval_begin_count = 0
    if needs_eval_begin:
        eval_begin_pattern = re.compile(
            r"^(?P<indent>[ \t]*)if args\.eval_interval is not None and "
            r"rollout_id == 0 and not args\.skip_eval_before_train:[ \t]*\n"
            r"(?P<body>[ \t]+ray\.get\(rollout_manager\.eval\.remote\(rollout_id\)\)[ \t]*\n)",
            re.M,
        )

        def _eval_begin_replacement(match: re.Match[str]) -> str:
            body = match.group("body")
            body_indent = body[: len(body) - len(body.lstrip(" \t"))]
            guard = match.group(0)[: len(match.group(0)) - len(body)]
            return (
                f"{guard}"
                f"{body_indent}# {EVAL_BEGIN_MARKER}: eval-before-train substep start\n"
                f"{body_indent}_tg_report('evaluate_rollouts', args, rollout_id, 'eval_begin')\n"
                f"{body}"
            )

        src, eval_begin_count = eval_begin_pattern.subn(
            _eval_begin_replacement, src, count=1
        )

    eval_end_count = 0
    if needs_eval_end:
        eval_end_pattern = re.compile(
            r"^(?P<indent>[ \t]*)if should_run_periodic_action\("
            r"rollout_id, args\.eval_interval, num_rollout_per_epoch\):[ \t]*\n"
            r"(?P<body>[ \t]+ray\.get\(rollout_manager\.eval\.remote\(rollout_id\)\)[ \t]*\n)",
            re.M,
        )

        def _eval_end_replacement(match: re.Match[str]) -> str:
            body = match.group("body")
            body_indent = body[: len(body) - len(body.lstrip(" \t"))]
            guard = match.group(0)[: len(match.group(0)) - len(body)]
            return (
                f"{guard}"
                f"{body_indent}# {EVAL_END_MARKER}: eval-after-train substep start\n"
                f"{body_indent}_tg_report('evaluate_rollouts', args, rollout_id, 'eval_end')\n"
                f"{body}"
            )

        src, eval_end_count = eval_end_pattern.subn(_eval_end_replacement, src, count=1)

    generate_rollout_count = 0
    if needs_generate_rollout:
        generate_rollout_pattern = re.compile(
            r"^(?P<indent>[ \t]*)(?P<line>rollout_data_ref = "
            r"ray\.get\(rollout_manager\.generate\.remote\("
            r"(?P<rollout_id>rollout_id)\)\))[ \t]*$",
            re.M,
        )

        def _generate_rollout_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            line = match.group("line")
            rollout_id = match.group("rollout_id").strip()
            return (
                f"{indent}# {GENERATE_ROLLOUT_MARKER}: rollout generation state\n"
                f"{indent}_tg_report('generate_rollouts', args, {rollout_id})\n"
                f"{indent}{line}"
            )

        src, generate_rollout_count = generate_rollout_pattern.subn(
            _generate_rollout_replacement, src, count=1
        )

    weight_sync_count = 0
    if needs_weight_sync:
        weight_sync_pattern = re.compile(
            r"^(?P<indent>[ \t]*)(?:[A-Za-z_][A-Za-z0-9_]*\.)?update_weights\(\)[ \t]*(?P<newline>\r?\n?)$"
        )
        lines = src.splitlines(keepends=True)
        patched_lines = []
        in_rollout_loop = False
        loop_indent = ""
        for line in lines:
            loop_match = re.match(
                r"^(?P<indent>[ \t]*)for[ \t]+rollout_id[ \t]+in[ \t]+.*:",
                line,
            )
            if loop_match:
                in_rollout_loop = True
                loop_indent = loop_match.group("indent")
                patched_lines.append(line)
                continue
            if not in_rollout_loop or not line.strip():
                patched_lines.append(line)
                continue
            indent = line[: len(line) - len(line.lstrip(" \t"))]
            if len(indent) <= len(loop_indent):
                in_rollout_loop = False
                patched_lines.append(line)
                continue
            if weight_sync_count == 0:
                weight_sync_match = weight_sync_pattern.match(line)
                if weight_sync_match:
                    newline = weight_sync_match.group("newline") or "\n"
                    patched_lines.extend(
                        [
                            f"{indent}# {WEIGHT_SYNC_MARKER}: weight sync state{newline}",
                            f"{indent}_tg_report('weight_sync', args, rollout_id){newline}",
                        ]
                    )
                    weight_sync_count += 1
            patched_lines.append(line)
        src = "".join(patched_lines)

    step_finish_count = 0
    if needs_step_finish:
        step_finish_pattern = re.compile(
            r"^(?P<indent>[ \t]*)(?P<guard>if "
            r"(?:release_train or )?should_run_periodic_action\("
            r"[ \t\r\n]*rollout_id,[ \t\r\n]*args\.save_interval,"
            r"[ \t\r\n]*num_rollout_per_epoch,[ \t\r\n]*args\.num_rollout"
            r"[ \t\r\n]*\):)",
            re.M,
        )

        def _step_finish_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return (
                f"{indent}# {STEP_FINISH_MARKER}: training step finish\n"
                f"{indent}_tg_report('weight_sync', args, rollout_id, 'finish')\n"
                f"{indent}{match.group('guard')}"
            )

        src, step_finish_count = step_finish_pattern.subn(
            _step_finish_replacement, src, count=1
        )

    failed = []
    if needs_rollout and rollout_count != 1:
        failed.append("rollout init")
    if needs_weight_sync and weight_sync_count != 1:
        failed.append("weight sync")
    if needs_generate_rollout and generate_rollout_count != 1:
        failed.append("generate rollout")
    if needs_compute_log_probs and compute_log_probs_count < 1:
        failed.append("compute log probs")
    if needs_step_finish and step_finish_count != 1:
        failed.append("step finish")
    if needs_step_start and step_start_count == 0:
        failed.append("step start")
    if needs_offload_rollout and offload_rollout_count != 1:
        failed.append("offload rollout")
    if needs_offload_train and offload_train_count != 1:
        failed.append("offload train")
    if needs_checkpoint_save and checkpoint_save_count != 1:
        failed.append("checkpoint save")
    if needs_substep_finish and substep_finish_count != 1:
        failed.append("substep finish")
    if needs_substep_start and substep_start_count != 1:
        failed.append("substep start")
    if needs_eval_begin and eval_begin_count != 1:
        failed.append("eval begin")
    if needs_eval_end and eval_end_count != 1:
        failed.append("eval end")
    if failed:
        print(f"WARNING: Could not patch {path.name} for: {', '.join(failed)}")

    path.write_text(src)
    print(f"Patched {path.name} with rollout status reporting")


def main() -> None:
    _patch_file(Path("/root/slime/train.py"))
    _patch_file(Path("/root/slime/train_async.py"))


if __name__ == "__main__":
    main()
