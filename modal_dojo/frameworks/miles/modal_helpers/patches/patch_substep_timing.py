"""Wrap miles' driver loop, rollout worker and actor with measured substep timing.

Slime has the same script over its own source: the lanes, phase names and record
format are the two frameworks' shared contract, so a change to one belongs in
the other (``frameworks/slime/modal_helpers/patches/patch_substep_timing.py``).

Puts each phase of the training loop inside ``with _tg_rec.phase(...)``. The
driver's phases all sit in one loop body, with a local (``_tg_rec``) recorder.
The rollout worker and the actor measure work further down the call stack, so
they open a lane at their entry point and the phases below use the module-level
``_tg_time_phase``, which records on whichever lane that process has open.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

PREAMBLE_MARKER = "PATCHED_TRAINING_GYM_TIMING_PREAMBLE"
RECORDER_MARKER = "PATCHED_TRAINING_GYM_TIMING_RECORDER"


def phase_marker(phase: str) -> str:
    return f"PATCHED_TRAINING_GYM_TIMING_{phase.upper()}"


PREAMBLE = (
    f"# {PREAMBLE_MARKER}: bootstrap substep-timing recorder\n"
    "import sys as _tg_sys\n"
    "if '/root' not in _tg_sys.path:\n"
    "    _tg_sys.path.insert(0, '/root')\n"
    "try:\n"
    "    from modal_dojo.common.timing_recorder import (\n"
    "        RoleRecorder as _TgRecorder,\n"
    "        recording_lane as _tg_role,\n"
    "        recording_lane_on_reporting_rank as _tg_mrec,\n"
    "        time_phase as _tg_time_phase,\n"
    "    )\n"
    "except ImportError:\n"
    "    print('WARNING: modal_dojo not importable; substep timing off')\n"
    "    from contextlib import contextmanager as _tg_cm\n"
    "\n"
    "    class _TgRecorder:\n"
    "        def __init__(self, role, rollout_id): pass\n"
    "\n"
    "        def __enter__(self): return self\n"
    "\n"
    "        def __exit__(self, *exc): pass\n"
    "\n"
    "        @_tg_cm\n"
    "        def phase(self, name):\n"
    "            yield\n"
    "\n"
    "    @_tg_cm\n"
    "    def _tg_role(role, rollout_id):\n"
    "        yield _TgRecorder(role, rollout_id)\n"
    "\n"
    "    @_tg_cm\n"
    "    def _tg_mrec(rollout_id, role='actor'):\n"
    "        yield _TgRecorder(role, rollout_id)\n"
    "\n"
    "    @_tg_cm\n"
    "    def _tg_time_phase(name):\n"
    "        yield\n"
    "\n"
)


def _inject_preamble(src: str) -> str:
    tree = ast.parse(src)
    end_line = 0
    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            end_line = node.end_lineno or node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            end_line = node.end_lineno or node.lineno
        else:
            break
    lines = src.splitlines(keepends=True)
    return "".join(lines[:end_line]) + PREAMBLE + "".join(lines[end_line:])


ROOT = Path("/root/miles")


def replace_once(source: str, old: str, new: str, path: Path) -> str:
    """Replace exactly one occurrence, or raise: anchors like
    ``actor_model.update_weights()`` appear more than once, and patching the
    wrong one is worse than failing the build.
    """
    positions = []
    start = 0
    while (index := source.find(old, start)) != -1:
        positions.append(index)
        start = index + 1
    if any(index and source[index - 1] != "\n" for index in positions):
        raise RuntimeError(f"{path}: anchor {old!r} does not start at a line boundary")
    if len(positions) != 1:
        raise RuntimeError(
            f"{path}: expected 1 occurrence of {old!r}, found {len(positions)}"
        )
    return source.replace(old, new, 1)


def indent_block(block: str) -> str:
    return "\n".join(f"    {ln}" if ln.strip() else ln for ln in block.splitlines())


def wrap_block(block: str, phase: str, opener: str = "_tg_rec.phase") -> str:
    """Wrap a block in ``with <opener>('<phase>'):``.

    For a bare ``if``, only the body is wrapped, so a skipped branch records
    nothing instead of a ~0s bar for work that never ran. An ``if/else`` is
    wrapped whole, since wrapping one arm alone would not parse.
    """
    lines = block.splitlines(keepends=True)
    outer = len(lines[0]) - len(lines[0].lstrip(" "))
    head, body = "", block
    if lines[0].lstrip().startswith("if "):
        header = next(i for i, ln in enumerate(lines) if ln.rstrip().endswith(":")) + 1
        has_dedent_to_outer = any(
            ln.strip() and len(ln) - len(ln.lstrip(" ")) == outer
            for ln in lines[header:]
        )
        if not has_dedent_to_outer:
            head, body = "".join(lines[:header]), "".join(lines[header:])
    indent = body[: len(body) - len(body.lstrip(" "))]
    return (
        head
        + f"{indent}# {phase_marker(phase)}\n"
        + f"{indent}with {opener}('{phase}'):\n{indent_block(body)}\n"
    )


def _wrap_driver_loop(src: str, path: Path) -> str:
    """Wrap the driver ``for rollout_id in range(...)`` body in a recording lane."""
    lines = src.splitlines(keepends=True)
    matches = [
        (i, line)
        for i, line in enumerate(lines)
        if "for rollout_id in range(args.start_rollout_id, args.num_rollout):" in line
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{path}: expected 1 driver rollout loop, found {len(matches)}"
        )
    i, line = matches[0]

    loop_indent = line[: len(line) - len(line.lstrip(" "))]
    start = i + 1
    j = start
    while j < len(lines):
        body_line = lines[j]
        if body_line.strip() == "":
            j += 1
            continue
        body_indent = body_line[: len(body_line) - len(body_line.lstrip(" "))]
        if len(body_indent) <= len(loop_indent):
            break
        j += 1

    with_line = f"{loop_indent}    with _tg_role('driver', rollout_id) as _tg_rec:\n"
    marker = f"{loop_indent}    # {RECORDER_MARKER}: driver lane active\n"
    new_body = []
    for body_line in lines[start:j]:
        if body_line.strip():
            new_body.append("    " + body_line)
        else:
            new_body.append(body_line)
    new_lines = lines[: i + 1] + [marker, with_line] + new_body + lines[j:]
    return "".join(new_lines)


def _wrap_bootstrap_sync(src: str, path: Path) -> str:
    anchor = (
        "    # always update weight first so that sglang has the loaded weights from training.\n"
        "    await actor_model.update_weights()\n"
    )
    replacement = (
        "    # always update weight first so that sglang has the loaded weights from training.\n"
        "    with _tg_role('driver', None) as _tg_rec:\n"
        "        with _tg_rec.phase('initial_weight_sync'):\n"
        "            await actor_model.update_weights()\n"
    )
    return replace_once(src, anchor, replacement, path)


@dataclass(frozen=True)
class PackageTarget:
    """One file in the framework package that measures a non-driver lane.

    ``scope`` opens the recorder: the first line of the function that knows the
    rollout id, the line ending its body, and the ``with`` header to insert.
    ``blocks`` are the phases; they may sit in a different module from the scope
    (``forward_backward`` is in ``model.py``, its recorder in ``actor.py``), so
    they are wrapped with the module-level ``_tg_time_phase``.
    """

    path: str
    scope: tuple[str, str, str] | None
    blocks: tuple[tuple[str, str], ...]


# status patcher runs first, so the anchors are the upstream lines.

_SYNC_PHASE_WRAPS = [
    (
        "        if args.eval_interval is not None and rollout_id == args.start_rollout_id and not args.skip_eval_before_train:\n"
        "            await rollout_manager.eval.remote(rollout_id)\n",
        "evaluate_rollouts",
    ),
    (
        "        rollout_data_pack = await rollout_manager.generate.remote(rollout_id)\n",
        "generate_rollouts",
    ),
    (
        "            await rollout_manager.offload.remote(tags=offload_tags)\n",
        "offload_rollout",
    ),
    (
        "            values = await critic_model.train(rollout_id, rollout_data_pack)\n",
        "train_models",
    ),
    (
        "                await actor_model.train(rollout_id, rollout_data_pack, external_data=values)\n",
        "train_models",
    ),
    (
        "            await actor_model.train(rollout_id, rollout_data_pack)\n",
        "train_models",
    ),
    (
        "            await save(rollout_id, force_sync=external_save)\n",
        "checkpoint_save",
    ),
    ("        await offload_train()\n", "offload_train"),
    ("                await critic_model.offload()\n", "offload_train"),
    ("                    await actor_model.offload()\n", "offload_train"),
    ("            await rollout_manager.onload_weights.remote()\n", "weight_sync"),
    (
        "        await actor_model.update_weights(rollout_id=rollout_id)\n",
        "weight_sync",
    ),
    ("            await rollout_manager.onload_kv.remote()\n", "weight_sync"),
    (
        "        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):\n"
        "            await rollout_manager.eval.remote(rollout_id)\n",
        "evaluate_rollouts_end",
    ),
]

_ASYNC_PHASE_WRAPS = [
    (
        "        if rollout_data_next_future is not None:\n"
        "            rollout_data_curr_ref = await rollout_data_next_future\n",
        "wait_for_rollout",
    ),
    (
        "            values = await critic_model.train(rollout_id, rollout_data_curr_ref)\n",
        "train_models",
    ),
    (
        "                await actor_model.train(rollout_id, rollout_data_curr_ref, external_data=values)\n",
        "train_models",
    ),
    (
        "            await actor_model.train(rollout_id, rollout_data_curr_ref)\n",
        "train_models",
    ),
    ("                await critic_model.offload()\n", "offload_train"),
    ("                    await actor_model.offload()\n", "offload_train"),
    (
        "            await save_training_model(actor_model, rollout_id, force_sync)\n",
        "checkpoint_save",
    ),
    (
        "                await save_training_model(critic_model, rollout_id, force_sync)\n",
        "checkpoint_save",
    ),
    # As in slime's async loop: the prefetched future is consumed before the
    # weight update, and measured apart from it. The future here is the *next*
    # rollout's generation, not this one's, so it is its own phase.
    (
        "            # sync generate before update weights to prevent update weight in the middle of generation\n"
        "            rollout_data_curr_ref = (await x) if (x := rollout_data_next_future) is not None else None\n"
        "            rollout_data_next_future = None\n",
        "wait_for_next_rollout",
    ),
    # Indented: the bring-up call before the loop is at module-function level.
    (
        "            await actor_model.update_weights(rollout_id=rollout_id)\n",
        "weight_sync",
    ),
]

_ASYNC_EVAL_BEFORE = (
    "    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:\n"
    "        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)\n"
)
_ASYNC_EVAL_AFTER = (
    "        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout):\n"
    "            await eval_dispatcher.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)\n"
)

ENTRYPOINTS = {
    "train.py": _SYNC_PHASE_WRAPS,
    "train_async.py": _ASYNC_PHASE_WRAPS,
}

PACKAGE_TARGETS: tuple[PackageTarget, ...] = (
    PackageTarget(
        path="miles/ray/rollout/rollout_manager.py",
        scope=(
            "    async def generate(self, rollout_id):\n",
            "        return dict(sample_indices=sample_indices, data_ref=data_ref)\n",
            "with _tg_role('rollout', rollout_id):",
        ),
        blocks=(
            (
                "generate_samples",
                '        with timer("rollout"):\n'
                "            data, metadata, metrics = await self._get_rollout_data(rollout_id=rollout_id)\n",
            ),
            (
                "reward_post_process",
                "        data = convert_samples_to_train_data(\n"
                "            self.args,\n"
                "            data,\n"
                "            metadata=metadata,\n"
                "            custom_convert_samples_to_train_data_func=self.custom_convert_samples_to_train_data_func,\n"
                "            custom_reward_post_process_func=self.custom_reward_post_process_func,\n"
                "        )\n",
            ),
        ),
    ),
    PackageTarget(
        path="miles/rollout/rm_hub/__init__.py",
        scope=None,
        blocks=(
            (
                "reward_batch",
                "    if args.custom_rm_path is not None and not is_multi_lora_enabled(args):\n"
                "        rm_function = load_function(args.custom_rm_path)\n"
                "        return await rm_function(args, samples, **kwargs)\n"
                "    tasks = [async_rm(args, sample, **kwargs) for sample in samples]\n"
                "    rewards = await asyncio.gather(*tasks)\n"
                "    return rewards\n",
            ),
        ),
    ),
    PackageTarget(
        path="miles/rollout/sglang_rollout.py",
        scope=None,
        blocks=(
            (
                "sample_generation",
                "            generate_fn = load_generate_function(custom_func_path) if custom_func_path else None\n"
                "            if generate_fn is not None:\n"
                "                output = await generate_fn(\n"
                "                    GenerateFnInput(state=state, sample=sample, sampling_params=sampling_params, evaluation=evaluation)\n"
                "                )\n"
                "                sample = output.samples\n"
                "            else:\n"
                "                sample = await generate(args, sample, sampling_params)\n",
            ),
            (
                "reward",
                "        if sample.reward is None:\n"
                "            sample.reward = await async_rm(args, sample)\n",
            ),
        ),
    ),
    PackageTarget(
        path="miles/backends/megatron_utils/actor.py",
        scope=(
            "    def train(\n"
            "        self,\n"
            "        rollout_id: int,\n"
            "        rollout_data_ref: Box,\n"
            "        witness_info: WitnessInfo | None = None,\n"
            "        attempt: int = 0,\n"
            "        external_data=None,\n"
            "    ):\n",
            "            return result\n",
            "with _tg_mrec(rollout_id, 'critic' if self.role == 'critic' else 'actor'):",
        ),
        blocks=(
            (
                "compute_log_probs",
                '        with timer(f"{store_prefix}log_probs"):\n'
                "            return forward_only(\n"
                "                get_log_probs_and_entropy,\n"
                "                self.args,\n"
                "                self.model,\n"
                "                data_iterator,\n"
                "                num_microbatches,\n"
                "                rollout_id=rollout_id,\n"
                "                store_prefix=store_prefix,\n"
                "            )\n",
            ),
            (
                "trainer_finalize",
                "        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)\n"
                "\n"
                "        for m in all_replay_managers:\n"
                "            if m.enabled:\n"
                "                m.clear_all()\n"
                "\n"
                "        if train_step_outcome == TrainStepOutcome.NORMAL:\n"
                "            # update the cpu actor weight to the latest model\n"
                "            if self._enable_weight_backup:\n"
                '                self.weights_backuper.backup("actor")\n'
                "            else:\n"
                "                torch.cuda.synchronize()\n"
                "\n"
                "            # Update ref model if needed\n"
                "            if (\n"
                "                self.args.ref_update_interval is not None\n"
                "                and (rollout_id + 1) % self.args.ref_update_interval == 0\n"
                '                and "ref" in self.weights_backuper.backup_tags\n'
                "            ):\n"
                '                with timer("ref_model_update"):\n'
                "                    if is_first_replica_megatron_main_rank():\n"
                '                        logger.info(f"Updating ref model at rollout_id {rollout_id}")\n'
                '                    self.weights_backuper.backup("ref")\n'
                "\n"
                "        if train_step_outcome == TrainStepOutcome.NORMAL and is_multi_lora_enabled(self.args):\n"
                "            from miles.backends.megatron_utils.multi_lora_utils import commit_trained_batch\n"
                "\n"
                "            commit_trained_batch(rollout_data, rollout_id, self._multi_lora_pending_push)\n"
                "\n"
                "        log_perf_data(rollout_id, self.args, extra_metrics=self.weight_updater.pop_metrics())\n"
                "\n"
                "        self._heartbeat.bump()\n",
            ),
        ),
    ),
    PackageTarget(
        path="miles/backends/megatron_utils/model.py",
        scope=None,
        blocks=(
            (
                "forward_backward",
                "    losses_reduced = forward_backward_func(\n",
            ),
            (
                "optimizer_step",
                "            update_successful, grad_norm, num_zeros_in_grad = optimizer.step()\n",
            ),
            (
                "train_step_finalize",
                "    # release grad (multi-LoRA retains accumulated grads; stepped slots were\n"
                "    # zeroed selectively inside step_adapter_slots)\n"
                "    if not multi_lora:\n"
                "        _zero_grads(model, optimizer, disable_optimizer)\n"
                "\n"
                "    log_structured(\n"
                "        logger.info,\n"
                '        op="train_step",\n'
                "        rollout=rollout_id,\n"
                "        step=step_id,\n"
                "        attempt=attempt,\n"
                "        outcome=outcome.name,\n"
                "        valid_step=valid_step,\n"
                "    )\n"
                "\n"
                "    if outcome == TrainStepOutcome.NORMAL:\n"
                "        dump_local_weight_checksums(args=args, model=model, optimizer=optimizer)\n"
                "        if args.enable_witness:\n"
                "            witness_dump_and_clear_stale(model=model, witness_info=witness_info, optimizer=optimizer)\n"
                "\n"
                "        if mpu.is_pipeline_last_stage(ignore_virtual=True):\n"
                "            metric_num_rollouts = None if args.calculate_per_token_loss else num_rollouts\n"
                "            loss_reduced = (\n"
                "                indep_dp_loss_reduced\n"
                "                if parallel_state.indep_dp.size > 1\n"
                "                else aggregate_train_losses(losses_reduced, metric_num_rollouts)\n"
                "            )\n"
                "            return loss_reduced, grad_norm, outcome\n"
                "\n"
                "    return {}, grad_norm, outcome\n",
            ),
        ),
    ),
)


def wrap_scope(src: str, scope: tuple[str, str, str], path: Path) -> str:
    """Put a function body inside the recorder's ``with``, anchored on its first
    and last lines: the phases inside it are already rewritten by now.
    """
    signature, last_line, header = scope
    if src.count(signature) != 1:
        raise RuntimeError(
            f"{path}: expected 1 occurrence of {signature.strip()!r}, "
            f"found {src.count(signature)}"
        )
    head, _, rest = src.partition(signature)
    body, sep, tail = rest.partition(last_line)
    if not sep:
        raise RuntimeError(f"{path}: scope end not found: {last_line.strip()!r}")
    indent = " " * (len(signature) - len(signature.lstrip(" ")) + 4)
    # An earlier line with the same text would close the lane early
    after = next((ln for ln in tail.splitlines() if ln.strip()), "")
    if after.startswith(indent):
        raise RuntimeError(
            f"{path}: {last_line.strip()!r} is not the end of "
            f"{signature.strip()!r}; {after.strip()!r} follows it"
        )
    return (
        head
        + signature
        + f"{indent}# {RECORDER_MARKER}\n"
        + f"{indent}{header}\n"
        + indent_block(body + last_line)
        + "\n"
        + tail
    )


def _patch_package_file(root: Path, target: PackageTarget) -> None:
    path = root / target.path
    if not path.exists():
        raise RuntimeError(f"{path}: not found; {root.name} layout changed")

    src = path.read_text()
    if PREAMBLE_MARKER in src:
        print(f"{target.path} already patched for substep timing")
        return

    for phase, block in target.blocks:
        src = replace_once(src, block, wrap_block(block, phase, "_tg_time_phase"), path)
    if target.scope is not None:
        src = wrap_scope(src, target.scope, path)  # last: it reindents the body
    src = _inject_preamble(src)

    missing = [phase for phase, _ in target.blocks if phase_marker(phase) not in src]
    if missing:
        raise RuntimeError(f"{path}: phases not instrumented: {missing}")
    if target.scope is not None and RECORDER_MARKER not in src:
        raise RuntimeError(f"{path}: recorder scope not instrumented")
    compile(src, str(path), "exec")

    path.write_text(src)
    print(f"Patched {target.path} for substep timing ({len(target.blocks)} phases)")


def patch_package_file(root: Path, target: PackageTarget) -> None:
    """Best-effort instrumentation for optional framework package files."""
    try:
        _patch_package_file(root, target)
    except Exception as exc:
        print(f"WARNING: {target.path} substep timing patch skipped: {exc}")


def _patch_file(path: Path, wraps: list[tuple[str, str]]) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping substep timing patch")
        return

    src = path.read_text()
    if PREAMBLE_MARKER in src:
        print(f"{path.name} already patched for substep timing")
        return

    src = _inject_preamble(src)
    if path.name == "train_async.py":
        src = replace_once(
            src,
            _ASYNC_EVAL_BEFORE,
            (
                "    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:\n"
                "        if not args.eval_uses_snapshots:\n"
                "            with _tg_role('driver', None) as _tg_rec:\n"
                "                # PATCHED_TRAINING_GYM_TIMING_EVALUATE_ROLLOUTS\n"
                "                with _tg_rec.phase('evaluate_rollouts'):\n"
                "                    await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)\n"
                "        else:\n"
                "            await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)\n"
            ),
            path,
        )
        src = replace_once(
            src,
            _ASYNC_EVAL_AFTER,
            (
                "        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout):\n"
                "            if not args.eval_uses_snapshots:\n"
                "                # PATCHED_TRAINING_GYM_TIMING_EVALUATE_ROLLOUTS_END\n"
                "                with _tg_rec.phase('evaluate_rollouts_end'):\n"
                "                    await eval_dispatcher.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)\n"
                "            else:\n"
                "                await eval_dispatcher.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)\n"
            ),
            path,
        )
    for old, phase in wraps:
        src = replace_once(src, old, wrap_block(old, phase), path)
    if path.name == "train_async.py":
        drain = "    await eval_dispatcher.drain()\n"
        replacement = (
            "    with _tg_role('driver', None) as _tg_rec:\n"
            "        # PATCHED_TRAINING_GYM_TIMING_EVALUATE_ROLLOUTS_END\n"
            "        with _tg_rec.phase('evaluate_rollouts_end'):\n"
            "            await eval_dispatcher.drain()\n"
        )
        src = replace_once(src, drain, replacement, path)
    src = _wrap_bootstrap_sync(src, path)
    src = _wrap_driver_loop(src, path)  # last: it reindents the loop body

    missing = [phase for _, phase in wraps if phase_marker(phase) not in src]
    if path.name == "train_async.py":
        for phase in ("evaluate_rollouts", "evaluate_rollouts_end"):
            if phase_marker(phase) not in src:
                missing.append(phase)
    if missing:
        raise RuntimeError(f"{path}: phases not instrumented: {missing}")
    compile(src, str(path), "exec")

    path.write_text(src)
    print(f"Patched {path.name} for substep timing ({len(wraps)} phases)")


def _patch_entrypoint(path: Path, wraps: list[tuple[str, str]]) -> None:
    try:
        _patch_file(path, wraps)
    except Exception as exc:
        print(f"WARNING: {path} substep timing patch skipped: {exc}")


def main() -> None:
    """Patch this image's framework checkout, if it has one."""
    if not ROOT.is_dir():
        return
    for name, wraps in ENTRYPOINTS.items():
        _patch_entrypoint(ROOT / name, wraps)
    for target in PACKAGE_TARGETS:
        patch_package_file(ROOT, target)


if __name__ == "__main__":
    main()
