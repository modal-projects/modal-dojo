"""Extend native Miles' complete adapter/optimizer checkpoints with RNG and sampler resume."""

from pathlib import Path


def replace_once(text, anchor, replacement):
    if text.count(anchor) != 1:
        raise ValueError(f"Native Miles checkpoint anchor changed: {anchor!r}")
    return text.replace(anchor, replacement)


def apply(root=Path("/root/miles/miles")):
    root = Path(root)
    path = root / "backends/megatron_utils/lora/utils.py"
    text = path.read_text()
    marker = "from modal_dojo.frameworks.miles.native_lora_state import capture_resume_state, restore_resume_state"
    if marker not in text:
        text = marker + "\n" + text
        text = replace_once(
            text,
            '                "iteration": iteration,\n',
            '                "iteration": iteration,\n                "dojo_resume_state": capture_resume_state(),\n',
        )
        anchor = '    return training_state["iteration"]\n'
        text = replace_once(
            text,
            anchor,
            '    if not getattr(args, "no_load_rng", False):\n'
            '        restore_resume_state(training_state.get("dojo_resume_state", training_state.get("tau_resume_state")))\n'
            + anchor,
        )
        compile(text, str(path), "exec")
        path.write_text(text)

    # Fresh base-weight initialization sets no_load_optim/no_load_rng=True.
    # Preserve the caller's adapter-resume intent across that separate load.
    path = root / "backends/megatron_utils/megatron_config.py"
    text = path.read_text()
    marker = "from modal_dojo.frameworks.miles.native_lora_state import preserve_resume_flags"
    if marker not in text:
        anchor = "def resolve_args_checkpoint_load(args: Namespace) -> None:\n"
        text = replace_once(
            text, anchor, anchor + f"    {marker}\n    preserve_resume_flags(args)\n"
        )
        compile(text, str(path), "exec")
        path.write_text(text)

    path = root / "backends/megatron_utils/model.py"
    text = path.read_text()
    marker = "from modal_dojo.frameworks.miles.native_lora_state import prepare_native_resume"
    if marker not in text:
        anchor = '    if is_lora_enabled(args) and role == "actor" and args.lora_adapter_path is not None:\n'
        text = replace_once(
            text,
            anchor,
            anchor + f"        {marker}\n        prepare_native_resume(args)\n",
        )
        # restore_lora_adapter already loads the scheduler's consumed-sample
        # counter. The generic iteration-based increment would count it twice.
        anchor = "    if opt_param_scheduler is not None and not (args.use_checkpoint_opt_param_scheduler and iteration > 0):\n"
        text = replace_once(
            text,
            anchor,
            "    if opt_param_scheduler is not None and not getattr(args, 'dojo_native_scheduler_restored', False) and not (args.use_checkpoint_opt_param_scheduler and iteration > 0):\n",
        )
        compile(text, str(path), "exec")
        path.write_text(text)

    path = root / "backends/megatron_utils/lora/utils.py"
    text = path.read_text()
    marker = "        args.dojo_native_scheduler_restored = True\n"
    if marker not in text:
        anchor = '        opt_param_scheduler.load_state_dict(training_state["opt_param_scheduler"])\n'
        text = replace_once(text, anchor, anchor + marker)
        compile(text, str(path), "exec")
        path.write_text(text)

    path = root / "ray/rollout/rollout_executor.py"
    text = path.read_text()
    marker = (
        "from modal_dojo.frameworks.miles.native_lora_state import resume_run_directory"
    )
    if marker not in text:
        anchor = '        if self.args.load is None:\n            logger.warning("no --load: the rollout side starts fresh")'
        text = replace_once(
            text,
            anchor,
            f"        {marker}\n"
            "        resume_directory = resume_run_directory(self.args, rollout_id)\n"
            '        if resume_directory is None:\n            logger.warning("no --load: the rollout side starts fresh")',
        )
        text = replace_once(
            text,
            "        directory = compute_rollout_checkpoint_dir(self.args.load, rollout_id=rollout_id)\n",
            "        directory = compute_rollout_checkpoint_dir(resume_directory, rollout_id=rollout_id)\n",
        )
        compile(text, str(path), "exec")
        path.write_text(text)

    # Loading the frozen base forces start_rollout_id=0 before the separate
    # native adapter is restored. Use the adapter's actual next rollout so the
    # coordinator also restores its sampler and does not replay batch zero.
    path = root / "ray/placement_group.py"
    text = path.read_text()
    marker = 'if (x := args.start_rollout_id) is None or getattr(args, "lora_adapter_path", None):'
    if marker not in text:
        text = replace_once(text, "if (x := args.start_rollout_id) is None:", marker)
        compile(text, str(path), "exec")
        path.write_text(text)


if __name__ == "__main__":
    apply()
