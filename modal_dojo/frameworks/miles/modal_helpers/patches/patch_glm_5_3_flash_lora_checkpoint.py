"""
image: radixark/miles:dev-202610180955
commit: https://github.com/radixark/miles/commit/24bd5fb7166a185f309399e52f43b100e4369203
file: miles/backends/training_utils/checkpoint/io.py,
      miles/backends/megatron_utils/model.py,
      miles/ray/rollout/rollout_executor.py

Create native adapter directories in every node's Modal Volume view and
commit all rank shards. Native LoRA bypasses Megatron's checkpoint hook.
On resume, restore rollout state beside the adapter, not the frozen base,
and do not advance the already-restored scheduler a second time.
TODO: remove when Miles supports these native save/resume paths directly.
"""

from pathlib import Path

MARKER = "PATCHED_MODAL_DOJO_NATIVE_LORA_CHECKPOINT"


def patch_source(source: str, target: str = "io.py") -> str:
    if MARKER in source:
        return source
    if target == "rollout_executor.py":
        anchor = "        directory = compute_rollout_checkpoint_dir(self.args.load, rollout_id=rollout_id)"
        replacement = """        load_root = self.args.load
        if self.args.megatron_to_hf_mode == "raw" and self.args.lora_adapter_path is not None:
            # Native checkpoints live at <run>/iter_<id>/adapter; --load selects the frozen base.
            adapter = Path(self.args.lora_adapter_path)
            if adapter.name != "adapter" or not adapter.parent.name.startswith("iter_"):
                raise ValueError("Native LoRA resume requires <run>/iter_<id>/adapter")
            load_root = str(adapter.parent.parent)
        directory = compute_rollout_checkpoint_dir(load_root, rollout_id=rollout_id)"""
        if source.count(anchor) != 1:
            raise ValueError("Miles rollout checkpoint loader changed")
        source = source.replace(anchor, replacement, 1)
        return source + f"\n# {MARKER}\n"
    if target == "model.py":
        anchor = '    if is_lora_enabled(args) and role == "actor" and args.lora_adapter_path is not None:'
        scheduler = "    if opt_param_scheduler is not None and not (args.use_checkpoint_opt_param_scheduler and iteration > 0):"
        if source.count(anchor) != 1 or source.count(scheduler) != 1:
            raise ValueError("Miles native adapter loader changed")
        source = source.replace(anchor, "    adapter_iteration = None\n" + anchor, 1)
        source = source.replace(
            scheduler,
            "    if adapter_iteration is None and opt_param_scheduler is not None and not (args.use_checkpoint_opt_param_scheduler and iteration > 0):",
            1,
        )
        return source + f"\n# {MARKER}\n"
    if target != "io.py":
        raise ValueError(f"Unknown native checkpoint target: {target}")
    anchor = "        write_shards(checkpoint_dir)"
    if source.count(anchor) != 1 or source.count("def write_checkpoint_dir(") != 1:
        raise ValueError("Miles native checkpoint writer changed")
    source = source.replace(
        anchor,
        "        checkpoint_dir.mkdir(parents=True, exist_ok=True)\n" + anchor,
    ).replace("def write_checkpoint_dir(", "def _dojo_write_checkpoint_dir(", 1)
    source += f"""

# {MARKER}
def write_checkpoint_dir(*args, **kwargs):
    from modal_dojo.common.torch_dist_checkpoint import commit_latest_checkpoint_volume

    result = _dojo_write_checkpoint_dir(*args, **kwargs)
    commit_latest_checkpoint_volume()
    return result
"""
    compile(source, "io.py", "exec")
    return source


def main() -> None:
    root = Path("/root/miles/miles")
    for relative in (
        "backends/training_utils/checkpoint/io.py",
        "backends/megatron_utils/model.py",
        "ray/rollout/rollout_executor.py",
    ):
        path = root / relative
        source = patch_source(path.read_text(), path.name)
        compile(source, str(path), "exec")
        path.write_text(source)
        print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
