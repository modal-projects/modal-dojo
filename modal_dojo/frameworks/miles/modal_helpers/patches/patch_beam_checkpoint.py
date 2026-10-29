"""
image: radixark/miles:v0.1.2
commit: https://github.com/radixark/miles/commit/819cfd23ab74de9975931597894500632dd708fc
file: miles/backends/training_utils/checkpoint/io.py,
      miles/backends/megatron_utils/{megatron_config,model,checkpoint,lora/utils}.py,
      miles/ray/rollout/rollout_executor.py, miles/rollout/data_source.py,
      train_async.py

Commit native adapter shards on every node, restore the optimizer and scheduler,
and publish a resumable checkpoint only after rollout state has been saved.
Save the next untrained rollout's cursor and buffered samples, excluding
the async producer's in-flight prefetch from the committed dataset position.
TODO: Remove once pinned Miles supports native bridge LoRA resume on Modal.
"""

from pathlib import Path

MARKER = "MODAL_DOJO_BEAM_CHECKPOINT"


def patch_source(source: str, target: str) -> str:
    if MARKER in source:
        return source

    def replace(old: str, new: str) -> None:
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"Miles checkpoint anchor changed: {target}: {old!r}")
        source = source.replace(old, new, 1)

    if target == "io.py":
        replace(
            "        write_shards(checkpoint_dir)",
            "        checkpoint_dir.mkdir(parents=True, exist_ok=True)\n        write_shards(checkpoint_dir)",
        )
        replace("def write_checkpoint_dir(", "def _dojo_write_checkpoint_dir(")
        source += """

def write_checkpoint_dir(*args, **kwargs):
    from modal_dojo.common.torch_dist_checkpoint import commit_latest_checkpoint_volume
    result = _dojo_write_checkpoint_dir(*args, **kwargs)
    commit_latest_checkpoint_volume()
    return result
"""
    elif target == "utils.py":
        replace(
            "    adapter_dir = Path(adapter_path).resolve()",
            """    adapter_dir = Path(adapter_path).resolve()
    from modal_dojo.frameworks.miles.modal_helpers.beam_checkpoint import MARKER
    manifest = json.loads((adapter_dir.parent / MARKER).read_text())
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if manifest["world_size"] != world_size:
        raise ValueError("Native Beam resume requires the saved world size")""",
        )
        replace(
            "    write_checkpoint_dir(save_dir, write_shards)",
            """    from modal_dojo.frameworks.miles.modal_helpers.beam_checkpoint import LAYOUT
    write_checkpoint_dir(save_dir, write_shards, metadata={
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "iteration": iteration,
        "layout": {key: getattr(args, key) for key in LAYOUT},
    })""",
        )
        replace(
            '    training_state = torch.load(state_path, map_location="cpu", weights_only=False)',
            """    training_state = torch.load(state_path, map_location="cpu", weights_only=False)
    if load_optimizer and (
        training_state.get("iteration") is None
        or training_state.get("optimizer") is None
        or training_state.get("opt_param_scheduler") is None
    ):
        raise ValueError("Native Beam resume requires complete training state")""",
        )
    elif target == "megatron_config.py":
        replace(
            '    if args.megatron_to_hf_mode == "bridge":',
            """    if args.megatron_to_hf_mode == "bridge" and getattr(args, "lora_adapter_path", None):
        from modal_dojo.frameworks.miles.modal_helpers.beam_checkpoint import validate_resume
        validate_resume(args)
        args.load = args.ref_load or args.hf_checkpoint
        args.start_rollout_id = None
        return
    if args.megatron_to_hf_mode == "bridge":""",
        )
    elif target == "model.py":
        replace(
            "    if opt_param_scheduler is not None and not (args.use_checkpoint_opt_param_scheduler and iteration > 0):",
            "    if not native_optimizer_restored and opt_param_scheduler is not None and not (args.use_checkpoint_opt_param_scheduler and iteration > 0):",
        )
    elif target == "checkpoint.py":
        replace(
            "            if loaded:\n",
            """            if not loaded or (optimizer is not None and (iteration is None or not native_optimizer_restored)):
                raise RuntimeError(f"Could not restore complete native LoRA training state: {adapter_path}")
            if loaded:
""",
        )
    elif target == "rollout_executor.py":
        replace(
            "    async def _get_rollout_data(self, rollout_id, trainer_model_id: str | None = None):",
            """    async def _get_rollout_data(self, rollout_id, trainer_model_id: str | None = None):
        from modal_dojo.frameworks.miles.modal_helpers.beam_checkpoint import remember_rollout_cursor
        remember_rollout_cursor(self, rollout_id)""",
        )
        replace(
            "        self.data_source.save(rollout_id)",
            """        from modal_dojo.frameworks.miles.modal_helpers.beam_checkpoint import save_rollout_cursor
        save_rollout_cursor(self, rollout_id)""",
        )
        replace(
            "        event_logger_checkpoint.snapshot(self.args, rollout_id)",
            """        event_logger_checkpoint.snapshot(self.args, rollout_id)
        from modal_dojo.frameworks.miles.modal_helpers.beam_checkpoint import publish_checkpoint
        publish_checkpoint(self.args, rollout_id)""",
        )
        replace(
            "    def load(self, rollout_id: int | None = None) -> None:",
            """    def load(self, rollout_id: int | None = None) -> None:
        previous = self.args.load
        if getattr(self.args, "lora_adapter_path", None):
            from pathlib import Path
            self.args.load = str(Path(self.args.lora_adapter_path).parent.parent)
        try:
            self._dojo_load(rollout_id)
        finally:
            self.args.load = previous

    def _dojo_load(self, rollout_id: int | None = None) -> None:""",
        )
    elif target == "train_async.py":
        # The timing patch may wrap this call in a context manager first.
        anchor = next(
            line
            for line in source.splitlines()
            if line.strip() == "await rollout_executor.save(rollout_id)"
        )
        indent = anchor[: len(anchor) - len(anchor.lstrip())]
        replace(
            anchor,
            "\n".join(
                indent + line
                for line in """# Publishing reloads the checkpoint volume. Agent subprocesses in
# the prefetched rollout must close their trace files first. Keep
# the completed future so normal consumption retains this batch.
if not args.fully_async and rollout_data_next_future is not None:
    await rollout_data_next_future
await rollout_executor.save(rollout_id)""".splitlines()
            ),
        )
    elif target == "data_source.py":
        replace(
            "        state_dict = torch.load(path)",
            "        state_dict = torch.load(path, weights_only=False)",
        )
        replace(
            '        self.metadata = state_dict.get("metadata", {})',
            """        self.metadata = state_dict.get("metadata", {})
        if hasattr(self, "buffer"):
            self.buffer = state_dict.get("buffer", [])""",
        )
    else:
        raise ValueError(f"Unknown Beam checkpoint target: {target}")
    source += f"\n# {MARKER}\n"
    compile(source, target, "exec")
    return source


def main() -> None:
    root = Path("/root/miles/miles")
    for relative in (
        "backends/training_utils/checkpoint/io.py",
        "backends/megatron_utils/lora/utils.py",
        "backends/megatron_utils/megatron_config.py",
        "backends/megatron_utils/model.py",
        "backends/megatron_utils/checkpoint.py",
        "ray/rollout/rollout_executor.py",
        "rollout/data_source.py",
        "../train_async.py",
    ):
        path = root / relative
        path.write_text(patch_source(path.read_text(), path.name))
        print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
