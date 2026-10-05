"""
image: radixark/miles:dev-202609251434
commit: https://github.com/radixark/miles/commit/41c5e38b94ea23677de93b01a4a77d55677a8f09
megatron-commit: f148a32b4385b758b66a77c9c3ad1641f1295d4b
file: miles/backends/megatron_utils/actor.py
file: miles/ray/train/group.py
file: megatron/core/dist_checkpointing/serialization.py
"""

from pathlib import Path

ACTOR = Path("/root/miles/miles/backends/megatron_utils/actor.py")
GROUP = Path("/root/miles/miles/ray/train/group.py")
SERIALIZATION = Path(
    "/root/Megatron-LM/megatron/core/dist_checkpointing/serialization.py"
)
MARKER = "PATCHED_K3_H200_STARTUP_TIMING"
HELPER = f"""
# {MARKER}
def _k3_startup_timed(op, callback, *args, **kwargs):
    from time import perf_counter

    started = perf_counter()
    logger.info("k3_startup op=%s phase=start", op)
    try:
        result = callback(*args, **kwargs)
    except BaseException:
        logger.info("k3_startup op=%s phase=end ok=false elapsed_s=%.3f", op, perf_counter() - started)
        raise
    logger.info("k3_startup op=%s phase=end ok=true elapsed_s=%.3f", op, perf_counter() - started)
    return result

"""
LOGGER = "logger = logging.getLogger(__name__)\n"
WAIT_ANCHOR = "        info = await self._inference_controller.start_update_weights()\n"
WAIT_REPLACEMENT = f"""        # {MARKER}
        from time import perf_counter

        readiness_started = perf_counter()
        logger.info("k3_startup op=inference_readiness phase=start")
        try:
            info = await self._inference_controller.start_update_weights()
        except BaseException:
            logger.info("k3_startup op=inference_readiness phase=end ok=false elapsed_s=%.3f", perf_counter() - readiness_started)
            raise
        logger.info("k3_startup op=inference_readiness phase=end ok=true elapsed_s=%.3f", perf_counter() - readiness_started)
        transfer_started = perf_counter()
        logger.info("k3_startup op=weight_transfer phase=start")
"""
END_ANCHOR = "        await self._inference_controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)\n"
END_REPLACEMENT = (
    END_ANCHOR
    + """        logger.info("k3_startup op=weight_transfer phase=end ok=true elapsed_s=%.3f", perf_counter() - transfer_started)
"""
)


def _replace(source: str, anchor: str, replacement: str, path: Path) -> str:
    if source.count(anchor) != 1:
        raise RuntimeError(f"{path}: unexpected startup timing source")
    return source.replace(anchor, replacement)


def apply(
    actor: Path = ACTOR, group: Path = GROUP, serialization: Path = SERIALIZATION
) -> None:
    changes = []
    for path in (actor, group, serialization):
        source = path.read_text()
        if MARKER in source:
            required = (
                (WAIT_REPLACEMENT, END_REPLACEMENT) if path == group else (HELPER,)
            )
            if any(source.count(fragment) != 1 for fragment in required):
                raise RuntimeError(f"{path}: unexpected patched startup timing source")
            continue
        if path == group:
            source = _replace(source, WAIT_ANCHOR, WAIT_REPLACEMENT, path)
            source = _replace(source, END_ANCHOR, END_REPLACEMENT, path)
        else:
            source = _replace(source, LOGGER, LOGGER + HELPER, path)
            if path == actor:
                start = source.index("    def sleep(self) -> None:\n")
                end = source.index(
                    "    def offload_grad_buffer(self) -> None:\n", start
                )
                body = source[start:end]
                for old, new in (
                    (
                        "clear_memory(clear_host_memory=True)",
                        '_k3_startup_timed("offload_cleanup", clear_memory, clear_host_memory=True)',
                    ),
                    (
                        "destroy_process_groups()",
                        '_k3_startup_timed("offload_process_groups", destroy_process_groups)',
                    ),
                ):
                    body = _replace(body, old, new, path)
                for tag in ('"grad_buffer"', '"default"', "None"):
                    label = tag.strip('"')
                    body = body.replace(
                        f"torch_memory_saver.pause(tag={tag})",
                        f'_k3_startup_timed("offload_{label}", torch_memory_saver.pause, tag={tag})',
                    )
                source = source[:start] + body + source[end:]
            else:
                for old, new in (
                    (
                        "sharded_strategy.load(sharded_state_dict, checkpoint_dir, async_strategy)",
                        '_k3_startup_timed("checkpoint_read", sharded_strategy.load, sharded_state_dict, checkpoint_dir, async_strategy)',
                    ),
                    (
                        "apply_factory_merges(common_state_dict, sh_ten_factories)",
                        '_k3_startup_timed("checkpoint_merge", apply_factory_merges, common_state_dict, sh_ten_factories)',
                    ),
                ):
                    source = _replace(source, old, new, path)
        compile(source, str(path), "exec")
        changes.append((path, source))
    for path, source in changes:
        path.write_text(source)
    print(
        "Patched K3 startup: time checkpoint reads, merges, offload, and inference readiness"
    )


if __name__ == "__main__":
    apply()
