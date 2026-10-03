"""Keep K3 generation health probes within the single resident adapter slot.

image: radixark/miles:dev-202609251434
commit: https://github.com/sgl-project/sglang/commit/880e3d2453eb7ef1738350e8c35ba2b956cc93a9
file: python/sglang/srt/entrypoints/http_server.py

A base-only health request would evict an installed no-CPU-backup adapter.
Probe the sole registered policy instead, and release its request lease on
both health completion and timeout. Actual generation remains enabled.
"""

from pathlib import Path

TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py")
MARKER = "PATCHED_K3_LORA_HEALTH"
ANCHOR = """    if _global_state.tokenizer_manager.is_generation:
        gri = GenerateReqInput(
            rid=rid,
            input_ids=[0],
            sampling_params=sampling_params,
            log_metrics=False,
        )
"""
REPLACEMENT = f"""    if _global_state.tokenizer_manager.is_generation:
        # {MARKER}: do not evict the sole non-reloadable policy.
        health_lora_path = None
        if (
            get_lora().enable_lora
            and get_lora().lora_no_cpu_backup
            and get_lora().max_loras_per_batch == 1
        ):
            adapters = _global_state.tokenizer_manager.lora_registry.get_all_adapters()
            if len(adapters) > 1:
                return Response(status_code=503)
            health_lora_path = next(iter(adapters), None)
        gri = GenerateReqInput(
            rid=rid,
            input_ids=[0],
            sampling_params=sampling_params,
            log_metrics=False,
            lora_path=health_lora_path,
        )
"""
SUCCESS_CLEANUP = """            task.cancel()
            _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
"""
TIMEOUT_CLEANUP = """    _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
    _global_state.tokenizer_manager.server_status = ServerStatus.UnHealthy
"""
SUCCESS_REPLACEMENT = """            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            _global_state.tokenizer_manager._finalize_lora_lease(
                _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
            )
"""
TIMEOUT_REPLACEMENT = """    await asyncio.gather(task, return_exceptions=True)
    _global_state.tokenizer_manager._finalize_lora_lease(
        _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
    )
    _global_state.tokenizer_manager.server_status = ServerStatus.UnHealthy
"""
CHANGES = (
    (ANCHOR, REPLACEMENT),
    (SUCCESS_CLEANUP, SUCCESS_REPLACEMENT),
    (TIMEOUT_CLEANUP, TIMEOUT_REPLACEMENT),
)


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        if any(source.count(new) != 1 or old in source for old, new in CHANGES):
            raise RuntimeError(f"{target}: unexpected patched K3 health source")
        return
    if any(source.count(old) != 1 for old, _ in CHANGES):
        raise RuntimeError(f"{target}: unexpected K3 health source")
    for old, new in CHANGES:
        source = source.replace(old, new)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"Patched {target}: generation health uses the resident K3 policy")


if __name__ == "__main__":
    apply()
