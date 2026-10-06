"""
image: radixark/miles:dev-202609251434
commit: https://github.com/radixark/Megatron-Bridge/commit/8cd3466d14d2337c8492827b3712482c2b3e4866
file: megatron/bridge/models/conversion/model_bridge.py

``load_weights_hf_to_megatron`` caches the HF tensor behind every grouped
mapping (``is_grouped_export``, i.e. one HF tensor holding all experts of a
layer) for the whole load, so each rank ends up holding the dequantized BF16
experts of every layer in host RAM: ~234 GB per rank for GPT-OSS-120B, ~1.9 TB
for eight ranks on one node. Drop each entry once the last task that reads it
has been converted; the next layer's tensor then reuses the same memory.
"""

from pathlib import Path

MARKER = "PATCHED_BRIDGE_IMPORT_CACHE_EVICT"
LOOP = """        _hf_import_cache: Dict[str, torch.Tensor] = {}
        for task in self._with_progress_tracking(hf_to_megatron_tasks, description):
"""
NEW_LOOP = f"""        # {MARKER}: evict each grouped HF tensor after its last consumer.
        if not isinstance(hf_to_megatron_tasks, list):
            hf_to_megatron_tasks = list(hf_to_megatron_tasks)
        _hf_import_remaining: Dict[str, int] = {{}}
        for _task in hf_to_megatron_tasks:
            if (
                _task is not None
                and _task.megatron_module is not None
                and getattr(_task.mapping, "is_grouped_export", False)
            ):
                _key = str(_task.mapping.hf_param)
                _hf_import_remaining[_key] = _hf_import_remaining.get(_key, 0) + 1
{LOOP}"""
CONVERT = """            # 2) Delegate conversion & distribution to the bridge
            converted_weights = self._convert_loaded_hf_weight(task, hf_weights)
"""
NEW_CONVERT = f"""{CONVERT}            if is_grouped:  # {MARKER}
                _hf_import_remaining[hf_param_key] = _hf_import_remaining.get(hf_param_key, 1) - 1
                if _hf_import_remaining[hf_param_key] <= 0:
                    _hf_import_cache.pop(hf_param_key, None)
                hf_weights = None
"""


def target() -> Path:
    from importlib.util import find_spec

    spec = find_spec("megatron.bridge")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("megatron.bridge is not installed")
    root = Path(next(iter(spec.submodule_search_locations)))
    return root / "models/conversion/model_bridge.py"


def apply(path: Path | None = None) -> None:
    path = path or target()
    source = path.read_text()
    if MARKER in source:
        return
    if source.count(LOOP) != 1 or source.count(CONVERT) != 1:
        raise RuntimeError(f"{path}: unexpected load_weights_hf_to_megatron source")
    source = source.replace(LOOP, NEW_LOOP).replace(CONVERT, NEW_CONVERT)
    compile(source, str(path), "exec")
    path.write_text(source)
    print(f"Patched {path}: evict grouped HF import cache entries after last use")


if __name__ == "__main__":
    apply()
