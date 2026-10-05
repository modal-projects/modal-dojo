"""
image: radixark/miles:dev-202609251434
commit: https://github.com/radixark/miles/commit/41c5e38b94ea23677de93b01a4a77d55677a8f09
file: miles/backends/training_utils/weight_update/updater.py
file: miles_plugins/models/kimi_k3/lora.py
"""

from pathlib import Path

UPDATER = Path("/root/miles/miles/backends/training_utils/weight_update/updater.py")
LORA = Path("/root/miles/miles_plugins/models/kimi_k3/lora.py")
MARKER = "PATCHED_K3_H200_LORA_OFFLOAD_EXPORT"
WEIGHTS_ANCHOR = """                self.weights_getter(),
                include_base=sync_base,
"""
WEIGHTS_REPLACEMENT = f"""                self.weights_getter() if sync_base else None,  # {MARKER}
                include_base=sync_base,
"""
EXPORT_ANCHOR = "def export_kimi_k3_lora_hf_chunks(model_chunks):\n"
HELPER = f"""# {MARKER}
def _k3_adapter_for_export(adapter):
    from copy import copy
    from megatron.training.global_vars import get_args
    from torch_memory_saver import torch_memory_saver

    if not getattr(get_args(), "offload_train", False):
        return adapter
    staged = copy(adapter)
    staged._parameters = adapter._parameters.copy()
    for name, parameter in adapter.named_parameters(recurse=False):
        if not parameter.is_cuda:
            continue
        backup = torch_memory_saver.get_cpu_backup(parameter, zero_copy=True)
        if backup is not None:
            staged._parameters[name] = nn.Parameter(
                backup.to(device=parameter.device), requires_grad=False
            )
    return staged


"""
ADAPTER_ANCHOR = """    for adapter in adapters:
        if adapter.kind in ("kda_attention", "mla_attention"):
"""
ADAPTER_REPLACEMENT = """    for adapter in adapters:
        adapter = _k3_adapter_for_export(adapter)
        if adapter.kind in ("kda_attention", "mla_attention"):
"""


def apply(updater: Path = UPDATER, lora: Path = LORA) -> None:
    changes = []
    for path, replacements in (
        (updater, ((WEIGHTS_ANCHOR, WEIGHTS_REPLACEMENT),)),
        (
            lora,
            (
                (EXPORT_ANCHOR, HELPER + EXPORT_ANCHOR),
                (ADAPTER_ANCHOR, ADAPTER_REPLACEMENT),
            ),
        ),
    ):
        source = path.read_text()
        if MARKER in source:
            if any(source.count(new) != 1 for _, new in replacements):
                raise RuntimeError(f"{path}: unexpected patched offloaded LoRA export")
        else:
            if any(source.count(old) != 1 for old, _ in replacements):
                raise RuntimeError(f"{path}: unexpected offloaded LoRA export source")
            for old, new in replacements:
                source = source.replace(old, new)
        compile(source, str(path), "exec")
        changes.append((path, source))
    for path, source in changes:
        path.write_text(source)
    print(
        "Patched K3 weight sync: skip unused base lookup and stage offloaded adapters"
    )


if __name__ == "__main__":
    apply()
