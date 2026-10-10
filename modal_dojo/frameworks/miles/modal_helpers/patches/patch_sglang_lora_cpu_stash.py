"""Bound GPU memory during streamed LoRA updates in pinned K3 SGLang.

Source: sglang@880e3d2453eb7ef1738350e8c35ba2b956cc93a9.
Opt in with DOJO_SGLANG_LORA_CPU_STASH=1. Keep the complete adapter on CPU
until the existing loader validates it and copies TP slices into served buffers.
"""

from pathlib import Path

TARGET = Path(
    "/sgl-workspace/sglang/python/sglang/srt/managers/"
    "scheduler_components/weight_updater.py"
)
MARKER = "DOJO_SGLANG_LORA_CPU_STASH"
CHANGES = (
    (
        "    _weight_update_loaded: bool = False\n",
        """    _weight_update_loaded: bool = False
    # This manager uses dataclass(slots=True); transaction state needs declared fields.
    _dojo_lora_stash_cpu: bool = False
    _dojo_lora_staged_bytes: int = 0
    _dojo_lora_copy_s: float = 0.0
""",
    ),
    (
        "        self._weight_update_selector = recv_req.selector\n",
        """        import os as _dojo_os
        self._dojo_lora_stash_cpu = _dojo_os.environ.get("DOJO_SGLANG_LORA_CPU_STASH") == "1"
        self._dojo_lora_staged_bytes = 0
        self._dojo_lora_copy_s = 0.0
        self._weight_update_selector = recv_req.selector
""",
    ),
    (
        """            if copy_tensors:
                # the stash outlives this RPC, but an IPC sender may reuse the bucket once it replies
                tensor = tensor.clone()
                if tensor.is_cuda:
                    copied_devices.add(tensor.device)
""",
        """            if getattr(self, "_dojo_lora_stash_cpu", False):
                import time as _dojo_time
                _dojo_started = _dojo_time.monotonic()
                # Blocking D2H owns the bytes before the sender can reuse its IPC bucket.
                # copy=True also prevents aliasing if a caller supplies CPU tensors.
                tensor = tensor.to(device="cpu", non_blocking=False, copy=True)
                self._dojo_lora_copy_s += _dojo_time.monotonic() - _dojo_started
                self._dojo_lora_staged_bytes += tensor.numel() * tensor.element_size()
            elif copy_tensors:
                # the stash outlives this RPC, but an IPC sender may reuse the bucket once it replies
                tensor = tensor.clone()
                if tensor.is_cuda:
                    copied_devices.add(tensor.device)
""",
    ),
    (
        "            success, message = self._apply_lora_stash(recv_req.expected_lora_checksums)\n",
        """            import time as _dojo_time
            _dojo_started = _dojo_time.monotonic()
            try:
                success, message = self._apply_lora_stash(recv_req.expected_lora_checksums)
            finally:
                if getattr(self, "_dojo_lora_stash_cpu", False):
                    # End closes the transaction, including checksum/partial-stream failures.
                    self._lora_stash = {}
                    logger.info(
                        "DOJO_LORA_STAGING bytes=%d copy_s=%.3f apply_s=%.3f device=cpu",
                        self._dojo_lora_staged_bytes, self._dojo_lora_copy_s,
                        _dojo_time.monotonic() - _dojo_started,
                    )
""",
    ),
)


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        if any(source.count(new) != 1 for _, new in CHANGES):
            raise RuntimeError(f"{target}: unexpected patched LoRA staging source")
        return
    for old, new in CHANGES:
        if source.count(old) != 1:
            raise RuntimeError(f"{target}: LoRA staging source changed")
        source = source.replace(old, new)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"Patched {target}: opt-in CPU staging for streamed LoRA")


if __name__ == "__main__":
    apply()
