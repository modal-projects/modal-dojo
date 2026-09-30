"""Patch Megatron's validate_sharding_integrity to warn instead of raising.

Hybrid architectures have layers with different parameter sets (e.g. GDN
layers carry linear_attn.dt_bias that standard attention layers lack).
Megatron rejects this because not every position in the global tensor is
covered.  This patch wraps the original function in a try/except so the
error becomes a warning.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

p = pathlib.Path("/root/Megatron-LM/megatron/core/dist_checkpointing/validation.py")
src = p.read_text()
old = "def validate_sharding_integrity("
if old in src and "_orig_impl" not in src:
    new = (
        "def validate_sharding_integrity(*_a, **_k):\n"
        "    import warnings as _w\n"
        "    try:\n"
        "        return _validate_sharding_integrity_orig_impl(*_a, **_k)\n"
        "    except Exception as _e:\n"
        '        _w.warn(f"Skipped sharding integrity validation: {_e}")\n'
        "\n\n"
        "def _validate_sharding_integrity_orig_impl("
    )
    p.write_text(src.replace(old, new, 1))
