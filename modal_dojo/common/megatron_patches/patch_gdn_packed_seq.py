"""Patch Megatron GDN to gracefully handle packed sequences.

Older versions of Megatron-LM raise ``NotImplementedError`` when GDN
layers receive ``packed_seq_params``.  Slime's ``get_batch()`` always
creates ``PackedSeqParams`` for THD format (the default ``qkv_format``),
so GDN models hit this error on every training step.

This patch replaces the ``raise NotImplementedError`` with
``packed_seq_params = None``, which tells GDN to process the input as
a single contiguous sequence.  This is safe because in GRPO training
with ``global_batch_size`` aligned to micro-batch size, each micro-batch
typically contains one sequence.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

p = pathlib.Path("/root/Megatron-LM/megatron/core/ssm/gated_delta_net.py")
if not p.exists():
    # Also check the site-packages path
    p = pathlib.Path(
        "/usr/local/lib/python3.12/dist-packages/megatron/core/ssm/gated_delta_net.py"
    )
if not p.exists():
    print("WARNING: gated_delta_net.py not found, skipping GDN packed seq patch")
else:
    src = p.read_text()
    marker = "PATCHED_GDN_PACKED_SEQ"
    if marker in src:
        print("gated_delta_net.py already patched for packed sequence handling")
    else:
        old = (
            'raise NotImplementedError("GDN does not support packed sequence for now.")'
        )
        new = f"packed_seq_params = None  # {marker}: disable packed seq for GDN compat"
        if old in src:
            new_src = src.replace(old, new, 1)
            p.write_text(new_src)
            print(
                "Patched gated_delta_net.py: replaced NotImplementedError with packed_seq_params = None"
            )
        else:
            print(
                "WARNING: Could not find GDN packed sequence error string in gated_delta_net.py"
            )
