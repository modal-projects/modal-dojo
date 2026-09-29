"""Patch Qwen3-ASR thinker model to gracefully handle packed_seq_params.

The megatron-bridge Qwen3-ASR thinker forward raises
``NotImplementedError("packed_seq_params is not supported")`` when
``packed_seq_params`` is passed.  As of slime nightly-dev-20260701a, slime's
``get_batch()`` always builds ``PackedSeqParams`` regardless of the
``qkv_format`` setting, so the Qwen3-ASR forward hits this error on the
``compute_log_prob`` step even with ``qkv_format="bshd"``.

This patch replaces the ``raise NotImplementedError`` with
``packed_seq_params = None``, which tells the model to process the input as
a single contiguous (non-packed) sequence — safe because the ASR recipe
trains with padded batches (``micro_batch_size=1``).

Idempotent. Run at image build:  python patch_qwen3_asr_packed_seq.py
"""

import pathlib


def main() -> None:
    marker = "PATCHED_ASR_PACKED_SEQ"
    old = 'raise NotImplementedError("packed_seq_params is not supported")'
    new = f"packed_seq_params = None  # {marker}: disable packed seq for ASR compat"

    for p in pathlib.Path("/usr/local/lib").glob(
        "python3.*/dist-packages/megatron/bridge/models/qwen3_asr/"
        "modeling_qwen3_asr/thinker_model.py"
    ):
        src = p.read_text()
        if marker in src:
            print("compat: thinker_model.py already patched for packed_seq_params:", p)
            continue
        if old in src:
            new_src = src.replace(old, new, 1)
            p.write_text(new_src)
            print("compat: patched thinker_model.py packed_seq_params:", p)
        else:
            print(
                "WARNING: Could not find packed_seq_params error string in "
                "thinker_model.py (may already be fixed upstream):",
                p,
            )


if __name__ == "__main__":
    main()
