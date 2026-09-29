"""Compat shim: default megatron-bridge's Qwen3-ASR pg_collection.

``Qwen3ASRThinkerModel.__init__`` hard-raises when ``pg_collection is None``, but
slime's megatron model_provider builds the model without one. The error message
itself names the intended default (``ProcessGroupCollection.use_mpu_process_groups()``),
which is available once slime has initialized model-parallel state — so default to
it instead of raising. (Other megatron-bridge models accept a None pg_collection;
Qwen3-ASR is the outlier.) Report upstream so this can be dropped.

Idempotent. Run at image build:  python patch_qwen3asr_pg_collection.py
"""

import pathlib


def main() -> None:
    for p in pathlib.Path("/usr/local/lib").glob(
        "python3.*/dist-packages/megatron/bridge/models/qwen3_asr/"
        "modeling_qwen3_asr/thinker_model.py"
    ):
        s = p.read_text()
        if (
            "use_mpu_process_groups()" in s
            and "pg_collection = ProcessGroupCollection" in s
        ):
            print("compat: bridge pg_collection already defaulted:", p)
            continue
        needle = (
            "        if pg_collection is None:\n"
            "            raise ValueError(\n"
            '                "pg_collection is required for Qwen3ASRThinkerModel. "\n'
            '                "Use ProcessGroupCollection.use_mpu_process_groups() to get the default collection."\n'
            "            )"
        )
        if needle in s:
            s = s.replace(
                needle,
                "        if pg_collection is None:\n"
                "            pg_collection = ProcessGroupCollection.use_mpu_process_groups()",
                1,
            )
            p.write_text(s)
            print("compat: defaulted bridge pg_collection:", p)
        else:
            print("compat: WARNING — pg_collection raise block not found; skipping")


if __name__ == "__main__":
    main()
