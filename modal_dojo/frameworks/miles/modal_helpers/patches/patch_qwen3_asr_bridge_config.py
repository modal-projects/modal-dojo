"""Compat shim: guard megatron-bridge's Qwen3-ASR config get_text_config.

megatron-bridge 0.5.0's hf_qwen3_asr config reads ``self.thinker_config`` in
``get_text_config()``, but transformers 5.6's ``validate_token_ids`` calls it
DURING ``super().__init__()`` — before ``__init__`` assigns ``thinker_config`` —
so loading ANY Qwen3-ASR config raises AttributeError. (Same bug class sglang
fixed in its config via PR #24187.) Guard the access; report upstream so this can
be dropped.

TODO(joy): remove this patch once the image's megatron-bridge fixes
``get_text_config()`` to not read ``thinker_config`` before ``__init__`` sets it
(the analogous fix to sglang PR #24187).

Idempotent. Run at image build:  python patch_qwen3asr_bridge_config.py
"""

import pathlib


def main() -> None:
    for p in pathlib.Path("/usr/local/lib").glob(
        "python3.*/dist-packages/megatron/bridge/models/qwen3_asr/hf_qwen3_asr/"
        "configuration_qwen3_asr.py"
    ):
        s = p.read_text()
        if 'hasattr(self, "thinker_config")' in s:
            print("compat: bridge config already guarded:", p)
            continue
        needle = "return self.thinker_config.get_text_config()"
        if needle in s:
            s = s.replace(
                "        " + needle,
                '        if not hasattr(self, "thinker_config"):\n'
                "            return self\n"
                "        " + needle,
                1,
            )
            p.write_text(s)
            print("compat: patched bridge get_text_config guard:", p)


if __name__ == "__main__":
    main()
