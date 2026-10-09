"""Compat shim: teach slime's load_processor about Qwen3-ASR.

For ``qwen3_asr``, transformers' AutoProcessor returns a bare tokenizer, so slime
falls back to its GLM-4V processor and crashes (``NoneType video_processor``).
sglang already ships a ``Qwen3ASRProcessor`` — use it before the GLM-4V fallback.
Report upstream so this can be dropped.

Idempotent. Run at image build:  python patch_qwen3asr_processor.py
"""

import pathlib

_HELPER = '''

def _try_load_qwen3_asr_processor(name_or_path, **kwargs):
    """Qwen3-ASR has no transformers AutoProcessor entry; build sglang's composite
    WhisperFeatureExtractor + tokenizer processor (the same one the engine uses)."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
    except Exception:
        return None
    if getattr(cfg, "model_type", "") != "qwen3_asr":
        return None
    try:
        from sglang.srt.configs.qwen3_asr import Qwen3ASRProcessor

        return Qwen3ASRProcessor.from_pretrained(name_or_path)
    except Exception:
        return None

'''


def main() -> None:
    p = pathlib.Path("/root/slime/slime/utils/processing_utils.py")
    if not p.exists():
        print("compat: slime processing_utils not found at", p)
        return
    s = p.read_text()
    if "_try_load_qwen3_asr_processor" in s:
        print("compat: slime processor already patched")
        return
    if "def _try_load_glm4v_processor(" not in s:
        print("compat: WARNING — slime load_processor shape changed; skipping")
        return
    s = s.replace(
        "def _try_load_glm4v_processor(",
        _HELPER + "def _try_load_glm4v_processor(",
        1,
    )
    s = s.replace(
        "        proc = _try_load_glm4v_processor(name_or_path, **kwargs)",
        "        proc = _try_load_qwen3_asr_processor(name_or_path, **kwargs)"
        " or _try_load_glm4v_processor(name_or_path, **kwargs)",
        1,
    )
    p.write_text(s)
    print("compat: patched slime load_processor for qwen3_asr")


if __name__ == "__main__":
    main()
