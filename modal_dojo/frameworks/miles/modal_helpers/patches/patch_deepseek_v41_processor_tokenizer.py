"""Patch sglang's ``get_processor`` so the DeepSeek-V4.1 branch can reach ``get_tokenizer``.

The sglang tree radixark ships for V4.1 (see patch_deepseek_v41_sglang_tree)
merges sgl-project/sglang#38798 onto a base where ``processor.py`` imports
``get_tokenizer`` lazily inside ``get_processor`` (the TokenizersBackend reload
branch). The PR's early return for ``deepseek_v41`` models with vision layers
calls ``get_tokenizer`` before that local import runs, so every engine dies at
startup with ``UnboundLocalError: cannot access local variable 'get_tokenizer'``.
Hoist the import to module scope, as the PR branch itself has it.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

MARKER = "PATCHED_DEEPSEEK_V41_PROCESSOR_TOKENIZER"

TARGET = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/utils/hf_transformers/processor.py"
)

OLD_IMPORT = """from .tokenizer import (
    _TOKENIZERS_BACKEND,
    _fix_added_tokens_encoding,
    _fix_special_tokens_pattern,
    _install_tokenizer_warnings_filter,
)
"""
NEW_IMPORT = f"""from .tokenizer import (  # {MARKER}
    _TOKENIZERS_BACKEND,
    _fix_added_tokens_encoding,
    _fix_special_tokens_pattern,
    _install_tokenizer_warnings_filter,
    get_tokenizer,
)
"""

OLD_LOCAL = """    if type(tokenizer).__name__ == _TOKENIZERS_BACKEND:
        from .tokenizer import get_tokenizer

"""
NEW_LOCAL = """    if type(tokenizer).__name__ == _TOKENIZERS_BACKEND:
"""

if not TARGET.exists():
    print(f"{TARGET} not found; skipping DeepSeek-V4.1 processor tokenizer patch")
    raise SystemExit(0)

src = TARGET.read_text()
if MARKER in src:
    print("DeepSeek-V4.1 processor tokenizer patch already applied")
    raise SystemExit(0)

for old in (OLD_IMPORT, OLD_LOCAL):
    if src.count(old) != 1:
        raise SystemExit(
            "DeepSeek-V4.1 processor tokenizer patch did not match; sglang's "
            "utils/hf_transformers/processor.py has changed. Re-check how "
            "get_processor imports get_tokenizer before shipping."
        )

src = src.replace(OLD_IMPORT, NEW_IMPORT, 1).replace(OLD_LOCAL, NEW_LOCAL, 1)
TARGET.write_text(src)
print("Patched sglang processor.py: get_tokenizer imported at module scope")
