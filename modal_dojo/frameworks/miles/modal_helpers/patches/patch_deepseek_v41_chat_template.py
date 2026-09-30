"""Patch Miles' DeepSeek chat-template bridge to recognize DeepSeek-V4.1.

DeepSeek-V4.1 checkpoints ship no jinja ``chat_template`` (``model_type``
``deepseek_v41``); prompts are rendered by sglang's ``encoding_dsv41``, which
sgl-project/sglang#38798 adds. radixark/miles#3179 wires the model into training
but leaves ``miles/utils/chat_template_utils/deepseek.py`` with only the
``deepseek_v32`` / ``deepseek_v4`` families, so ``--apply-chat-template`` falls
through to ``tokenizer.apply_chat_template`` and raises on the missing template.
Register the V4.1 family on the same ``DeepSeekFamily`` bridge, mirroring V4:
``encoding_dsv41`` shares the V4 ``encode_messages`` convention and the
``<｜Assistant｜>`` + thinking-token generation suffix.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

MARKER = "PATCHED_DEEPSEEK_V41_CHAT_TEMPLATE"

TARGET = pathlib.Path("/root/miles/miles/utils/chat_template_utils/deepseek.py")

OLD_IMPORT = "from sglang.srt.entrypoints.openai import encoding_dsv4\n"
NEW_IMPORT = f"from sglang.srt.entrypoints.openai import encoding_dsv4, encoding_dsv41  # {MARKER}\n"

OLD_FAMILIES = """V32 = DeepSeekV32Family()
V4 = DeepSeekV4Family()

_FAMILIES = {
    "deepseek_v32": V32,
    "deepseek_v4": V4,
}
"""
NEW_FAMILIES = """class DeepSeekV41Family(DeepSeekV4Family):
    template = encoding_dsv41


V32 = DeepSeekV32Family()
V4 = DeepSeekV4Family()
V41 = DeepSeekV41Family()

_FAMILIES = {
    "deepseek_v32": V32,
    "deepseek_v4": V4,
    "deepseek_v41": V41,
}
"""

if not TARGET.exists():
    print(f"{TARGET} not found; skipping DeepSeek-V4.1 chat template patch")
    raise SystemExit(0)

src = TARGET.read_text()
if MARKER in src:
    print("DeepSeek-V4.1 chat template patch already applied")
    raise SystemExit(0)

if OLD_IMPORT not in src or OLD_FAMILIES not in src:
    raise SystemExit(
        "DeepSeek-V4.1 chat template patch did not match; miles' "
        "chat_template_utils/deepseek.py has changed. Re-check the family table "
        "before shipping."
    )

src = src.replace(OLD_IMPORT, NEW_IMPORT, 1).replace(OLD_FAMILIES, NEW_FAMILIES, 1)
TARGET.write_text(src)
print("Patched miles chat_template_utils/deepseek.py with the deepseek_v41 family")
