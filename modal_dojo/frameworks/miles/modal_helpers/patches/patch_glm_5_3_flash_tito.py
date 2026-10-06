"""
image: radixark/miles:glm53next
commit: https://github.com/radixark/miles/commit/5a5353d36c9133958f9ddb62c93be463bc849f88
file: miles/utils/chat_template_utils/tito_tokenizer.py

Backport GLM53 TITO support from https://github.com/radixark/miles/pull/3087
while retaining the pinned GLM LoRA/Bridge stack. Uses the native template
with clear_thinking=False and enable_thinking=True for text sessions.
"""

from pathlib import Path

GLM53_TOKENIZER = """class GLM53TITOTokenizer(GLM47TITOTokenizer):
    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"clear_thinking": False, "enable_thinking": True},
    )


"""


def patch_source(source: str) -> str:
    replacements = (
        (
            "class Nemotron3TITOTokenizer(Qwen3TITOTokenizer):\n",
            GLM53_TOKENIZER + "class Nemotron3TITOTokenizer(Qwen3TITOTokenizer):\n",
        ),
        ('    GLM47 = "glm47"\n', '    GLM47 = "glm47"\n    GLM53 = "glm53"\n'),
        (
            "            case cls.GLM47:\n                return GLM47TITOTokenizer\n",
            "            case cls.GLM47:\n                return GLM47TITOTokenizer\n"
            "            case cls.GLM53:\n                return GLM53TITOTokenizer\n",
        ),
    )
    if all(new in source for _, new in replacements):
        return source
    if "GLM53" in source or '"glm53"' in source:
        raise ValueError("GLM53 TITO source changed or backport is incomplete")
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"GLM53 TITO source changed: expected one {old!r}")
        source = source.replace(old, new, 1)
    compile(source, "tito_tokenizer.py", "exec")
    return source


def main(root: Path = Path("/root/miles")) -> None:
    path = root / "miles/utils/chat_template_utils/tito_tokenizer.py"
    path.write_text(patch_source(path.read_text()))
    print(f"Applied upstream GLM53 TITO backport to {path}")


if __name__ == "__main__":
    main()
