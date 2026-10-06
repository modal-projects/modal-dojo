"""Backport the runtime tokenizer change from radixark/miles#3087.

image: radixark/miles:glm53next
commit: https://github.com/radixark/miles/commit/5a5353d36c9133958f9ddb62c93be463bc849f88
file: miles/utils/chat_template_utils/tito_tokenizer.py

Upstream: a584462a5c5d42b3f189176d81b13dba815cae92. Keep the experimental
GLM LoRA driver and Bridge stack pinned until they can move together to a ref
with native GLM53 TITO support, then remove this backport.
"""

from pathlib import Path

GLM53_TOKENIZER = '''class GLM53TITOTokenizer(GLM47TITOTokenizer):
    """GLM-5.3 native text renderer with the shared GLM token boundary.

    The GLM-5.3 and GLM-5.3-Flash templates start generation with ``<think>`` even when ``enable_thinking=False``, so this family pins ``enable_thinking=True``. Flash support covers tokenizer text inputs, not multimodal processor inputs.
    """

    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"clear_thinking": False, "enable_thinking": True},
    )


'''


def patch_source(source: str) -> str:
    replacements = (
        ("# GLM 4.7 implementation\n", "# GLM family implementation\n"),
        (
            "# ---------------------------------------------------------------------------\n"
            "# Nemotron 3 implementation\n",
            GLM53_TOKENIZER
            + "# ---------------------------------------------------------------------------\n"
            "# Nemotron 3 implementation\n",
        ),
        ('    GLM47 = "glm47"\n', '    GLM47 = "glm47"\n    GLM53 = "glm53"\n'),
        (
            "            case cls.GLM47:\n                return GLM47TITOTokenizer\n",
            "            case cls.GLM47:\n                return GLM47TITOTokenizer\n"
            "            case cls.GLM53:\n                return GLM53TITOTokenizer\n",
        ),
    )
    # Accept the complete upstream backport, but fail on partial application or
    # source drift instead of silently building an image with the wrong family.
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
