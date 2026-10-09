"""
image: radixark/miles:glm53next
file: fla/ops/kda/chunk_intra_token_parallel.py (FLA 0.4.2)

Pass BK from the host; Triton 3.7 rejects next_power_of_2 inside the JIT kernel.
"""

from pathlib import Path

MARKER = "PATCHED_MODAL_DOJO_GLM53_KDA"


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    replacements = (
        ("    K: tl.constexpr,\n", "    K: tl.constexpr,\n    BK: tl.constexpr,\n"),
        ("    BK: tl.constexpr = triton.next_power_of_2(K)\n", ""),
        ("        K=K,\n", "        K=K,\n        BK=triton.next_power_of_2(K),\n"),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"GLM KDA kernel changed: expected one {old!r}")
        source = source.replace(old, new, 1)
    return f"# {MARKER}\n" + source


def main() -> None:
    from importlib.util import find_spec

    path = (
        Path(find_spec("fla").origin).parent / "ops/kda/chunk_intra_token_parallel.py"
    )
    path.write_text(patch_source(path.read_text()))
    print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
