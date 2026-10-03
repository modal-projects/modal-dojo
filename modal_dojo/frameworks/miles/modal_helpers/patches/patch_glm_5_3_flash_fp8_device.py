"""
image: radixark/miles:glm53next
file: mbridge/models/ext/deepseek_v3/dequant_fp8_safetensor_io.py

Keep safetensors reads on each torchrun rank's GPU; bare cuda selects GPU 0.
"""

from pathlib import Path

MARKER = "PATCHED_MODAL_DOJO_GLM53_FP8_DEVICE"


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    old = 'device="cuda"'
    if source.count(old) != 2:
        raise ValueError("GLM FP8 reader changed: expected two CUDA device anchors")
    # safetensors interprets bare 'cuda' as cuda:0, independent of set_device.
    source = source.replace(old, 'device=f"cuda:{torch.cuda.current_device()}"')
    return f"# {MARKER}\n" + source


def main() -> None:
    from importlib.util import find_spec

    path = (
        Path(find_spec("mbridge").origin).parent
        / "models/ext/deepseek_v3/dequant_fp8_safetensor_io.py"
    )
    path.write_text(patch_source(path.read_text()))
    print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
