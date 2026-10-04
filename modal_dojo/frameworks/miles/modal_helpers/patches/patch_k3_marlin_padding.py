"""
image: radixark/miles:dev-202609251434
commit: https://github.com/sgl-project/sglang/commit/880e3d2453eb7ef1738350e8c35ba2b956cc93a9
file: python/sglang/srt/layers/quantization/mxfp4.py
"""

from pathlib import Path

TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/layers/quantization/mxfp4.py")
MARKER = "PATCHED_K3_TP16_MARLIN_PADDING"
ANCHOR = """        if self.use_marlin and not self.use_mega_moe:
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 128
            )
"""
REPLACEMENT = f"""        if self.use_marlin and not self.use_mega_moe:
            # {MARKER}
            alignment = (
                64 if hidden_size == 3584 and intermediate_size_per_partition == 192
                else 128
            )
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, alignment
            )
"""


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        if source.count(REPLACEMENT) != 1:
            raise RuntimeError(f"{target}: unexpected patched MXFP4 allocation source")
        return
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: unexpected MXFP4 allocation source")
    source = source.replace(ANCHOR, REPLACEMENT)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"Patched {target}: compact K3 TP16 Marlin expert storage")


if __name__ == "__main__":
    apply()
