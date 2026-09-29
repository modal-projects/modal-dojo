"""Dequantize DeepSeek block-scaled HF checkpoints as mbridge reads them.

DeepSeek's native checkpoints (V4.1 onwards) ship fp8 e4m3 dense weights with
(32, 32)-block e8m0 scales and packed-e2m1 fp4 routed experts with per-row
32-column e8m0 scales, each next to a sibling ``<name>.scale`` tensor. mbridge
maps HF names onto bf16 Megatron parameters and only reshapes: a raw fp8 tensor
is upcast without its scale and a packed fp4 tensor has half the columns the
parameter expects, so the load scatters mismatched shapes. This module wraps the
bridge's safetensor reader so every weight with a sibling scale comes back
dequantized to bf16 and the rest of the bridge sees a plain bf16 checkpoint.

Tensors are read with positional file reads rather than ``safe_open``, which maps
every shard whole (``UntypedStorage.from_file``) on each call; under a sandboxed
kernel those file pages are charged to the container and a 485 GB checkpoint
read one tensor at a time gets the rank killed long before host memory runs out.
"""

from __future__ import annotations

import json
import os
from glob import glob

import torch

SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    # e8m0 is decoded by hand (see e8m0_to_float), so its raw byte is enough.
    "F8_E8M0": torch.uint8,
}

FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
FP4_BLOCK = 32


def scale_name(weight_name: str) -> str:
    return weight_name.removesuffix(".weight") + ".scale"


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Decode e8m0 (biased power-of-two exponent) scales without relying on dtype casts."""
    if scale.dtype in (torch.float32, torch.bfloat16, torch.float16):
        return scale.to(torch.float32)
    exponent = scale.view(torch.uint8).to(torch.int32) - 127
    return torch.exp2(exponent.to(torch.float32))


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp8 e4m3 weight [M, N] with one scale per (M/bm, N/bn) block -> bf16 [M, N]."""
    m, n = weight.shape
    bm, bn = m // scale.shape[0], n // scale.shape[1]
    scale = (
        e8m0_to_float(scale).repeat_interleave(bm, dim=0).repeat_interleave(bn, dim=1)
    )
    return (weight.to(torch.float32) * scale).to(torch.bfloat16)


def dequant_fp4_packed(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 weight [M, N/2] (low nibble first) with scale [M, N/32] -> bf16 [M, N]."""
    m, half = weight.shape
    packed = weight.view(torch.uint8)
    nibbles = torch.stack([packed & 0x0F, packed >> 4], dim=-1).reshape(m, half * 2)
    values = FP4_TABLE.to(weight.device)[nibbles.to(torch.long)]
    scale = e8m0_to_float(scale).repeat_interleave(FP4_BLOCK, dim=1)
    return (values * scale).to(torch.bfloat16)


def dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.dtype == torch.float8_e4m3fn:
        return dequant_fp8_block(weight, scale)
    if weight.dtype in (torch.int8, torch.uint8):
        return dequant_fp4_packed(weight, scale)
    raise TypeError(f"no block dequantization for {weight.dtype}")


QUANTIZED_DTYPES = (torch.float8_e4m3fn, torch.int8, torch.uint8)


class ShardReader:
    """Read single tensors out of safetensors shards with positional reads.

    Headers are parsed once per shard; each tensor read touches only its own byte
    range and lands on ``device`` before any arithmetic.
    """

    def __init__(self, hf_dir: str, index: dict[str, str], device: torch.device):
        self.hf_dir = hf_dir
        self.index = index
        self.device = device
        self._headers: dict[str, tuple[int, dict]] = {}

    @classmethod
    def from_dir(cls, hf_dir: str, device: torch.device) -> ShardReader:
        index_file = os.path.join(hf_dir, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file) as f:
                index = json.load(f)["weight_map"]
        else:
            index = {}
            for path in glob(os.path.join(hf_dir, "*.safetensors")):
                _, header = cls._read_header(path)
                index.update(dict.fromkeys(header, os.path.basename(path)))
        return cls(hf_dir, index, device)

    @staticmethod
    def _read_header(path: str) -> tuple[int, dict]:
        with open(path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_len))
        header.pop("__metadata__", None)
        return 8 + header_len, header

    def _header(self, filename: str) -> tuple[int, dict]:
        if filename not in self._headers:
            self._headers[filename] = self._read_header(
                os.path.join(self.hf_dir, filename)
            )
        return self._headers[filename]

    def read(self, name: str) -> torch.Tensor:
        filename = self.index[name]
        data_start, header = self._header(filename)
        meta = header[name]
        start, end = meta["data_offsets"]
        buf = torch.empty(end - start, dtype=torch.uint8)
        if end > start:
            with open(os.path.join(self.hf_dir, filename), "rb") as f:
                f.seek(data_start + start)
                f.readinto(memoryview(buf.numpy()))
        return (
            buf.view(SAFETENSORS_DTYPES[meta["dtype"]])
            .reshape(meta["shape"])
            .to(self.device)
        )


def load_dequantized(reader: ShardReader, hf_weight_names: list[str]) -> dict:
    out = {}
    for name in hf_weight_names:
        weight = reader.read(name)
        scale = scale_name(name)
        if scale != name and scale in reader.index and weight.dtype in QUANTIZED_DTYPES:
            weight = dequant(weight, reader.read(scale))
        out[name] = weight
    return out


def wrap_safetensor_io(io):
    """Make ``io.load_some_hf_weight`` return dequantized bf16 for block-scaled weights.

    Wraps the instance rather than a class: DeepSeek bridges swap in
    ``SafeTensorIO`` subclasses (the V3 ``_scale_inv`` dequantizer passes
    ``.scale`` checkpoints through untouched), so a class patch on the base
    would be shadowed by their overrides.
    """
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    reader = ShardReader.from_dir(io.hf_dir, device)

    def load_some_hf_weight(hf_weight_names: list[str]) -> dict:
        return load_dequantized(reader, list(hf_weight_names))

    io.load_some_hf_weight = load_some_hf_weight
    return io


def install() -> None:
    """Hook ``Bridge.load_weights`` so the reader it builds is wrapped.

    ``_get_safetensor_io`` itself is overridden down the DeepSeek bridge MRO, so
    it is resolved through the instance at call time rather than patched on a class.
    """
    from mbridge.core.bridge import Bridge

    original_load_weights = Bridge.load_weights

    def load_weights(self, *args, **kwargs):
        get_io = self._get_safetensor_io
        self._get_safetensor_io = lambda weights_path: wrap_safetensor_io(
            get_io(weights_path)
        )
        print(
            "[hf_block_dequant] dequantizing block-scaled weights on load", flush=True
        )
        return original_load_weights(self, *args, **kwargs)

    Bridge.load_weights = load_weights
