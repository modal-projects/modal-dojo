"""Dequantize MXFP4 compressed-tensors experts as mbridge reads them.

Expose each packed weight/scale pair as one BF16 weight to KimiK3Bridge,
avoiding a separate ~5.6 TB BF16 checkpoint before torch_dist conversion.
"""

from __future__ import annotations

import json
import os

import torch

try:
    from .hf_block_dequant import FP4_TABLE, ShardReader, e8m0_to_float
except ImportError:  # torchrun runs the converter as a script, not a package
    from hf_block_dequant import FP4_TABLE, ShardReader, e8m0_to_float

PACKED_SUFFIX = ".weight_packed"
SCALE_SUFFIX = ".weight_scale"
WEIGHT_SUFFIX = ".weight"
MXFP4_FORMAT = "mxfp4-pack-quantized"


def group_size_from_config(hf_dir: str) -> int:
    """Read the MXFP4 group size, or return 0 for an unquantized checkpoint."""
    with open(os.path.join(hf_dir, "config.json")) as f:
        config = json.load(f)
    section = config.get("text_config", config)
    quantization = section.get("quantization_config")
    if not quantization:
        return 0
    if quantization.get("format") != MXFP4_FORMAT:
        raise ValueError(
            f"{hf_dir} is quantized as {quantization.get('format')!r}, "
            f"not {MXFP4_FORMAT!r}"
        )
    return int(quantization["config_groups"]["group_0"]["weights"]["group_size"])


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 ``[M, N/2]`` with e8m0 ``[M, N/group]`` scales -> bf16 ``[M, N]``."""
    m, half = packed.shape
    n = half * 2
    if scale.shape[0] != m or n % scale.shape[1]:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not tile packed weight "
            f"{tuple(packed.shape)}"
        )
    group = n // scale.shape[1]
    packed = packed.view(torch.uint8)
    nibbles = torch.stack([packed & 0x0F, packed >> 4], dim=-1).reshape(m, n)
    values = FP4_TABLE.to(packed.device)[nibbles.to(torch.long)]
    scale = e8m0_to_float(scale).repeat_interleave(group, dim=1)
    return (values * scale).to(torch.bfloat16)


def rewrite_index(index: dict[str, str]) -> dict[str, str]:
    """Present each ``.weight_packed``/``.weight_scale`` pair as one ``.weight``."""
    rewritten = {}
    for name, filename in index.items():
        if name.endswith(SCALE_SUFFIX):
            packed = name.removesuffix(SCALE_SUFFIX) + PACKED_SUFFIX
            if packed not in index:
                raise KeyError(f"orphan MXFP4 scale {name} has no {packed}")
            continue
        if name.endswith(PACKED_SUFFIX):
            scale = name.removesuffix(PACKED_SUFFIX) + SCALE_SUFFIX
            if scale not in index:
                raise KeyError(f"MXFP4 weight {name} has no {scale}")
            name = name.removesuffix(PACKED_SUFFIX) + WEIGHT_SUFFIX
        rewritten[name] = filename
    return rewritten


def load_dequantized(
    reader: ShardReader, hf_weight_names: list[str], group_size: int
) -> dict:
    out = {}
    for name in hf_weight_names:
        packed = name.removesuffix(WEIGHT_SUFFIX) + PACKED_SUFFIX
        if name.endswith(WEIGHT_SUFFIX) and packed in reader.index:
            scale = reader.read(name.removesuffix(WEIGHT_SUFFIX) + SCALE_SUFFIX)
            weight = dequant_mxfp4(reader.read(packed), scale)
            if group_size and weight.shape[1] != group_size * scale.shape[1]:
                raise ValueError(
                    f"{name}: config group_size {group_size} does not match the "
                    f"stored scale layout {tuple(scale.shape)}"
                )
        else:
            weight = reader.read(name)
        out[name] = weight
    return out


def wrap_safetensor_io(io):
    """Expose a BF16 index while reading packed tensors through the original index."""
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    reader = ShardReader.from_dir(io.hf_dir, device)
    group_size = group_size_from_config(io.hf_dir)
    io.index = rewrite_index(dict(io.index))

    def load_some_hf_weight(hf_weight_names: list[str]) -> dict:
        return load_dequantized(reader, list(hf_weight_names), group_size)

    io.load_some_hf_weight = load_some_hf_weight
    return io


def install() -> None:
    """Wrap the reader before KimiK3Bridge rejects packed weight names."""
    from mbridge.core.bridge import Bridge
    from mbridge.core.safetensor_io import SafeTensorIO

    original_load_weights = Bridge.load_weights

    def load_weights(self, *args, **kwargs):
        def get_io(weights_path: str):
            return wrap_safetensor_io(
                SafeTensorIO(self._get_actual_hf_path(weights_path))
            )

        self._get_safetensor_io = get_io
        print("[hf_mxfp4_dequant] dequantizing MXFP4 experts on load", flush=True)
        return original_load_weights(self, *args, **kwargs)

    Bridge.load_weights = load_weights
