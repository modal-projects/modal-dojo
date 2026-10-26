"""Convert a Miles native-LoRA checkpoint into a PEFT adapter SGLang can serve."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_REPLICATED, _COLUMN, _ROW = "replicated", "column", "row"

_DEEPSEEK_V41 = {
    "self_attention.lora_q_a_adapter.q_a": ("attn.wq_a", _REPLICATED),
    "self_attention.lora_q_b_adapter.q_b": ("attn.wq_b", _COLUMN),
    "self_attention.lora_kv_adapter.kv": ("attn.wkv", _REPLICATED),
    "self_attention.lora_o_a_adapter.o_a": ("attn.wo_a", _COLUMN),
    "self_attention.lora_o_b_adapter.o_b": ("attn.wo_b", _ROW),
    "mlp.shared_experts.lora_fc1_adapter.gate": ("ffn.shared_experts.w1", _COLUMN),
    "mlp.shared_experts.lora_fc1_adapter.up": ("ffn.shared_experts.w3", _COLUMN),
    "mlp.shared_experts.lora_fc2_adapter.down": ("ffn.shared_experts.w2", _ROW),
}

_SHARD = re.compile(r"adapter_megatron_rank(\d+)\.pt")
_PARAM = re.compile(r"(?:^|\.)decoder\.layers\.(\d+)\.(.+)_([AB])$")


def _load_shards(input_dir: Path) -> list[dict[str, torch.Tensor]]:
    ranks = sorted(
        int(match.group(1))
        for match in map(_SHARD.fullmatch, os.listdir(input_dir))
        if match
    )
    if not ranks or ranks != list(range(len(ranks))):
        raise ValueError(
            f"{input_dir}: expected adapter_megatron_rank0..N-1.pt, found ranks {ranks}"
        )
    return [
        torch.load(
            input_dir / f"adapter_megatron_rank{rank}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for rank in ranks
    ]


def _base_shapes(origin_hf_dir: Path) -> dict[str, list[int]]:
    with open(origin_hf_dir / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    shapes = {}
    for file in sorted(set(weight_map.values())):
        with safe_open(origin_hf_dir / file, framework="pt") as handle:
            for name in handle.keys():
                shapes[name] = handle.get_slice(name).get_shape()
    return shapes


def _projections(shard: dict[str, torch.Tensor]) -> dict[str, tuple[str, str, str]]:
    """Native parameter prefix -> (HF module, layout) for every adapter in a shard."""
    projections = {}
    for name in shard:
        match = _PARAM.search(name)
        if match is None or match.group(2) not in _DEEPSEEK_V41:
            raise ValueError(f"unsupported native adapter parameter {name!r}")
        layer, native, _ = match.groups()
        hf_leaf, layout = _DEEPSEEK_V41[native]
        projections[name[: -len("_A")]] = (f"layers.{layer}.{hf_leaf}", layout)
    for prefix in projections:
        if f"{prefix}_A" not in shard or f"{prefix}_B" not in shard:
            raise ValueError(f"native adapter {prefix!r} lacks its A/B pair")
    return projections


def _tensor_parallel_size(shard, projections, base_shapes) -> int:
    for prefix, (hf_name, layout) in projections.items():
        if layout == _COLUMN:
            rows = base_shapes[f"{hf_name}.weight"][0]
            local_rows = shard[f"{prefix}_B"].shape[0]
            if rows % local_rows:
                raise ValueError(
                    f"{hf_name}: {local_rows} local rows do not divide {rows}"
                )
            return rows // local_rows
    raise ValueError(
        "no column-parallel adapter to infer the tensor-parallel size from"
    )


def convert(
    input_dir: Path, output_dir: Path, origin_hf_dir: Path
) -> dict[str, torch.Tensor]:
    shards = _load_shards(input_dir)
    projections = _projections(shards[0])
    base_shapes = _base_shapes(origin_hf_dir)
    tp_size = _tensor_parallel_size(shards[0], projections, base_shapes)
    if len(shards) % tp_size:
        raise ValueError(
            f"{len(shards)} rank shards are not a multiple of TP {tp_size}"
        )

    for rank, shard in enumerate(shards):
        replica = shards[rank % tp_size]
        if shard.keys() != replica.keys() or any(
            not torch.equal(shard[name], replica[name]) for name in shard
        ):
            raise ValueError(
                f"rank {rank} does not match its tensor-parallel replica rank {rank % tp_size}"
            )

    tensors = {}
    for prefix, (hf_name, layout) in projections.items():
        a = shards[0][f"{prefix}_A"]
        b = shards[0][f"{prefix}_B"]
        if layout == _COLUMN:
            b = torch.cat(
                [shards[rank][f"{prefix}_B"] for rank in range(tp_size)], dim=0
            )
        elif layout == _ROW:
            a = torch.cat(
                [shards[rank][f"{prefix}_A"] for rank in range(tp_size)], dim=1
            )
        expected = base_shapes.get(f"{hf_name}.weight")
        if expected != [b.shape[0], a.shape[1]] or a.shape[0] != b.shape[1]:
            raise ValueError(
                f"{hf_name}: adapter B{list(b.shape)} @ A{list(a.shape)} does not match base weight {expected}"
            )
        tensors[f"{hf_name}.lora_A.weight"] = a.to(torch.bfloat16).contiguous()
        tensors[f"{hf_name}.lora_B.weight"] = b.to(torch.bfloat16).contiguous()

    config = json.loads((input_dir / "adapter_config.json").read_text())
    config.pop("format", None)
    config.pop("experts_shared_outer_loras", None)
    config["target_modules"] = sorted({name.rsplit(".lora_", 1)[0] for name in tensors})

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output_dir / "adapter_model.safetensors")
    (output_dir / "adapter_config.json").write_text(json.dumps(config))
    print(
        f"Wrote {len(tensors)} adapter tensors from {len(shards)} rank shards (TP {tp_size}) to {output_dir}"
    )
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-dir",
        required=True,
        help="native adapter directory, e.g. iter_0000010/adapter",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--origin-hf-dir",
        required=True,
        help="base HF checkpoint the adapter was trained on",
    )
    args = parser.parse_args()
    convert(Path(args.input_dir), Path(args.output_dir), Path(args.origin_hf_dir))


if __name__ == "__main__":
    main()
