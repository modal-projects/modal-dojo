"""Avoid redundant full-width buffers in Beam's BF16 expert-TP1 adapters."""

from pathlib import Path

INPUT = "        grouped_input = grouped_inputs[0] if len(grouped_inputs) == 1 else torch.cat(grouped_inputs, dim=0)\n"
INPUT_PATCH = """        # MODAL_DOJO_BEAM_LORA_MEMORY: no padding or sharding at BF16 expert TP1.
        if not use_te_fp8 and expert_tp_size == 1:
            grouped_input = x
        else:
            grouped_input = grouped_inputs[0] if len(grouped_inputs) == 1 else torch.cat(grouped_inputs, dim=0)
"""
SCALE = """        if all(pad_len == 0 for pad_len in pad_lengths):
            return expert_output * (self.alpha / self.dim)
"""
SCALE_PATCH = """        if all(pad_len == 0 for pad_len in pad_lengths):
            if not use_te_fp8 and expert_tp_size == 1:
                return expert_output.mul_(self.alpha / self.dim)
            return expert_output * (self.alpha / self.dim)
"""


def patch_source(source: str) -> str:
    if INPUT_PATCH in source and SCALE_PATCH in source:
        return source
    if source.count(INPUT) != 1 or source.count(SCALE) != 1:
        raise RuntimeError("Pinned grouped expert LoRA implementation changed")
    return source.replace(INPUT, INPUT_PATCH).replace(SCALE, SCALE_PATCH)


if __name__ == "__main__":
    path = Path(
        "/opt/sglang/lib/python3.12/site-packages/megatron/bridge/peft/utils.py"
    )
    path.write_text(patch_source(path.read_text()))
    print(
        "Beam expert LoRA memory patch: reuse unpadded input and scale output in place"
    )
