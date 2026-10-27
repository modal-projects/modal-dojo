"""
image: radixark/miles:v0.1.2
file: megatron/bridge/peft/{utils,lora_layers}.py

Avoid redundant full-width buffers in Beam's BF16 expert-TP1 adapters.
"""

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


WRAPPER = """        # Serialize full-width branch buffers to keep long-context peak at two outputs.
        combined_output = linear_output.clone()
        del linear_output
        adapter_output = self.adapter_forward(self.adapter, layernorm_output.contiguous(), *args, **kwargs)
        adapter_output = adapter_output.reshape(combined_output.shape)
        combined_output.add_(adapter_output)
"""
WRAPPER_PATCH = """        # MODAL_DOJO_BEAM_LORA_ACCUMULATE: leave the base output unmodified.
        from megatron.bridge.peft.utils import GroupedExpertLinearAdapter
        if (
            isinstance(self.adapter, GroupedExpertLinearAdapter)
            and (self.adapter.config.expert_tensor_parallel_size or 1) == 1
            and not self.adapter._is_te_fp8_enabled()
            and linear_output.dtype == torch.bfloat16
        ):
            adapter_output = self.adapter_forward(self.adapter, layernorm_output.contiguous(), *args, **kwargs)
            combined_output = adapter_output.reshape(linear_output.shape)
            combined_output.add_(linear_output)
        else:
            combined_output = linear_output.clone()
            del linear_output
            adapter_output = self.adapter_forward(self.adapter, layernorm_output.contiguous(), *args, **kwargs)
            adapter_output = adapter_output.reshape(combined_output.shape)
            combined_output.add_(adapter_output)
"""


def patch_wrapper(source: str) -> str:
    if WRAPPER_PATCH in source:
        return source
    if source.count(WRAPPER) != 1:
        raise RuntimeError("Pinned LoRA output accumulation changed")
    return source.replace(WRAPPER, WRAPPER_PATCH)


def main() -> None:
    path = Path(
        "/opt/sglang/lib/python3.12/site-packages/megatron/bridge/peft/utils.py"
    )
    path.write_text(patch_source(path.read_text()))
    wrapper = path.with_name("lora_layers.py")
    wrapper.write_text(patch_wrapper(wrapper.read_text()))
    print(
        "Beam expert LoRA memory patch: reuse unpadded input and scale output in place"
    )


if __name__ == "__main__":
    main()
