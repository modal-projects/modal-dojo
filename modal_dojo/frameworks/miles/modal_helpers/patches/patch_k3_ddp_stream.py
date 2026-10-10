"""
image: radixark/miles:dev-202609251434
commit: https://github.com/radixark/Megatron-LM/commit/f148a32b4385b758b66a77c9c3ad1641f1295d4b
file: megatron/training/training.py::get_model
"""

from pathlib import Path

TARGET = Path("/root/Megatron-LM/megatron/training/training.py")
MARKER = "PATCHED_K3_DDP_STREAM"
ANCHOR = "        ddp_stream = torch.cuda.Stream()\n"
REPLACEMENT = f"""        # {MARKER}
        # Eager K3 LoRA backward uses the current stream. Create its retained
        # AccumulateGrad nodes on that stream too; keep graph/FSDP paths intact.
        eager_k3_lora = (
            getattr(args, "model_name", None) == "kimi_k3"
            and bool(getattr(args, "lora_rank", None))
            and getattr(args, "cuda_graph_impl", "none") == "none"
            and not getattr(args, "enable_cuda_graph", False)
            and not getattr(args, "external_cuda_graph", False)
            and not getattr(args, "use_megatron_fsdp", False)
            and not getattr(args, "use_torch_fsdp2", False)
        )
        ddp_stream = torch.cuda.current_stream() if eager_k3_lora else torch.cuda.Stream()
"""


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        if source.count(REPLACEMENT) != 1:
            raise RuntimeError(f"{target}: unexpected patched DDP stream source")
        return
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: expected one DDP stream initialization")
    source = source.replace(ANCHOR, REPLACEMENT)
    compile(source, str(target), "exec")
    target.write_text(source)


if __name__ == "__main__":
    apply()
