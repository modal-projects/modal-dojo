"""
image: slimerl/slime:nightly-dev-20260722a (sglang v0.5.15.post1)
commit: https://github.com/sgl-project/sglang/commit/76ee4bb98c64e9dc95ceb330bb12fd1cda405e17
file: sglang/python/sglang/srt/model_executor/model_runner.py::ModelRunner.forward
"""

from __future__ import annotations

from pathlib import Path

MARKER = "PATCHED_TRTLLM_MOE_REPACK"
TAIL = """

# PATCHED_TRTLLM_MOE_REPACK
def _tg_already_packed(module):
    named = getattr(module, "named_parameters", None)
    if named is None:
        return False
    for _, param in named(recurse=True):
        if not hasattr(param, "ndim"):
            continue
        if param.ndim >= 4 or (param.ndim >= 3 and param.shape[-1] == 64):
            return True
    return False


def _tg_repack_moe_after_hot_update(model):
    processed = 0
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            continue
        if hasattr(quant_method, "repack_weights_after_hot_update"):
            quant_method.repack_weights_after_hot_update(module)
            processed += 1
            continue
        if not (
            getattr(quant_method, "use_flashinfer_trtllm_moe", False)
            or "moe" in type(module).__name__.lower()
            or "moe" in type(quant_method).__name__.lower()
        ):
            continue
        if _tg_already_packed(module):
            continue
        if hasattr(quant_method, "process_weights_after_loading"):
            quant_method.process_weights_after_loading(module)
            processed += 1
    return processed


def _tg_model(owner):
    model = getattr(owner, "model", None)
    if model is not None:
        return model
    runner = getattr(owner, "model_runner", None)
    return getattr(runner, "model", None) if runner is not None else None


def _tg_try_repack(owner, label):
    model = _tg_model(owner)
    if model is None:
        return
    try:
        n = _tg_repack_moe_after_hot_update(model)
        if n:
            print(f"TRTLLM_MOE_REPACK {label} n={n}", flush=True)
    except Exception as exc:
        raise RuntimeError("TRT-LLM MoE repack failed") from exc


def _tg_wrap_forward(cls):
    original = getattr(cls, "forward", None)
    if original is None or getattr(original, "_tg_moe_repack", False):
        return

    def wrapped(self, *args, **kwargs):
        _tg_try_repack(self, "before forward")
        return original(self, *args, **kwargs)

    wrapped._tg_moe_repack = True
    cls.forward = wrapped


_tg_cls = globals().get("ModelRunner")
if _tg_cls is not None:
    _tg_wrap_forward(_tg_cls)
"""

_TARGETS = (
    Path("/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py"),
    Path(
        "/sgl-workspace/sglang/python/sglang/srt/model_executor/"
        "model_runner_components/weight_updater.py"
    ),
)


def _patch_file(path: Path) -> None:
    if not path.is_file():
        print(f"WARNING: {path} is absent; skipping {MARKER}")
        return
    source = path.read_text()
    if MARKER in source:
        print(f"{path.name} already has {MARKER}")
        return
    path.write_text(source + TAIL)
    print(f"Patched {path} to repack TRT-LLM MoE weights after hot update")


def main() -> None:
    for path in _TARGETS:
        _patch_file(path)


if __name__ == "__main__":
    main()
