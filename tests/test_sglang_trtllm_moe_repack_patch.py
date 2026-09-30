from __future__ import annotations

import pytest

from modal_dojo.frameworks.slime.modal_helpers.patches import (
    patch_sglang_trtllm_moe_repack as patcher,
)


class _Quant:
    def __init__(self) -> None:
        self.calls = 0

    def process_weights_after_loading(self, module) -> None:
        self.calls += 1
        module.repacked = True


class _Param:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.ndim = len(shape)


class _MoE:
    def __init__(self, *, packed: bool = False) -> None:
        self.quant_method = _Quant()
        self.repacked = False
        self._params = {"w13": _Param((4, 32, 768, 64) if packed else (4, 1024, 2048))}

    def named_parameters(self, recurse=True):
        return self._params.items()


class _Model:
    def __init__(self, *, packed: bool = False) -> None:
        self.moe = _MoE(packed=packed)

    def named_modules(self):
        yield "moe", self.moe


_RUNNER_SRC = (
    "class ModelRunner:\n"
    "    def __init__(self, model):\n"
    "        self.model = model\n"
    "    def update_weights_from_distributed(self, *args, **kwargs):\n"
    "        return True, 'ok'\n"
    "    def update_weights_from_tensor(self, *args, **kwargs):\n"
    "        return True, 'ok'\n"
    "    def forward(self, *args, **kwargs):\n"
    "        return 'fwd'\n"
)


def test_patch_appends_wrap_and_is_idempotent(tmp_path) -> None:
    target = tmp_path / "model_runner.py"
    target.write_text(_RUNNER_SRC)

    patcher._patch_file(target)
    patched = target.read_text()
    assert patcher.MARKER in patched
    patcher._patch_file(target)
    assert target.read_text() == patched

    ns: dict[str, object] = {}
    exec(patched, ns)
    runner = ns["ModelRunner"](_Model())
    assert runner.update_weights_from_distributed() == (True, "ok")
    assert runner.update_weights_from_tensor() == (True, "ok")
    assert runner.model.moe.repacked is False
    assert runner.model.moe.quant_method.calls == 0
    assert runner.forward() == "fwd"
    assert runner.model.moe.repacked is True
    assert runner.model.moe.quant_method.calls == 1


def test_patch_skips_already_packed_moe(tmp_path) -> None:
    target = tmp_path / "model_runner.py"
    target.write_text(_RUNNER_SRC)
    patcher._patch_file(target)
    ns: dict[str, object] = {}
    exec(target.read_text(), ns)
    runner = ns["ModelRunner"](_Model(packed=True))
    assert runner.forward() == "fwd"
    assert runner.model.moe.quant_method.calls == 0


def test_repack_failure_stops_forward(tmp_path) -> None:
    target = tmp_path / "model_runner.py"
    target.write_text(_RUNNER_SRC)
    patcher._patch_file(target)
    ns: dict[str, object] = {}
    exec(target.read_text(), ns)

    model = _Model()

    def boom(_module):
        raise ValueError("packed layout invalid")

    model.moe.quant_method.process_weights_after_loading = boom
    runner = ns["ModelRunner"](model)
    with pytest.raises(RuntimeError, match="TRT-LLM MoE repack failed") as ei:
        runner.forward()
    assert isinstance(ei.value.__cause__, ValueError)
    assert str(ei.value.__cause__) == "packed layout invalid"
