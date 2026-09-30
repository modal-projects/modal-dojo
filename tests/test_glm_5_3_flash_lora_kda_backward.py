"""Contract for the image-scoped Hopper KDA workaround."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_training_gym import GLM_5_3_Flash_LoRA_Recipe, GLM_5_3_Flash_Recipe
from modal_training_gym.common.patches import encode_patch
from modal_training_gym.frameworks.miles.modal_helpers.patches import (
    patch_glm_5_3_flash_lora_kda_backward as patch,
)

ROOT = Path(__file__).parent / "testdata/miles/glm_5_3_flash_lora"


def test_pinned_backward_patch_and_drift():
    source = (ROOT / "chunk_bwd.py.input").read_text()
    actual = patch.patch_source(source)
    assert actual == (ROOT / "chunk_bwd.py.output").read_text()
    compile(actual, "chunk_bwd.py", "exec")
    assert patch.patch_source(actual) == actual
    with pytest.raises(ValueError, match="autotuner changed"):
        patch.patch_source(source.replace("for BK in BK_LIST", "for BK in [64]"))


@pytest.mark.parametrize("hopper", [True, False])
def test_only_hopper_autotuner_changes(hopper):
    def configs(source):
        tree = ast.parse(source)
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "chunk_kda_bwd_kernel_wy_dqkg_fused"
        )
        tune = fn.decorator_list[1]
        expression = ast.Expression(
            next(k.value for k in tune.keywords if k.arg == "configs")
        )
        return eval(
            compile(expression, "configs", "eval"),
            {
                "triton": SimpleNamespace(Config=lambda fields, **kw: (fields, kw)),
                "IS_NVIDIA_HOPPER": hopper,
                "BK_LIST": [32, 64],
                "BV_LIST": [64, 128],
                "NUM_WARPS": [2, 4],
            },
        )

    original = (ROOT / "chunk_bwd.py.input").read_text()
    changed = configs(patch.patch_source(original))
    if hopper:
        assert changed == [({"BK": 32, "BV": 32}, {"num_warps": 4, "num_stages": 1})]
    else:
        assert changed == configs(original)


def test_patch_is_lora_image_only():
    payload = encode_patch(Path(patch.__file__).stem, Path(patch.__file__).parent)
    assert any(payload in c for c in GLM_5_3_Flash_LoRA_Recipe().image_run_commands)
    assert all(payload not in c for c in GLM_5_3_Flash_Recipe().image_run_commands)
