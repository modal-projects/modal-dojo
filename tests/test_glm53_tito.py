"""GLM53 TITO backport and recipe integration, without GPU dependencies.

The input snapshot is tito_tokenizer.py at the recipe's Miles pin (5a5353d).
The output was produced by applying the upstream a584462a (#3087) file diff,
not by this patcher. Refresh both when changing the recipe's Miles pin.
"""

from pathlib import Path

import pytest

from modal_dojo.common.models.glm_5_3_flash import GLM_5_3_Flash_LoRA
from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_glm_5_3_flash_tito as patcher,
)
from modal_dojo.train_recipes.miles_recipe import MilesRecipe
from modal_dojo.train_recipes.miles_recipe.glm_5_3_flash_lora import (
    GLM_5_3_Flash_LoRA_Recipe,
)

TESTDATA = Path(__file__).parent / "testdata" / "miles"


def test_backport_matches_upstream_and_is_idempotent(tmp_path):
    source = (TESTDATA / "glm53_tito_tokenizer.py.input").read_text()
    expected = (TESTDATA / "glm53_tito_tokenizer.py.output").read_text()
    target = tmp_path / "miles/utils/chat_template_utils/tito_tokenizer.py"
    target.parent.mkdir(parents=True)
    target.write_text(source)

    patcher.main(tmp_path)
    assert target.read_text() == expected
    compile(expected, str(target), "exec")
    patcher.main(tmp_path)
    assert target.read_text() == expected


@pytest.mark.parametrize(
    "old,new",
    [
        ('GLM47 = "glm47"', 'GLM47 = "renamed"'),
        ("class GLM47TITOTokenizer", "class GLM53TITOTokenizer"),
        ("# Nemotron 3 implementation", "# Changed upstream boundary"),
    ],
)
def test_backport_rejects_drift_and_partial_application(old, new):
    source = (TESTDATA / "glm53_tito_tokenizer.py.input").read_text()
    with pytest.raises(ValueError, match="GLM53 TITO source changed"):
        patcher.patch_source(source.replace(old, new))


def test_glm53_default_recipe_emits_session_tito_flags():
    recipe = MilesRecipe.get_base_recipe(GLM_5_3_Flash_LoRA())
    args = recipe.cli_args()
    assert "--use-session-server" in args
    assert args[args.index("--tito-model") + 1] == "glm53"
    # The backport must run after checkout, alongside the Bridge/KDA patches.
    assert recipe.miles_git_ref == "5a5353d36c9133958f9ddb62c93be463bc849f88"
    from modal_dojo.common.patches import encode_patch

    encoded_patch = encode_patch(
        "patch_glm_5_3_flash_tito", Path(patcher.__file__).parent
    )
    assert any(encoded_patch in command for command in recipe.image_run_commands)


def test_tito_can_be_disabled_or_use_tree_sessions():
    args = GLM_5_3_Flash_LoRA_Recipe(
        use_session_server=False, tito_model="default"
    ).cli_args()
    assert "--use-session-server" not in args
    assert args[args.index("--tito-model") + 1] == "default"
    args = GLM_5_3_Flash_LoRA_Recipe(use_session_server="v2").cli_args()
    assert args[args.index("--use-session-server") + 1] == "v2"
