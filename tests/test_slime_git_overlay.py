import pytest

from modal_dojo.frameworks.slime.launcher import (
    SLIME_ROOT,
    _SLIME_EXTERNAL_PATCHES_B64,
    _SLIME_ROOT_PATCHES_B64,
    _overlay_slime_source,
    _patch_commands,
    _PATCH_SGLANG_PARALLEL_ALIASES_B64,
    _PATCH_SUBSTEP_TIMING_B64,
    _slime_git_overlay_command,
)
from modal_dojo.train_recipes.slime_recipe import Qwen3_6_27B_Recipe


REPOSITORY = "https://github.com/modal-projects/slime.git"
REVISION = "ba324bebdd3a3cbfc1946b58404a012ad607f38b"


class RecordingImage:
    def __init__(self) -> None:
        self.operations: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def add_local_dir(self, *args: object, **kwargs: object) -> "RecordingImage":
        self.operations.append(("add_local_dir", args, kwargs))
        return self

    def run_commands(self, *commands: object) -> "RecordingImage":
        self.operations.append(("run_commands", commands, {}))
        return self


def test_root_and_external_patches_are_disjoint() -> None:
    assert not (set(_SLIME_ROOT_PATCHES_B64) & set(_SLIME_EXTERNAL_PATCHES_B64))


def test_slime_git_overlay_requires_repository_and_full_revision() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        Qwen3_6_27B_Recipe(slime_git_repository=REPOSITORY)

    with pytest.raises(ValueError, match="full 40-character"):
        Qwen3_6_27B_Recipe(
            slime_git_repository=REPOSITORY,
            slime_git_revision="ba324beb",
        )


def test_slime_git_overlay_rejects_local_or_credentialed_sources() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        Qwen3_6_27B_Recipe(
            local_slime="/tmp/slime",
            slime_git_repository=REPOSITORY,
            slime_git_revision=REVISION,
        )

    with pytest.raises(ValueError, match="must not contain credentials"):
        Qwen3_6_27B_Recipe(
            slime_git_repository="https://token@example.com/slime.git",
            slime_git_revision=REVISION,
        )


def test_slime_git_overlay_command_is_revision_pinned() -> None:
    recipe = Qwen3_6_27B_Recipe(
        slime_git_repository=REPOSITORY,
        slime_git_revision=REVISION.upper(),
    )

    assert recipe.slime_git_revision == REVISION
    command = _slime_git_overlay_command(
        recipe.slime_git_repository or "", recipe.slime_git_revision or ""
    )
    assert "fetch --depth=1 origin" in command
    assert REVISION in command
    assert "checkout --detach FETCH_HEAD" in command


def test_git_overlay_reapplies_slime_source_patches_after_replacement() -> None:
    recipe = Qwen3_6_27B_Recipe(
        slime_git_repository=REPOSITORY,
        slime_git_revision=REVISION,
    )
    image = RecordingImage()

    assert _overlay_slime_source(image, recipe) is image

    assert [operation[0] for operation in image.operations] == [
        "run_commands",
        "run_commands",
    ]
    overlay_commands = image.operations[0][1]
    assert len(overlay_commands) == 1
    assert f"rm -rf /tmp/training-gym-slime/.git {SLIME_ROOT}" in overlay_commands[0]
    assert image.operations[1][1] == _patch_commands(_SLIME_ROOT_PATCHES_B64)
    for patch in (_PATCH_SGLANG_PARALLEL_ALIASES_B64, _PATCH_SUBSTEP_TIMING_B64):
        assert _patch_commands((patch,))[0] in image.operations[1][1]


def test_local_overlay_preserves_unpatched_dev_checkout_behavior() -> None:
    recipe = Qwen3_6_27B_Recipe(local_slime="/tmp/local-slime")
    image = RecordingImage()

    assert _overlay_slime_source(image, recipe) is image

    assert [operation[0] for operation in image.operations] == ["add_local_dir"]
    assert image.operations[0][1] == ("/tmp/local-slime",)
    assert image.operations[0][2]["remote_path"] == SLIME_ROOT
