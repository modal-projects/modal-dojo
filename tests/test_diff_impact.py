from __future__ import annotations

from scripts.diff_impact import TUTORIAL_SRC_ROOT, _tutorial_slug_for_path, analyze_diff
from scripts.tutorial_index import TutorialEntry


def test_model_file_diff_does_not_infer_tutorial_relationships() -> None:
    diff = (
        "diff --git a/modal_dojo/common/models/qwen3_5_9b.py "
        "b/modal_dojo/common/models/qwen3_5_9b.py\n"
        "index 1234567..89abcde 100644\n"
        "--- a/modal_dojo/common/models/qwen3_5_9b.py\n"
        "+++ b/modal_dojo/common/models/qwen3_5_9b.py\n"
        "@@ -1,3 +1,3 @@\n"
    )

    report = analyze_diff(diff)

    assert "Qwen3_5_9B" in report.affected_classes
    assert report.affected_tutorials == ()


def test_flat_tutorial_diff_maps_to_tutorial() -> None:
    diff = (
        "diff --git a/tutorials/on_policy_distillation.py "
        "b/tutorials/on_policy_distillation.py\n"
        "index 1234567..89abcde 100644\n"
        "--- a/tutorials/on_policy_distillation.py\n"
        "+++ b/tutorials/on_policy_distillation.py\n"
        "@@ -1,3 +1,3 @@\n"
    )

    report = analyze_diff(diff)

    assert "on_policy_distillation" in {
        slug for slug, _, _ in report.affected_tutorials
    }


def test_tutorial_slug_for_path_reads_discovered_slug() -> None:
    flat_main = TUTORIAL_SRC_ROOT / "main.py"
    nested_main = TUTORIAL_SRC_ROOT / "nested" / "main.py"
    helper = TUTORIAL_SRC_ROOT / "nested" / "env.py"
    tutorials = {
        "main": TutorialEntry(
            path=flat_main, slug="main", order=0, title="Main", deps=()
        ),
        "nested": TutorialEntry(
            path=nested_main, slug="nested", order=1, title="Nested", deps=()
        ),
    }

    assert _tutorial_slug_for_path(flat_main, tutorials) == "main"
    assert _tutorial_slug_for_path(nested_main, tutorials) == "nested"
    assert _tutorial_slug_for_path(helper, tutorials) == "nested"
