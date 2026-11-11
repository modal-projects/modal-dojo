from pathlib import Path
import subprocess
import sys

import pytest

from scripts.api_reference_manifest import (
    API_REFERENCE_MANIFEST,
    CLASS_REFERENCE_PATHS,
)
from scripts.generate_models_table import (
    DOCS_URL,
    FamilyRow,
    RecipeRow,
    collect_models,
    iter_registered_recipes,
    render_section,
    render_snippet,
)


def test_model_table_links_and_recipes(tmp_path: Path) -> None:
    families = collect_models()
    rendered = render_section(families)
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/generate_api_reference.py"),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    entries = {entry["class_name"]: entry for entry in API_REFERENCE_MANIFEST}
    assert "<thead><tr><th>Family</th><th>Model</th></tr></thead>" in rendered
    rows = [row for family in families for row in family.rows]
    assert len({row.config for row in rows}) == len(rows)
    assert len({row.slug for row in rows}) == len(rows)
    registered = list(iter_registered_recipes())
    for family in families:
        for row in family.rows:
            entry = entries[row.config]
            page = tmp_path / entry["group"] / f"{entry['class_name'].lower()}.md"
            assert page.is_file(), row.config
            assert (
                f'<a href="{DOCS_URL}{CLASS_REFERENCE_PATHS[row.config]}">'
                f"{row.display_name}</a>"
            ) in rendered
            assert (
                f'<details id="model-{row.slug}" data-model="{row.slug}">' in rendered
            )
            assert f"<summary>Run {row.recipe}</summary>" in rendered
            assert (
                f'<td><a href="{DOCS_URL}{CLASS_REFERENCE_PATHS[row.config]}">'
                f"{row.display_name}</a></td>"
            ) in rendered
            names = sorted(
                {
                    name
                    for name, _, config, _ in registered
                    if config.__name__ == row.config
                },
                key=lambda name: (len(name), name),
            )
            assert row.recipe == names[0]
            assert row.other_recipes == tuple(names[1:])


@pytest.mark.parametrize("others", [(), ("LongRecipe",), ("LongRecipe", "OtherRecipe")])
def test_snippet_alternate_recipes(others: tuple[str, ...]) -> None:
    row = RecipeRow("org/Model", "Model", "BaseRecipe", others)
    snippet = "\n".join(render_snippet(row))
    line = '    recipe=BaseRecipe(rm_type="deepscaler"),'
    if others:
        line += f"  # or {', '.join(others)}"
    assert line in snippet.splitlines()
    imports = snippet.split(")", 1)[0]
    assert "    BaseRecipe," in imports
    assert all(name not in imports for name in others)
    assert ("# or" in snippet) == bool(others)


def test_snippet_tutorial_link() -> None:
    row = RecipeRow("Qwen/Qwen3-ASR-1.7B", "Qwen3_ASR_1_7B", "Qwen3_ASR_1_7B_Recipe")
    assert render_snippet(row) == [
        "See the [tutorial](https://dojo.modal.dev/tutorials/audio_asr) for a runnable example."
    ]


def test_empty_model_table() -> None:
    rendered = render_section(())
    assert "<tbody>\n</tbody>" in rendered
    assert 'class="model-row"' not in rendered


def test_model_table_rejects_missing_reference() -> None:
    row = RecipeRow("org/Unknown", "Unknown", "Unknown_Recipe")
    with pytest.raises(KeyError, match="Unknown"):
        render_section((FamilyRow("Unknown", (row,)),))
