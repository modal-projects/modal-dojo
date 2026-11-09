from collections import Counter
from pathlib import Path

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
    render_section,
)


def test_model_table_links_and_recipes() -> None:
    families = collect_models()
    rendered = render_section(families)
    root = Path(__file__).resolve().parents[1]
    entries = {entry["class_name"]: entry for entry in API_REFERENCE_MANIFEST}
    assert "<thead><tr><th>Family</th><th>Model</th></tr></thead>" in rendered
    for family in families:
        counts = Counter(row.display_name for row in family.rows)
        for row in family.rows:
            entry = entries[row.config]
            page = (
                root
                / "docs-next/src/content/docs/reference"
                / entry["group"]
                / f"{entry['class_name'].lower()}.md"
            )
            assert page.is_file(), row.config
            assert (
                f'<a href="{DOCS_URL}{CLASS_REFERENCE_PATHS[row.config]}">'
                f"{row.display_name}</a>"
            ) in rendered
            assert (
                f'<details id="model-{row.slug}" data-model="{row.slug}">' in rendered
            )
            assert f"<summary>Run {row.recipe}</summary>" in rendered
            recipe_link = (
                f'<a href="{DOCS_URL}{CLASS_REFERENCE_PATHS[row.recipe]}">'
                f"{row.recipe}</a>"
            )
            assert (recipe_link in rendered) == (counts[row.display_name] > 1)


def test_empty_model_table() -> None:
    rendered = render_section(())
    assert "<tbody>\n</tbody>" in rendered
    assert 'class="model-row"' not in rendered


def test_model_table_rejects_missing_reference() -> None:
    row = RecipeRow("org/Unknown", "Unknown", "Unknown_Recipe")
    with pytest.raises(KeyError, match="Unknown"):
        render_section((FamilyRow("Unknown", (row,)),))
