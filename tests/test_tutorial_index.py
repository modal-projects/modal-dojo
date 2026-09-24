from pathlib import Path
import inspect
import sys

import modal_dojo
import pytest
import yaml

from scripts.api_reference_manifest import (
    API_REFERENCE_DENYLIST,
    API_REFERENCE_MANIFEST,
    CLASS_REFERENCE_PATHS,
)
from scripts.generate_models_table import (
    collect_model_preset_names,
    iter_registered_recipes,
)
from scripts.generate_docs_pages import generate_starlight
from scripts.generate_llms_txt import (
    GUIDES_DIR,
    README,
    _collect_guides,
    _guide_section,
    _guide_section_sort_key,
    _readme_heading_and_intro,
    _render,
    flatten_doc_id,
)
from scripts.tutorial_index import discover_tutorial_paths, parse_tutorial

ROOT = Path(__file__).resolve().parents[1]
GUIDE_PAGES = tuple(
    path for path in sorted(GUIDES_DIR.rglob("*.md")) if path.stem != "index"
)


def test_discover_tutorial_paths_finds_flat_and_nested(tmp_path: Path) -> None:
    (tmp_path / "flat.py").write_text("# ---\n# order: 0\n# ---\n# # Flat\n")
    (tmp_path / "main.py").write_text("# ---\n# order: 1\n# ---\n# # Main\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "main.py").write_text("# ---\n# order: 2\n# ---\n# # Nested\n")
    (nested / "env.py").write_text("VALUE = 1\n")
    (tmp_path / "helpers").mkdir()
    (tmp_path / "helpers" / "env.py").write_text("VALUE = 2\n")

    assert discover_tutorial_paths(tmp_path) == (
        (tmp_path / "flat.py", "flat"),
        (tmp_path / "main.py", "main"),
        (nested / "main.py", "nested"),
    )


@pytest.mark.parametrize("name", ["has.dot", "has space"])
def test_discover_tutorial_paths_rejects_unsafe_folder_names(
    tmp_path: Path, name: str
) -> None:
    nested = tmp_path / name
    nested.mkdir()
    (nested / "main.py").write_text("# ---\n# order: 0\n# ---\n# # Nested\n")

    with pytest.raises(ValueError, match="not a valid Python module name"):
        discover_tutorial_paths(tmp_path)


def test_discover_tutorial_paths_rejects_slug_collision(tmp_path: Path) -> None:
    (tmp_path / "duplicate.py").write_text("# ---\n# order: 0\n# ---\n# # Flat\n")
    nested = tmp_path / "duplicate"
    nested.mkdir()
    (nested / "main.py").write_text("# ---\n# order: 1\n# ---\n# # Nested\n")

    with pytest.raises(ValueError, match="Tutorial slug 'duplicate'"):
        discover_tutorial_paths(tmp_path)


@pytest.mark.parametrize("order", ["+0", "-0", "1_0"])
def test_parse_tutorial_rejects_non_decimal_order(tmp_path: Path, order: str) -> None:
    tutorial = tmp_path / "example.py"
    tutorial.write_text(f"# ---\n# order: {order}\n# ---\n# # Example\n")

    with pytest.raises(
        ValueError,
        match="frontmatter requires a non-negative integer order",
    ):
        parse_tutorial(tutorial, "example")


def test_parse_tutorial_github_defaults_none(tmp_path: Path) -> None:
    tutorial = tmp_path / "example.py"
    tutorial.write_text("# ---\n# order: 0\n# ---\n# # Example\n")

    entry = parse_tutorial(tutorial, "example")
    assert entry.github is None


def test_parse_tutorial_github_override(tmp_path: Path) -> None:
    tutorial = tmp_path / "example.py"
    tutorial.write_text(
        "# ---\n"
        "# order: 0\n"
        "# github: https://github.com/modal-labs/sf3/blob/main/src/train/main.py\n"
        "# ---\n"
        "# # Example\n"
    )

    entry = parse_tutorial(tutorial, "example")
    assert (
        entry.github == "https://github.com/modal-labs/sf3/blob/main/src/train/main.py"
    )


def test_parse_tutorial_rejects_github_with_deps(tmp_path: Path) -> None:
    tutorial = tmp_path / "example.py"
    tutorial.write_text(
        "# ---\n"
        "# order: 0\n"
        "# github: https://github.com/modal-labs/sf3\n"
        "# deps: pillow\n"
        "# ---\n"
        "# # Example\n"
    )

    with pytest.raises(ValueError, match="cannot set deps when github is overridden"):
        parse_tutorial(tutorial, "example")


def test_parse_tutorial_rejects_non_https_github(tmp_path: Path) -> None:
    tutorial = tmp_path / "example.py"
    tutorial.write_text(
        "# ---\n# order: 0\n# github: http://example.com\n# ---\n# # Example\n"
    )

    with pytest.raises(ValueError, match="github must be an https URL"):
        parse_tutorial(tutorial, "example")


def _frontmatter_lines(text: str) -> list[str]:
    assert text.startswith("---\n"), text[:40]
    parts = text.split("---\n", 2)
    assert len(parts) == 3
    return [line for line in parts[1].splitlines() if line.strip()]


def test_homepage_frontmatter(tmp_path: Path) -> None:
    generate_starlight(tmp_path)
    text = (tmp_path / "index.md").read_text()
    assert _frontmatter_lines(text) == [
        "order: 0",
        "head:",
        "  - tag: title",
        "    content: Modal Dojo",
    ]
    assert "\n# Modal Dojo\n" in text


def test_authored_pages_use_order_and_h1() -> None:
    assert GUIDE_PAGES, f"no guide pages under {GUIDES_DIR}"
    for path in GUIDE_PAGES:
        text = path.read_text()
        lines = _frontmatter_lines(text)
        assert len(lines) == 1, path
        assert lines[0].startswith("order: ")
        body = text.split("---\n", 2)[2]
        assert any(line.startswith("# ") for line in body.splitlines()), path


def test_collect_guides_orders_by_section_then_order() -> None:
    assert GUIDE_PAGES, f"no guide pages under {GUIDES_DIR}"
    orders_by_section: dict[str, list[int]] = {}
    expected: list[tuple[str, int, str, str]] = []

    for path in GUIDE_PAGES:
        text = path.read_text()
        parts = text.split("---\n", 2)
        metadata = yaml.safe_load(parts[1])
        assert isinstance(metadata, dict), path
        order = metadata.get("order")
        assert type(order) is int, path
        title = next(
            line[2:].strip() for line in parts[2].splitlines() if line.startswith("# ")
        )
        slug = path.relative_to(GUIDES_DIR).with_suffix("").as_posix()
        section = _guide_section(slug)
        orders_by_section.setdefault(section, []).append(order)
        expected.append((section, order, title.lower(), slug))

    for section, orders in orders_by_section.items():
        assert len(orders) == len(set(orders)), section

    expected.sort(key=lambda guide: (_guide_section_sort_key(guide[0]), *guide[1:]))
    guides = _collect_guides()
    assert [slug for slug, _, _ in guides] == [slug for _, _, _, slug in expected]


def test_readme_heading_and_intro_skips_badges_and_rewrites_anchors() -> None:
    markdown = (
        "# Modal Dojo\n"
        "\n"
        "[![ci](https://img.shields.io/badge/ci-ok)](https://example.com)\n"
        "\n"
        "First paragraph with a [Quickstart](#quickstart).\n"
        "\n"
        "Second paragraph.\n"
        "\n"
        "## Quickstart\n"
    )
    assert _readme_heading_and_intro(markdown) == (
        "Modal Dojo",
        "First paragraph with a "
        "[Quickstart](https://dojo.modal.dev/#quickstart).\n\n"
        "Second paragraph.",
    )


def test_render_groups_guides_by_section() -> None:
    text = _render(
        (),
        [
            ("start/model", "Model", 0),
            ("start/dataset", "Dataset", 1),
            ("tools/wandb-integration", "Weights & Biases integration", 2),
            ("migration/dataset-migration", "Dataset migration", 0),
        ],
    )
    heading, intro = _readme_heading_and_intro(README.read_text())
    assert text.startswith(f"# {heading}\n")
    start = text.index("### Start")
    tools = text.index("### Tools")
    migration = text.index("### Migration")
    assert start < tools < migration
    assert text.index("[Model]", start) < text.index("[Dataset]", start) < tools
    assert intro in text
    assert "https://dojo.modal.dev/guides/model)" in text
    assert "https://dojo.modal.dev/guides/dataset)" in text
    assert "https://dojo.modal.dev/guides/wandb-integration)" in text
    assert "/guides/start/" not in text
    assert "/guides/tools/" not in text
    for line in text.splitlines():
        if line.startswith("- ["):
            assert "): " not in line


def test_flatten_doc_id_keeps_section_drops_grouping_folders() -> None:
    assert flatten_doc_id("guides/tools/dashboard.md") == "guides/dashboard"
    assert flatten_doc_id("guides/start/model.md") == "guides/model"
    assert flatten_doc_id("guides/tools/agent.md") == "guides/agent"
    assert flatten_doc_id("reference/core/trainconfig.md") == "reference/trainconfig"
    assert (
        flatten_doc_id("reference/recipes/qwen3_4b_recipe.md")
        == "reference/qwen3_4b_recipe"
    )
    assert flatten_doc_id("reference/cli.md") == "reference/cli"
    assert flatten_doc_id("reference/cli/index.md") == "reference/cli"
    assert flatten_doc_id("tutorials/rl_basics.md") == "tutorials/rl_basics"
    assert flatten_doc_id("reference/index.md") == "reference"
    assert flatten_doc_id("reference/sdk.md") == "reference/sdk"
    assert flatten_doc_id("index.md") == "index"


def test_agent_guide_stem_is_agent() -> None:
    assert (GUIDES_DIR / "tools" / "agent.md").is_file()
    assert not (GUIDES_DIR / "tools" / "agent-driven-training.md").exists()


def test_class_reference_paths_are_section_plus_stem() -> None:
    for path in CLASS_REFERENCE_PATHS.values():
        parts = path.strip("/").split("/")
        assert parts[0] == "reference"
        assert len(parts) == 2


def test_public_api_reference_coverage() -> None:
    documented = {entry["class_name"] for entry in API_REFERENCE_MANIFEST}
    public = {
        name
        for name in modal_dojo.__all__
        if name not in API_REFERENCE_DENYLIST
        and (
            inspect.isclass(getattr(modal_dojo, name))
            or inspect.isfunction(getattr(modal_dojo, name))
            or inspect.isroutine(getattr(modal_dojo, name))
        )
    }
    assert documented == public
    assert documented.isdisjoint(API_REFERENCE_DENYLIST)
    assert "parse_qwen3_response" not in modal_dojo.__all__
    assert "METADATA_VOLUME_NAME" not in modal_dojo.__all__


def test_registered_presets_are_exported_and_sidebar_excluded() -> None:
    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from generate_api_reference import build_reference_sidebar

    documented = {entry["class_name"] for entry in API_REFERENCE_MANIFEST}
    excluded = {
        entry["class_name"]
        for entry in API_REFERENCE_MANIFEST
        if entry.get("sidebar_excluded")
    }
    labels = {item["label"] for item in build_reference_sidebar()["sdk"]}
    model_presets = collect_model_preset_names()
    registered = {name for name, *_ in iter_registered_recipes()}

    assert registered <= set(modal_dojo.__all__)
    assert registered <= documented
    assert model_presets & set(modal_dojo.__all__) <= excluded
    assert excluded.isdisjoint(labels)
    assert {
        "SlimeRecipe",
        "MilesRecipe",
        "ModelConfig",
        "HFModelConfiguration",
        "SglangRecipe",
        "VllmRecipe",
    }.isdisjoint(model_presets)


def test_eval_classes_are_not_documented() -> None:
    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from generate_api_reference import build_reference_sidebar

    documented = {entry["class_name"] for entry in API_REFERENCE_MANIFEST}
    labels = {item["label"] for item in build_reference_sidebar()["sdk"]}
    assert documented.isdisjoint(API_REFERENCE_DENYLIST)
    assert labels.isdisjoint(API_REFERENCE_DENYLIST)
    for name in API_REFERENCE_DENYLIST:
        assert inspect.isclass(getattr(modal_dojo, name))
    assert "MetadataStore" in API_REFERENCE_DENYLIST
    assert {"extract_code", "score_in_sandbox"} <= documented
    assert {"extract_code", "score_in_sandbox"}.isdisjoint(API_REFERENCE_DENYLIST)


def test_function_pages_use_getdoc() -> None:
    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from generate_api_reference import generate_function_page

    entries = {entry["class_name"]: entry for entry in API_REFERENCE_MANIFEST}
    doc = inspect.getdoc(modal_dojo.extract_code)
    assert doc
    page = generate_function_page(modal_dojo.extract_code, entries["extract_code"], 0)
    assert doc.splitlines()[0] in page


def test_package_exports_match_all() -> None:
    assert set(modal_dojo._EXPORTS) == set(modal_dojo.__all__)
    assert "parse_qwen3_response" not in modal_dojo._EXPORTS
    assert "METADATA_VOLUME_NAME" not in modal_dojo._EXPORTS


def test_api_reference_sidebar_lists_classes_before_functions() -> None:
    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from generate_api_reference import _page_heading, build_reference_sidebar

    assert _page_heading(0, "SDK") == [
        "---",
        "order: 0",
        "---",
        "",
        "# SDK",
        "",
    ]
    kinds = {entry["class_name"]: entry["kind"] for entry in API_REFERENCE_MANIFEST}
    labels = [item["label"] for item in build_reference_sidebar()["sdk"]]
    function_labels = [label for label in labels if kinds.get(label) == "function"]
    class_labels = [label for label in labels if kinds.get(label) != "function"]
    assert class_labels
    assert function_labels
    assert labels.index(class_labels[-1]) < labels.index(function_labels[0])


def test_excluded_preset_pages_omit_attributes() -> None:
    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from generate_api_reference import generate_class_page

    entries = {entry["class_name"]: entry for entry in API_REFERENCE_MANIFEST}
    for name in (
        "Qwen3_4B_Recipe",
        "Qwen3_5_4B_Miles_Recipe",
    ):
        entry = entries[name]
        assert entry.get("sidebar_excluded")
        page = generate_class_page(getattr(modal_dojo, name), entry, 0)
        assert "**Attributes**" not in page
    for name in ("SlimeRecipe", "MilesRecipe", "SglangRecipe"):
        entry = entries[name]
        assert not entry.get("sidebar_excluded")
        page = generate_class_page(getattr(modal_dojo, name), entry, 0)
        assert "**Attributes**" in page


def test_api_reference_overview_has_descriptions() -> None:
    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from generate_api_reference import _class_lede

    missing = [
        entry["class_name"]
        for entry in API_REFERENCE_MANIFEST
        if not _class_lede(getattr(modal_dojo, entry["class_name"]))
    ]
    assert missing == []


def test_mobile_sidebar_does_not_accent_every_row() -> None:
    css = (ROOT / "docs-next/src/styles/custom.css").read_text()
    assert "li:has(.active) > :is(" not in css
