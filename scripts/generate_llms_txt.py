"""Generate docs-next/public/llms.txt from tutorial + API catalogs.

uv run python -m scripts.generate_llms_txt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import yaml

from scripts.api_reference_manifest import (
    API_REFERENCE_MANIFEST,
    CLASS_REFERENCE_PATHS,
    GROUPS,
)
from scripts.tutorial_index import TutorialEntry, load_tutorial_index

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
GUIDES_DIR = ROOT / "docs-next" / "src" / "content" / "docs" / "guides"
DEFAULT_OUTPUT = ROOT / "docs-next" / "public" / "llms.txt"

SITE = "https://gym.modal.dev"
REPO = "https://github.com/modal-projects/modal-dojo"
GUIDE_SECTION_ORDER = ("start", "tools", "migration")


def flatten_doc_id(entry: str) -> str:
    path = entry.replace("\\", "/")
    suffix_at = path.rfind(".")
    if suffix_at > path.rfind("/"):
        path = path[:suffix_at]
    if path.endswith("/index"):
        path = path[: -len("/index")]
    parts = [part for part in path.split("/") if part]
    if len(parts) <= 2:
        return path
    return f"{parts[0]}/{parts[-1]}"


def _site_url(*parts: str) -> str:
    path = "/".join(part.strip("/") for part in parts if part)
    return f"{SITE}/{path}" if path else f"{SITE}/"


def _first_heading(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _guide_section(slug: str) -> str:
    return slug.split("/", 1)[0] if "/" in slug else ""


def _guide_section_sort_key(section: str) -> tuple[int, str]:
    try:
        return (GUIDE_SECTION_ORDER.index(section), "")
    except ValueError:
        return (len(GUIDE_SECTION_ORDER), section)


def _is_badge_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("[![") or stripped.startswith("![")


def _item(label: str, url: str) -> str:
    return f"- [{label}]({url})"


def _readme_heading_and_intro(markdown: str) -> tuple[str, str]:
    lines = markdown.splitlines()
    index = 0
    while index < len(lines) and not lines[index].startswith("# "):
        index += 1
    if index == len(lines):
        raise ValueError("README is missing an H1 heading")
    heading = lines[index][2:].strip()
    if not heading:
        raise ValueError("README H1 is empty")
    index += 1
    body: list[str] = []
    while index < len(lines) and not lines[index].startswith("## "):
        if not _is_badge_line(lines[index]):
            body.append(lines[index])
        index += 1
    intro = "\n".join(body).strip()
    if not intro:
        raise ValueError("README is missing intro text after the H1")
    return heading, intro.replace("](#", f"]({_site_url()}#")


def _collect_guides() -> list[tuple[str, str, int]]:
    """Return (slug, title, order) for authored guides."""
    guides: list[tuple[str, str, int]] = []
    for path in sorted(GUIDES_DIR.rglob("*.md")):
        text = path.read_text()
        if not text.startswith("---\n"):
            raise ValueError(f"Guide is missing frontmatter: {path}")
        parts = text.split("---\n", 2)
        if len(parts) != 3:
            raise ValueError(f"Guide has invalid frontmatter: {path}")
        frontmatter = parts[1]
        body = parts[2]

        try:
            metadata = yaml.safe_load(frontmatter)
        except yaml.YAMLError as exc:
            raise ValueError(f"Guide has invalid YAML frontmatter: {path}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"Guide frontmatter must be a mapping: {path}")

        order = metadata.get("order")
        if type(order) is not int:
            raise ValueError(f"Guide frontmatter requires an integer order: {path}")
        title = _first_heading(body)
        if not title:
            raise ValueError(f"Guide is missing an H1 heading: {path}")

        slug = path.relative_to(GUIDES_DIR).with_suffix("").as_posix()
        if slug != "index":
            guides.append((slug, title, order))

    guides.sort(
        key=lambda guide: (
            _guide_section_sort_key(_guide_section(guide[0])),
            guide[2],
            guide[1].lower(),
        )
    )
    return guides


def _render(
    tutorials: tuple[TutorialEntry, ...],
    guides: list[tuple[str, str, int]],
) -> str:
    heading, intro = _readme_heading_and_intro(README.read_text())
    lines: list[str] = [
        f"# {heading}",
        "",
        intro,
        "",
        f"Docs: {_site_url()}",
        f"Repo: {REPO}",
        "",
        "## Docs",
        "",
        _item("Home", _site_url()),
        _item("Guides", _site_url("guides")),
        _item("Tutorials", _site_url("tutorials")),
        _item("SDK Reference", _site_url("reference/sdk")),
        _item("CLI Reference", _site_url("reference/cli")),
        "",
        "## Guides",
        "",
    ]

    for section, section_guides in groupby(
        guides, key=lambda guide: _guide_section(guide[0])
    ):
        if section:
            lines.append(f"### {section.replace('-', ' ').title()}")
            lines.append("")
        for slug, title, _order in section_guides:
            lines.append(_item(title, _site_url(flatten_doc_id(f"guides/{slug}"))))
        lines.append("")

    lines.extend(
        [
            "## Tutorials",
            "",
        ]
    )

    for tutorial in tutorials:
        lines.append(_item(tutorial.title, _site_url("tutorials", tutorial.slug)))
    lines.append("")

    lines.extend(
        [
            "## Reference",
            "",
            _item("SDK Reference", _site_url("reference/sdk")),
            _item("CLI Reference", _site_url("reference/cli")),
            "",
        ]
    )

    by_group: dict[str, list[dict]] = defaultdict(list)
    for entry in API_REFERENCE_MANIFEST:
        by_group[entry["group"]].append(entry)

    for group_key, group_meta in sorted(
        GROUPS.items(), key=lambda item: item[1]["order"]
    ):
        entries = by_group.get(group_key)
        if not entries:
            continue
        lines.append(f"### {group_meta['label']}")
        lines.append("")
        for entry in entries:
            class_name = entry["class_name"]
            label = entry.get("sidebar_label") or class_name
            path = CLASS_REFERENCE_PATHS[class_name]
            lines.append(_item(label, _site_url(path)))
        lines.append("")

    lines.extend(
        [
            "## Optional",
            "",
            _item("AGENTS.md", f"{REPO}/blob/main/AGENTS.md"),
            _item("skills/", f"{REPO}/tree/main/skills"),
            "",
            "<!-- Generated by scripts/generate_llms_txt.py; do not edit by hand. -->",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate docs-next/public/llms.txt from documentation sources."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for llms.txt",
    )
    args = parser.parse_args()

    tutorials = load_tutorial_index()
    if not tutorials:
        raise SystemExit("No tutorials found")
    guides = _collect_guides()
    if not guides:
        raise SystemExit("No Markdown guides found")

    text = _render(tutorials, guides)
    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)

    n_api = len(API_REFERENCE_MANIFEST)
    try:
        display_path = out_path.relative_to(ROOT)
    except ValueError:
        display_path = out_path
    print(f"Wrote {display_path}")
    print(f"  guides: {len(guides)}")
    print(f"  tutorials: {len(tutorials)}")
    print(f"  api classes: {n_api}")


if __name__ == "__main__":
    main()
