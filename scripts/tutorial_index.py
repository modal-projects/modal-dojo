from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_DIR = REPO_ROOT / "tutorials"
DEP_PATTERN = re.compile(
    r"^[A-Za-z0-9_.-]+"
    r"(?: @ (?:git\+)?https://[A-Za-z0-9._/-]+(?:@[A-Za-z0-9._-]+)?)?"
    r"$"
)
FOLDER_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ORDER_PATTERN = re.compile(r"^\d+$")
MAX_SAFE_INTEGER = (1 << 53) - 1


@dataclass(frozen=True)
class TutorialEntry:
    path: Path
    slug: str
    order: int
    title: str
    deps: tuple[str, ...]
    github: str | None = None


def parse_tutorial(path: Path, slug: str) -> TutorialEntry:
    prefix = "" if path.suffix == ".md" else "# "
    lines = path.read_text().splitlines()
    if not lines or lines[0] != f"{prefix}---":
        raise ValueError(f"{path} must start with tutorial frontmatter")

    try:
        frontmatter_end = lines.index(f"{prefix}---", 1)
    except ValueError as exc:
        raise ValueError(f"{path} has unterminated tutorial frontmatter") from exc

    field_pattern = re.compile(rf"{prefix}([a-z_]+):\s*(.*)")
    fields: dict[str, str] = {}
    for line in lines[1:frontmatter_end]:
        match = field_pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"{path} has invalid frontmatter line: {line!r}")
        name, value = match.groups()
        if name not in {"order", "deps", "github"}:
            raise ValueError(f"{path} has unsupported frontmatter field: {name}")
        if name in fields:
            raise ValueError(f"{path} has duplicate frontmatter field: {name}")
        fields[name] = value

    order_text = fields.get("order")
    if order_text is None or ORDER_PATTERN.fullmatch(order_text) is None:
        raise ValueError(f"{path} frontmatter requires a non-negative integer order")
    order = int(order_text)
    if order > MAX_SAFE_INTEGER:
        raise ValueError(f"{path} frontmatter order exceeds the safe integer range")

    github = fields.get("github")
    if github is not None and not github.startswith("https://"):
        raise ValueError(f"{path} frontmatter github must be an https URL")

    deps = tuple(
        dependency.strip()
        for dependency in fields.get("deps", "").split(",")
        if dependency.strip()
    )
    if len(deps) != len(set(deps)):
        raise ValueError(f"{path} frontmatter deps must be unique")
    invalid_deps = [
        dependency for dependency in deps if not DEP_PATTERN.fullmatch(dependency)
    ]
    if invalid_deps:
        raise ValueError(f"{path} has invalid frontmatter deps: {invalid_deps}")
    if deps and (github is not None or path.suffix == ".md"):
        raise ValueError(
            f"{path} cannot set deps when github is overridden or in Markdown"
        )

    title_prefix = f"{prefix}# "
    title_line = next(
        (
            line
            for line in lines[frontmatter_end + 1 :]
            if line.startswith(title_prefix)
        ),
        None,
    )
    if title_line is None:
        raise ValueError(f"{path} is missing an H1 heading")

    return TutorialEntry(
        path=path,
        slug=slug,
        order=order,
        title=title_line.removeprefix(title_prefix).strip(),
        deps=deps,
        github=github,
    )


def discover_tutorial_paths(
    tutorials_dir: Path = TUTORIALS_DIR,
) -> tuple[tuple[Path, str], ...]:
    discovered: list[tuple[Path, str]] = []
    paths_by_slug: dict[str, Path] = {}
    if not tutorials_dir.is_dir():
        return ()
    for child in sorted(tutorials_dir.iterdir(), key=lambda path: path.name):
        if child.is_file() and child.suffix in {".py", ".md"}:
            candidate = child
            slug = child.stem
        elif child.is_dir() and (child / "main.py").is_file():
            if FOLDER_NAME_PATTERN.fullmatch(child.name) is None:
                raise ValueError(
                    f"Tutorial folder {child.name!r} is not a valid Python module "
                    f"name; use only letters, digits, and underscores"
                )
            candidate = child / "main.py"
            slug = child.name
        else:
            continue
        previous = paths_by_slug.get(slug)
        if previous is not None:
            raise ValueError(
                f"Tutorial slug {slug!r} is defined by both {previous} and {candidate}"
            )
        paths_by_slug[slug] = candidate
        discovered.append((candidate, slug))
    return tuple(discovered)


def load_tutorial_index(
    tutorials_dir: Path = TUTORIALS_DIR,
) -> tuple[TutorialEntry, ...]:
    entries = tuple(
        sorted(
            (
                parse_tutorial(path, slug)
                for path, slug in discover_tutorial_paths(tutorials_dir)
            ),
            key=lambda entry: (entry.order, entry.slug),
        )
    )
    orders = [entry.order for entry in entries]
    if orders != list(range(len(entries))):
        raise ValueError(f"Tutorial orders must be contiguous from 0: {orders}")
    return entries
