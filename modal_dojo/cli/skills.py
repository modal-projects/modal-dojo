"""Install Training Gym agent skills into a project."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import click

from .commands import _TrainingGymGroup
from .errors import CLIError


SKILLS_DIRECTORY = Path(".agents") / "skills"
CLAUDE_SKILLS_DIRECTORY = Path(".claude") / "skills"


def _bundled_skills_path() -> Path:
    """Locate bundled skills in an installed wheel or source checkout."""
    package_root = Path(__file__).resolve().parent.parent
    candidates = (
        package_root / "_skills",
        package_root.parent / "skills",
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*/SKILL.md")):
            return candidate
    raise CLIError(
        "The bundled Training Gym skills are unavailable.",
        error="skills_not_bundled",
        hint="Reinstall modal-dojo and try again.",
    )


def _bundled_skills() -> dict[str, Path]:
    """Return every registered skill, keyed by directory name."""
    skills_path = _bundled_skills_path()
    return {
        skill_path.parent.name: skill_path.parent
        for skill_path in sorted(skills_path.glob("*/SKILL.md"))
    }


def _find_project_root(start: Path) -> Path:
    """Return the nearest Git repository containing ``start``."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise click.UsageError(
        "Could not find a Git repository. Run this command inside a repository "
        "or pass --project-dir."
    )


def _symlinked_claude_link_parent(project_root: Path) -> Path | None:
    """Return the first symlink in the Claude skill parent hierarchy."""
    claude_directory = project_root / ".claude"
    claude_skills_directory = project_root / CLAUDE_SKILLS_DIRECTORY
    for candidate in (claude_directory, claude_skills_directory):
        if candidate.is_symlink():
            return candidate
    return None


def _directory_contents(path: Path) -> dict[Path, bytes]:
    """Return file contents keyed by paths relative to ``path``."""
    return {
        file.relative_to(path): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }


def _canonical_skill_is_current(source: Path, destination: Path) -> bool:
    """Return whether a canonical destination already matches its source."""
    return (
        not destination.is_symlink()
        and destination.is_dir()
        and _directory_contents(source) == _directory_contents(destination)
    )


def _install_claude_link(
    link: Path,
    destination: Path,
    *,
    skill_name: str,
    replace_existing: bool,
) -> None:
    """Link Claude's skill directory to the canonical skill."""
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{skill_name}-",
            dir=link.parent,
            ignore_cleanup_errors=True,
        ) as staging:
            staging_root = Path(staging)
            staged_link = staging_root / skill_name
            target = Path(os.path.relpath(destination, start=link.parent))
            staged_link.symlink_to(target, target_is_directory=True)

            backup: Path | None = None
            if replace_existing:
                backup = staging_root / "previous"
                link.rename(backup)
            try:
                staged_link.rename(link)
            except OSError:
                if backup is not None:
                    backup.rename(link)
                raise
    except OSError as exc:
        raise CLIError(
            f"Could not link {skill_name} at {link}: {exc}",
            error="skill_install_failed",
        ) from exc


def _install_canonical_skill(
    source: Path,
    destination: Path,
    *,
    skill_name: str,
    force: bool,
) -> bool:
    """Install the canonical skill and return whether it was already installed."""
    destination_is_symlink = destination.is_symlink()
    destination_exists = destination_is_symlink or destination.exists()
    if _canonical_skill_is_current(source, destination):
        return True
    if destination_exists and not force:
        raise CLIError(
            f"{skill_name} already exists at {destination}.",
            error="skill_destination_exists",
            hint="Rerun with --force to replace it.",
        )

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{skill_name}-",
            dir=destination.parent,
            ignore_cleanup_errors=True,
        ) as staging:
            staging_root = Path(staging)
            staged_skill = staging_root / skill_name
            shutil.copytree(source, staged_skill)

            backup: Path | None = None
            if destination_exists:
                backup = staging_root / "previous"
                destination.rename(backup)
            try:
                staged_skill.rename(destination)
            except OSError:
                if backup is not None:
                    backup.rename(destination)
                raise
    except OSError as exc:
        raise CLIError(
            f"Could not install {skill_name} at {destination}: {exc}",
            error="skill_install_failed",
        ) from exc
    return False


def _ensure_claude_compatibility(
    project_root: Path,
    destination: Path,
    *,
    skill_name: str,
    force: bool,
) -> None:
    """Expose the canonical skill to Claude when its paths are safe to manage."""
    skills_directory = project_root / CLAUDE_SKILLS_DIRECTORY
    link = skills_directory / skill_name
    symlinked_parent = _symlinked_claude_link_parent(project_root)

    if symlinked_parent is not None:
        click.echo(
            f"Skipped Claude skill link because {symlinked_parent} is a symbolic link.",
            err=True,
        )
        return

    link_is_symlink = link.is_symlink()
    link_exists = link_is_symlink or link.exists()
    link_points_to_canonical = False
    if link_is_symlink:
        try:
            link_points_to_canonical = link.resolve() == destination.resolve()
        except (OSError, RuntimeError):
            pass
    if link_points_to_canonical:
        click.echo(f"Claude skill already linked at {link}")
        return
    if link_exists and not force:
        click.echo(
            f"Skipped Claude skill link because {link} already exists; "
            "rerun with --force to replace it.",
            err=True,
        )
        return

    try:
        _install_claude_link(
            link,
            destination,
            skill_name=skill_name,
            replace_existing=link_exists,
        )
    except CLIError as exc:
        click.echo(f"Skipped Claude skill link: {exc.format_message()}", err=True)
        return
    click.echo(f"Linked Claude skill at {link}")


def install_skills(*, project_dir: Path | None, force: bool) -> tuple[Path, ...]:
    """Install every bundled skill and return their destinations."""
    project_root = (
        project_dir.expanduser().resolve()
        if project_dir is not None
        else _find_project_root(Path.cwd())
    )
    skills = _bundled_skills()
    destinations = tuple(
        project_root / SKILLS_DIRECTORY / skill_name for skill_name in skills
    )
    installed_destinations: list[tuple[str, Path]] = []

    for (skill_name, source), destination in zip(
        skills.items(), destinations, strict=True
    ):
        try:
            skill_installed = _install_canonical_skill(
                source,
                destination,
                skill_name=skill_name,
                force=force,
            )
        except CLIError as exc:
            if exc.error != "skill_destination_exists":
                raise
            click.echo(f"Skipped {skill_name}: {exc.format_message()}", err=True)
            continue

        if skill_installed:
            click.echo(f"{skill_name} is already installed at {destination}")
        else:
            click.echo(f"Installed {skill_name} at {destination}")
        installed_destinations.append((skill_name, destination))

    for skill_name, destination in installed_destinations:
        _ensure_claude_compatibility(
            project_root,
            destination,
            skill_name=skill_name,
            force=force,
        )
    return tuple(destination for _, destination in installed_destinations)


@click.group("skills", cls=_TrainingGymGroup)
def skills_group() -> None:
    """Manage agent skills."""


@skills_group.command("install")
@click.option(
    "--project-dir",
    type=click.Path(
        path_type=Path,
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
    default=None,
    metavar="DIR",
    help="Project root, or the nearest Git repository when omitted.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Replace existing canonical skills or manageable Claude child paths.",
)
def install_command(*, project_dir: Path | None, force: bool) -> None:
    """Install bundled skills and Claude-compatible links."""
    install_skills(project_dir=project_dir, force=force)
