# Bundled agent skills

This directory is the registry for agent skills distributed by Modal Dojo.
The package build includes the complete directory, and
`modal-dojo skills install` discovers each direct child containing a
`SKILL.md` file.

To add a skill:

1. Create `skills/<skill-name>/SKILL.md`.
2. Add any supporting files beneath the same skill directory.
3. Run `uv run pytest tests/test_cli_skills.py`.

No Python registry or packaging configuration change is required.
