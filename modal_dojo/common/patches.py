"""Image-build-time patch encoding.

A ``patch_*.py`` script is filed by what it edits, not by who applies it. Scripts
that patch Megatron (``/root/Megatron-LM``, shipped by the miles
image) live in ``common/megatron_patches/``; scripts that patch a framework's own
tree live in that framework's ``modal_helpers/patches/`` directory. The helper below
reads a script and base64-encodes it so the launcher can embed it in an
``Image.run_commands`` call without quoting issues.
"""

from __future__ import annotations

import base64
from pathlib import Path

# Every patch whose target is Megatron itself, whichever framework applies it; miles
# applies them all at image build.
_MEGATRON_PATCHES = Path(__file__).parent / "megatron_patches"


def encode_patch(name: str, patches_dir: Path) -> str:
    """Return base64-encoded contents of ``<patches_dir>/<name>.py``."""
    return base64.b64encode((patches_dir / f"{name}.py").read_bytes()).decode()
