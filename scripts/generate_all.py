"""Run all doc generators then build the Starlight site.

uv run scripts/generate_all.py          # generate + build
uv run scripts/generate_all.py --skip-build  # generate only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs-next"

GENERATORS = [
    [sys.executable, "scripts/generate_models_table.py"],
    [sys.executable, "scripts/generate_quickstart.py"],
    [sys.executable, "scripts/generate_docs_pages.py"],
    [sys.executable, "scripts/generate_api_reference.py"],
    [sys.executable, "-m", "scripts.generate_llms_txt"],
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate all docs and build the site."
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="Only run generators, skip npm build"
    )
    args = parser.parse_args()

    for cmd in GENERATORS:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    if args.skip_build:
        return

    print("Installing docs-next dependencies...")
    subprocess.run(["npm", "ci"], cwd=DOCS_DIR, check=True)

    print("Building Starlight site...")
    subprocess.run(["npm", "run", "build"], cwd=DOCS_DIR, check=True)

    print("Done. Deploy with: uv run modal deploy docs-next/docs_next_app.py")


if __name__ == "__main__":
    main()
