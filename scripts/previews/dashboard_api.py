"""Deploy (or stop) a PR's dashboard backend as its own Modal app.

The frontend preview is static nginx, so before this its ``/api`` had to be
proxied at the deployed dashboard — a PR that changes both halves of the
dashboard then previewed its new frontend against the old API. Each PR now gets
its own copy of the backend, deployed in preview mode (no scheduled jobs, no
warm container) from the PR's checkout.

Run with plain ``python``, not ``modal run``:

    python scripts/previews/dashboard_api.py deploy 489
    python scripts/previews/dashboard_api.py stop 489

``deploy`` prints the app's web URL as its last line.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import modal

from modal_dojo.common.dashboard import DASHBOARD_PREVIEW_ENV_KEY

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_APP_PATH = REPO_ROOT / "dashboards" / "app.py"
WEB_FUNCTION = "fastapi_app"


def app_name(pr_number: int) -> str:
    return f"modal-dojo-dashboard-pr-{pr_number}"


def deploy(pr_number: int) -> str:
    """Deploy the PR's dashboard app and return its web URL."""
    name = app_name(pr_number)
    print(f"Deploying dashboard API preview for #{pr_number} as {name}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "deploy",
            str(DASHBOARD_APP_PATH),
            "--name",
            name,
        ],
        check=True,
        cwd=REPO_ROOT,
        env={**os.environ, DASHBOARD_PREVIEW_ENV_KEY: "1"},
    )
    return modal.Function.from_name(name, WEB_FUNCTION).get_web_url()


def is_already_gone(output: str, name: str) -> bool:
    """Whether ``modal app stop`` failed because there's nothing left to stop.

    The app has to be the missing thing: a failure about some other absent
    resource leaves the backend running.
    """
    lowered = output.lower()
    if name.lower() not in lowered:
        return False
    return any(
        phrase in lowered
        for phrase in (
            "app not found",
            "no app with name",
            "no such app",
            "lookup failed for app",
        )
    )


def stop(pr_number: int) -> None:
    """Stop the PR's dashboard app, tolerating one that's already gone."""
    name = app_name(pr_number)
    print(f"Stopping dashboard API preview {name}")
    result = subprocess.run(
        [sys.executable, "-m", "modal", "app", "stop", name],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if is_already_gone(result.stdout + result.stderr, name):
        print(f"{name} is already stopped.")
        return
    # Anything else leaves a backend running against real metadata, so the
    # cleanup job should go red and be retried.
    raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["deploy", "stop"])
    parser.add_argument("pr_number", type=int, help="PR number to act on")
    args = parser.parse_args()

    if args.command == "stop":
        stop(args.pr_number)
        return

    print(deploy(args.pr_number))


if __name__ == "__main__":
    main()
