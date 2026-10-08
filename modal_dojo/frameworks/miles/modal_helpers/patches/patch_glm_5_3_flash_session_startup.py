"""
image: radixark/miles:glm53next
commit: https://github.com/radixark/miles/commit/5a5353d36c9133958f9ddb62c93be463bc849f88
file: miles/ray/rollout/router_manager.py

Backport the separate 300-second session-server readiness budget from
https://github.com/radixark/miles/pull/3034 to the pinned process launcher.
Preserves the 120-second router timeout and Miles' default worker count.
"""

from pathlib import Path


def patch_source(source: str) -> str:
    replacements = (
        (
            "_SERVER_READY_TIMEOUT_SECS = 120\n",
            "_ROUTER_READY_TIMEOUT_SECONDS = 120.0\n"
            "_SESSION_SERVER_READY_TIMEOUT_SECONDS = 300.0\n",
        ),
        (
            "wait_for_server_ready(router_ip, router_port, process, "
            "timeout=_SERVER_READY_TIMEOUT_SECS)",
            "wait_for_server_ready(router_ip, router_port, process, "
            "timeout=_ROUTER_READY_TIMEOUT_SECONDS)",
        ),
        (
            "wait_for_server_ready(ip, port, process, "
            "timeout=_SERVER_READY_TIMEOUT_SECS)",
            "wait_for_server_ready(ip, port, process, "
            "timeout=_SESSION_SERVER_READY_TIMEOUT_SECONDS)",
        ),
    )
    if all(new in source for _, new in replacements):
        return source
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"Session startup source changed: expected one {old!r}")
        source = source.replace(old, new, 1)
    compile(source, "router_manager.py", "exec")
    return source


def main(root: Path = Path("/root/miles")) -> None:
    path = root / "miles/ray/rollout/router_manager.py"
    path.write_text(patch_source(path.read_text()))
    print(f"Applied upstream session-server readiness budget to {path}")


if __name__ == "__main__":
    main()
