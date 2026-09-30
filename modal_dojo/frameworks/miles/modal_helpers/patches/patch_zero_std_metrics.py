"""Normalize integer/float reward buckets in Miles' rollout diagnostics.

The percentage lookup uses "0.0"/"1.0", but round(int, 1) yields an int.
Normalize only the logged bucket key; rewards and training are untouched.
Supports both the older rollout.py and the executor-era rollout/metrics.py.
"""

from pathlib import Path

OLD = "str(round(g[0].get_reward_value(args), 1))"
NEW = "str(round(float(g[0].get_reward_value(args)), 1))"


def patch_source(source: str) -> str:
    if NEW in source:
        return source
    if source.count(OLD) != 1:
        raise ValueError("Miles zero-std reward bucket anchor changed")
    return source.replace(OLD, NEW, 1)


def main() -> None:
    root = Path("/root/miles/miles/ray")
    for path in (root / "rollout.py", root / "rollout/metrics.py"):
        if path.exists() and "def _compute_zero_std_metrics(" in path.read_text():
            path.write_text(patch_source(path.read_text()))
            print(f"Patched zero-std reward buckets in {path}")


if __name__ == "__main__":
    main()
