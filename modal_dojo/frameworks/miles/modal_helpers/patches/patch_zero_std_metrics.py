"""Normalize integer and float reward buckets in Miles' rollout diagnostics."""

from pathlib import Path

# radixark/miles#3722 hoisted the reward into ``uniform_rewards`` but still
# buckets an int 1 as "1" and a float 1.0 as "1.0".
_ANCHORS = (
    (
        "str(round(g[0].get_reward_value(args), 1))",
        "str(round(float(g[0].get_reward_value(args)), 1))",
    ),
    (
        "str(round(reward, 1)) for reward in uniform_rewards",
        "str(round(float(reward), 1)) for reward in uniform_rewards",
    ),
)


def patch_source(source: str) -> str:
    for old, new in _ANCHORS:
        if new in source:
            return source
        if source.count(old) == 1:
            return source.replace(old, new, 1)
    raise ValueError("Miles zero-std reward bucket anchor changed")


def main() -> None:
    root = Path("/root/miles/miles/ray")
    for path in (root / "rollout.py", root / "rollout/metrics.py"):
        if path.exists() and "def _compute_zero_std_metrics(" in path.read_text():
            path.write_text(patch_source(path.read_text()))
            print(f"Patched zero-std reward buckets in {path}")


if __name__ == "__main__":
    main()
