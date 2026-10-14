"""
image: radixark/miles:glm53next
commit: https://github.com/radixark/miles/commit/5a5353d36c9133958f9ddb62c93be463bc849f88
file: miles/backends/training_utils/loss_hub/losses.py::policy_loss_function

TODO: Remove once the pinned Miles image/ref includes this merged fix:
https://github.com/radixark/miles/pull/3655
"""

from pathlib import Path

TARGET = Path("/root/miles/miles/backends/training_utils/loss_hub/losses.py")
OLD = "    train_scored_log_probs = old_log_probs\n"
NEW = """    train_scored_log_probs = (
        torch.cat(trainer_scored_log_probs, dim=0)
        if trainer_scored_log_probs is not None else log_probs.detach()
    )
"""


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if NEW in source:
        return
    if source.count(OLD) != 1:
        raise RuntimeError(f"{target}: expected one train-rollout metric anchor")
    source = source.replace(OLD, NEW, 1)
    compile(source, str(target), "exec")
    target.write_text(source)


if __name__ == "__main__":
    apply()
