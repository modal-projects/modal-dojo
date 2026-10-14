"""
image: radixark/miles:glm53next
commit: https://github.com/radixark/miles/commit/5a5353d36c9133958f9ddb62c93be463bc849f88
image: radixark/miles:dev-202608041247
commit: https://github.com/radixark/miles/commit/5c517599fb55c2528a55c749e6eda5f99f51f0ad
file: miles/backends/training_utils/loss_hub/losses.py::policy_loss_function

TODO: Remove once all pinned Miles images/refs include this merged fix:
https://github.com/radixark/miles/pull/3655
"""

from pathlib import Path

ROOT = Path("/root/miles/miles/backends/training_utils")
OLD = "    train_scored_log_probs = old_log_probs\n"
NEW = """    train_scored_log_probs = (
        torch.cat({scores}, dim=0).detach()
        if {scores} is not None else log_probs.detach()
    )
"""


def apply(target: Path) -> None:
    source = target.read_text()
    scores = (
        "trainer_scored_log_probs"
        if "trainer_scored_log_probs =" in source
        else 'batch.get("log_probs")'
    )
    replacement = NEW.format(scores=scores)
    if replacement in source or (
        OLD not in source
        and "if trainer_scored_log_probs is not None else log_probs.detach()" in source
    ):
        return
    if source.count(OLD) != 1:
        raise RuntimeError(f"{target}: expected one train-rollout metric anchor")
    source = source.replace(OLD, replacement, 1)
    compile(source, str(target), "exec")
    target.write_text(source)


if __name__ == "__main__":
    for relative in ("loss.py", "loss_hub/losses.py", "loss/hub/losses.py"):
        target = ROOT / relative
        if target.is_file() and "def policy_loss_function(" in target.read_text():
            apply(target)
