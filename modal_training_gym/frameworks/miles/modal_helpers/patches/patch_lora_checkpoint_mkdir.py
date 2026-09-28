"""Create the LoRA checkpoint directory on every rank, not only rank 0.

``miles/backends/training_utils/checkpoint_io.py::write_checkpoint_dir`` has
rank 0 create the checkpoint directory, broadcasts success over gloo, and
then has every rank ``torch.save`` its adapter shard into it. On Modal each
node sees the shared checkpoints Volume through its own mount, and a
directory one container creates reaches the others only after a commit and
reload; ranks on another node can therefore hit ``Parent directory ... does
not exist`` while the rest of the cluster writes fine (Kimi-K3 attempt 12
lost two of 64 ranks this way, and rank 0's cleanup then removed the whole
checkpoint). Creating the directory on every rank is idempotent: Volumes
merge directory entries, and only the per-rank files inside are written by
one container each.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

MARKER = "PATCHED_LORA_CHECKPOINT_MKDIR"

TARGET = pathlib.Path("/root/miles/miles/backends/training_utils/checkpoint_io.py")
OLD = "    if prepare_error[0] is not None:\n        raise prepare_error[0]\n"
NEW = OLD + f"    checkpoint_dir.mkdir(parents=True, exist_ok=True)  # {MARKER}\n"

if not TARGET.exists():
    print(f"{TARGET} not found; skipping LoRA checkpoint mkdir patch")
    raise SystemExit(0)

src = TARGET.read_text()
if MARKER in src:
    print("LoRA checkpoint mkdir patch already applied")
    raise SystemExit(0)

if src.count(OLD) != 1:
    raise SystemExit(
        "LoRA checkpoint mkdir patch did not match exactly once; miles' "
        "checkpoint_io.py has changed. Re-check write_checkpoint_dir before shipping."
    )

TARGET.write_text(src.replace(OLD, NEW, 1))
print("Patched write_checkpoint_dir to mkdir the checkpoint directory on every rank")
