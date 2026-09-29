"""Patch Megatron bridge to skip None conversion tasks (both directions).

When using bridge mode for text-only MoE models (e.g. Qwen3.6-35B-A3B),
the Qwen35VLMoEBridge generates conversion tasks for vision-model parameters
that don't exist in the text-only variant.  These tasks are ``None``, which
crashes whichever task field is dereferenced first:

  * Megatron→HF save (``stream_weights_megatron_to_hf``): ``task.param_weight``
  * HF→Megatron load (``load_weights_hf_to_megatron``):   ``task.megatron_module``

The save path is exercised during training (rollout weight sync); the load
path is exercised at startup when bridge mode loads the HF weights directly
via AutoBridge (no torch_dist).  Guard BOTH so a fresh bridge-mode run can
load the reference weights and then sync them back.

This patch inserts ``if task is None: continue`` before each dereference,
handling ALL occurrences in the file via re.sub.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib
import re

p = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/megatron/bridge/models/conversion/model_bridge.py"
)
if not p.exists():
    print("WARNING: model_bridge.py not found, skipping patch")
else:
    src = p.read_text()
    marker = "PATCHED_NONE_TASK_CHECK"
    if marker in src:
        print("model_bridge.py already patched for None task check")
    else:

        def _add_none_guard(m: re.Match) -> str:
            indent = m.group(1)
            original = m.group(0)
            return (
                f"{indent}if task is None:  # {marker}\n"
                f"{indent}    continue\n"
                f"{original}"
            )

        # Guard the first dereference of `task` in BOTH conversion directions:
        #   save: `if isinstance(task.param_weight, DTensor):`
        #   load: `if task.megatron_module is None:`
        new_src, count = re.subn(
            r"^( +)(if isinstance\(task\.param_weight, DTensor\):"
            r"|if task\.megatron_module is None:)",
            _add_none_guard,
            src,
            flags=re.MULTILINE,
        )
        if count > 0:
            p.write_text(new_src)
            print(
                f"Patched model_bridge.py: added None task guard at {count} location(s)"
            )
        else:
            print("WARNING: Could not find target string in model_bridge.py")
