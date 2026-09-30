"""Install opt-in actor diagnostics only in the GLM53 LoRA image."""

import ast
from pathlib import Path

MARKER = "PATCHED_TRAINING_GYM_GLM53_LORA_DEBUG"
SUFFIX = f"""
# {MARKER}
from modal_training_gym.frameworks.miles.glm_5_3_flash_lora_debug import instrument_actor as _glm53_debug_actor
_glm53_debug_actor(MegatronTrainRayActor)
"""


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    classes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"
    ]
    if len(classes) != 1:
        raise ValueError("expected one MegatronTrainRayActor")
    methods = {
        node.name for node in classes[0].body if isinstance(node, ast.FunctionDef)
    }
    required = {"init", "wake_up", "compute_log_prob", "train_actor", "update_weights"}
    if missing := required - methods:
        raise ValueError(f"missing GLM diagnostic methods: {sorted(missing)}")
    result = source + SUFFIX
    compile(result, "actor.py", "exec")
    return result


if __name__ == "__main__":
    path = Path("/root/miles/miles/backends/megatron_utils/actor.py")
    path.write_text(patch_source(path.read_text()))
    print(f"Applied {MARKER} to {path}")
