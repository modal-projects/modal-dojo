"""Make slime's bridge-mode PPO critic work for vision-language models.

Bridge mode builds Qwen3-VL as a wrapper (``Qwen3VLModel``) whose ``forward()``
delegates to ``self.language_model``, which owns the real LM head. slime's critic
provider assigns the 1-output value head to ``model.output_layer`` on the wrapper,
where ``forward()`` never reads it, so the critic still emits vocab-sized logits
and ``get_values()`` fails its ``size(-1) == 1`` assertion on the first step.

Two edits, applied together or not at all:

* ``model_provider.py``: attach the value head to ``model.language_model`` when the
  model has one (unchanged for plain GPT models).
* ``checkpoint.py``: when a critic loads HF weights, register slime's value-head class
  with Megatron-Bridge (the image's Bridge predates its built-in registration) and let
  the policy's vocab-sized ``lm_head`` skip the 1-output value head, which keeps its
  fresh init. The actor's load stays strict.

Both files are left alone when their bridge code path does not exist (newer slime
removed it). Executed at image-build time via ``python3 <this file>``.
"""

from __future__ import annotations

from pathlib import Path

MARKER = "PATCHED_VLM_CRITIC_VALUE_HEAD"

MODEL_PROVIDER = Path("/root/slime/slime/backends/megatron_utils/model_provider.py")
CHECKPOINT = Path("/root/slime/slime/backends/megatron_utils/checkpoint.py")

PROVIDER_ANCHOR = """\
                if post_process:
                    model.output_layer = LinearForLastLayer(
                        input_size=model.config.hidden_size, output_size=1, config=model.config
                    )
                return model

            return _critic_provide
"""
PROVIDER_REPLACEMENT = f"""\
                if post_process:
                    # {MARKER}: VLM wrappers run the LM head inside language_model.
                    head_owner = getattr(model, "language_model", None)
                    if head_owner is None:
                        head_owner = model
                    head_owner.output_layer = LinearForLastLayer(
                        input_size=head_owner.config.hidden_size, output_size=1, config=head_owner.config
                    )
                return model

            return _critic_provide
"""

LOAD_ANCHOR = """\
        bridge.load_hf_weights(ddp_model)
"""
LOAD_REPLACEMENT = f"""\
        # {MARKER}: the policy lm_head does not fit the critic's 1-output value head.
        is_critic = getattr(ddp_model[0], "role", None) == "critic"
        if is_critic:
            from megatron.bridge.models.conversion.param_mapping import AutoMapping

            AutoMapping.register_module_type("LinearForLastLayer", "replicated")
        bridge.load_hf_weights(
            ddp_model, allowed_mismatched_params=["*output_layer.weight"] if is_critic else None
        )
"""


def patch_model_provider(source: str) -> str:
    if MARKER in source or 'megatron_to_hf_mode == "bridge"' not in source:
        return source
    if PROVIDER_ANCHOR not in source:
        raise RuntimeError(
            "bridge critic value-head anchor not found in model_provider.py"
        )
    return source.replace(PROVIDER_ANCHOR, PROVIDER_REPLACEMENT, 1)


def patch_checkpoint(source: str) -> str:
    if MARKER in source or "bridge.load_hf_weights(" not in source:
        return source
    if LOAD_ANCHOR not in source:
        raise RuntimeError("bridge HF load anchor not found in checkpoint.py")
    return source.replace(LOAD_ANCHOR, LOAD_REPLACEMENT, 1)


def main() -> None:
    # Compute both edits before writing either, so a failed anchor leaves both untouched.
    updates = [
        (path, patch(path.read_text()))
        for path, patch in (
            (MODEL_PROVIDER, patch_model_provider),
            (CHECKPOINT, patch_checkpoint),
        )
    ]
    for path, new_source in updates:
        if new_source != path.read_text():
            path.write_text(new_source)
            print(f"Patched {path} for VLM critic value head")
        else:
            print(f"{path.name}: VLM critic value-head patch not needed")


if __name__ == "__main__":
    main()
