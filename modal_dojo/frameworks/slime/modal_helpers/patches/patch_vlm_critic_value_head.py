"""Fix slime's bridge-mode PPO critic for VLMs such as Qwen3-VL.

The critic's value head is assigned to the wrapper model, but the wrapper's forward()
runs ``language_model``'s own vocab-sized head. Attach the value head to
``language_model`` instead, and let the critic's HF load skip the policy ``lm_head``
(registering the value-head class, which the image's Megatron-Bridge doesn't know).
"""

from __future__ import annotations

from pathlib import Path

MARKER = "PATCHED_VLM_CRITIC_VALUE_HEAD"
SLIME = Path("/root/slime/slime/backends/megatron_utils")

PROVIDER_ANCHOR = """\
                if post_process:
                    model.output_layer = LinearForLastLayer(
                        input_size=model.config.hidden_size, output_size=1, config=model.config
                    )
"""
PROVIDER_REPLACEMENT = f"""\
                if post_process:
                    # {MARKER}: VLM wrappers run the LM head inside language_model.
                    head_owner = getattr(model, "language_model", None) or model
                    head_owner.output_layer = LinearForLastLayer(
                        input_size=head_owner.config.hidden_size, output_size=1, config=head_owner.config
                    )
"""
LOAD_ANCHOR = """\
        bridge.load_hf_weights(ddp_model)
"""
LOAD_REPLACEMENT = f"""\
        # {MARKER}: the policy lm_head does not fit the critic's 1-output value head.
        allowed_mismatched_params = None
        if getattr(ddp_model[0], "role", None) == "critic":
            from megatron.bridge.models.conversion.param_mapping import AutoMapping

            AutoMapping.register_module_type("LinearForLastLayer", "replicated")
            allowed_mismatched_params = ["*output_layer.weight"]
        bridge.load_hf_weights(ddp_model, allowed_mismatched_params=allowed_mismatched_params)
"""


def _patch_file(path: Path, anchor: str, replacement: str) -> None:
    source = path.read_text()
    if MARKER in source:
        print(f"{path.name} already patched for VLM critic value head")
        return
    if anchor not in source:
        raise RuntimeError(f"Could not find VLM critic value-head anchor in {path}")
    path.write_text(source.replace(anchor, replacement, 1))
    print(f"Patched {path} for VLM critic value head")


def main() -> None:
    _patch_file(SLIME / "model_provider.py", PROVIDER_ANCHOR, PROVIDER_REPLACEMENT)
    _patch_file(SLIME / "checkpoint.py", LOAD_ANCHOR, LOAD_REPLACEMENT)


if __name__ == "__main__":
    main()
