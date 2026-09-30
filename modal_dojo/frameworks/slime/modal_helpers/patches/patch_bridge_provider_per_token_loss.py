"""Patch slime's bridge model provider to propagate missing config fields.

The bridge code path in ``model_provider.py`` copies parallelism fields
(``context_parallel_size``, ``tensor_model_parallel_size``, etc.) from
``args`` to the Megatron ``provider`` object, but omits:

- ``calculate_per_token_loss``: defaults to ``False`` in TransformerConfig,
  triggers an assertion in the Qwen3-VL model when CP > 1.
- ``attention_backend``: defaults to ``auto`` in TransformerConfig, which
  selects cuDNN fused attention for CP — this hits a stream mismatch bug.
  Propagating ``flash`` from args makes CP use flash-attn instead.

This patch inserts the missing assignments right before ``provider.finalize()``.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

p = pathlib.Path("/root/slime/slime/backends/megatron_utils/model_provider.py")
if not p.exists():
    print("WARNING: model_provider.py not found, skipping per-token-loss patch")
else:
    src = p.read_text()
    marker = "PATCHED_BRIDGE_PROVIDER_CONFIG"
    if marker in src:
        print("model_provider.py already patched for bridge provider config")
    else:
        old = "        provider.finalize()"
        new = (
            "        provider.calculate_per_token_loss = getattr("
            "args, 'calculate_per_token_loss', False)"
            f"  # {marker}\n"
            "        provider.attention_backend = getattr("
            "args, 'attention_backend', None)"
            f"  # {marker}\n"
            "        provider.finalize()"
        )
        if old in src:
            new_src = src.replace(old, new, 1)
            p.write_text(new_src)
            print(
                "Patched model_provider.py: added calculate_per_token_loss "
                "and attention_backend propagation to bridge provider"
            )
        else:
            print("WARNING: Could not find 'provider.finalize()' in model_provider.py")
