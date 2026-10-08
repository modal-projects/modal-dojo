"""Compat shim: make slime's torch_dist->HF converter handle Qwen3-VL.

slime's ``tools/convert_torch_dist_to_hf.py`` un-stacks flattened transformer layers
by asserting each stacked layer tensor has ``shape[0] == args.num_layers`` (the LLM
depth). Qwen3-VL's frozen vision tower is *also* stored as stacked layers, but with
the *vision* depth (27) != the LLM's ``num_layers`` (36), so ``get_layer_param`` trips
its assertion before the ``qwen3_vl`` name-mapping (``patch_qwen3_vl_export``) ever
runs.

Fix: skip the frozen vision tower (``vision_model.*``) during conversion and let the
tool's origin-HF fill copy the ViT verbatim from the base HF checkpoint. This is exact
because the ViT is frozen during RL (see ``Qwen3_VL_8B_Recipe.freeze_params_name_list``),
so its weights equal the base weights. The fill only adds tensors whose HF names were
not already produced by the (trained) language-stack conversion, so trained weights
always win.

Baked into the slime base image (only the deploy/eval conversion runs this tool).
Additive + idempotent, so non-VL runs are untouched. Report upstream; drop once
fixed there.

Run at image build:  python patch_qwen3_vl_torch_dist.py
"""

import pathlib
import re

_TOOL = pathlib.Path("/root/slime/tools/convert_torch_dist_to_hf.py")

_SKIP_MARKER = "module.module.vision_model."
_FILL_MARKER = "# qwen3-vl: force origin-HF fill"


def main() -> None:
    if not _TOOL.exists():
        print("compat: torch_dist converter not found at", _TOOL)
        return
    src = _TOOL.read_text()
    changed = False

    # 1) Skip the frozen vision tower in get_layer_param so its stacked layers
    #    (vision depth != LLM num_layers) don't trip the num_layers assertion.
    if _SKIP_MARKER not in src:
        new_src, n = re.subn(
            r"(def get_layer_param\(args, name, param\):\n)([ \t]+)",
            r'\1\2if name.startswith("module.module.vision_model."):\n'
            r"\2    return\n\2",
            src,
            count=1,
        )
        if n != 1:
            print(
                "compat: WARNING - get_layer_param shape changed; skipping vision-skip"
            )
        else:
            src = new_src
            changed = True

    # 2) Force missing-weight fill from the origin HF checkpoint so the skipped
    #    (frozen) vision tower is copied verbatim into the exported model.
    if _FILL_MARKER not in src:
        new_src, n = re.subn(
            r"(\n([ \t]*)args = parser\.parse_args\(\)\n)",
            r"\1\2args.add_missing_from_origin_hf = True  " + _FILL_MARKER + "\n",
            src,
            count=1,
        )
        if n != 1:
            print("compat: WARNING - parse_args anchor changed; skipping fill-force")
        else:
            src = new_src
            changed = True

    if changed:
        _TOOL.write_text(src)
        print(
            "compat: patched torch_dist converter for Qwen3-VL "
            "(vision-skip + origin-HF fill)"
        )
    else:
        print("compat: torch_dist converter already Qwen3-VL-ready")


if __name__ == "__main__":
    main()
