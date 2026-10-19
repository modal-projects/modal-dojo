"""
image: radixark/miles:dev-202609251434
commit: https://github.com/sgl-project/sglang/commit/880e3d2453eb7ef1738350e8c35ba2b956cc93a9
file: python/sglang/srt/state_capturer/routed_experts.py::RoutedExpertsCapturer.__init__
"""

import pathlib

MARKER = "PATCHED_K3_ROUTED_EXPERTS_TOPK"

TARGET = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/state_capturer/routed_experts.py"
)
ANCHOR = "        topk_size = model_config.hf_text_config.num_experts_per_tok\n"
REPLACEMENT = f"""        # {MARKER}: KimiLinearConfig names this num_experts_per_token.
        text_config = model_config.hf_text_config
        topk_size = (
            text_config.num_experts_per_token
            if hasattr(text_config, "num_experts_per_token")
            else text_config.num_experts_per_tok
        )
"""

if not TARGET.exists():
    print(f"{TARGET} not found; skipping K3 routed experts topk patch")
    raise SystemExit(0)

src = TARGET.read_text()
if MARKER in src:
    print("K3 routed experts topk patch already applied")
    raise SystemExit(0)

if src.count(ANCHOR) != 1:
    raise SystemExit(
        "K3 routed experts topk patch did not match; sglang's routed_experts.py "
        "has changed. Re-check RoutedExpertsCapturer.__init__ before shipping."
    )

TARGET.write_text(src.replace(ANCHOR, REPLACEMENT))
print("Patched RoutedExpertsCapturer topk_size for KimiLinearConfig")
