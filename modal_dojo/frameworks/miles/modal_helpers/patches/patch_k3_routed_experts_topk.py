"""
image: radixark/miles:dev-202609251434
commit: https://github.com/sgl-project/sglang/commit/880e3d2453eb7ef1738350e8c35ba2b956cc93a9
files:
  python/sglang/srt/state_capturer/routed_experts.py::RoutedExpertsCapturer.__init__
  python/sglang/srt/models/kimi_k3.py (fused MoE TopK construction)

Kimi-K3 routing replay (use_rollout_routing_replay) needs two sglang fixes:
the capturer sizes its buffer from num_experts_per_tok, which KimiLinearConfig
names num_experts_per_token, and K3's TopK is built without layer_id, so the
capture writes buffer[:batch, None, :] and fails during cuda-graph capture.
Upstream fixed the latter in sglang#32541 follow-ups by passing layer_id.
"""

import pathlib

MARKER = "PATCHED_K3_ROUTED_EXPERTS_TOPK"
SGLANG = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt")

EDITS = (
    (
        SGLANG / "state_capturer/routed_experts.py",
        "        topk_size = model_config.hf_text_config.num_experts_per_tok\n",
        f"""        # {MARKER}: KimiLinearConfig names this num_experts_per_token.
        text_config = model_config.hf_text_config
        topk_size = (
            text_config.num_experts_per_token
            if hasattr(text_config, "num_experts_per_token")
            else text_config.num_experts_per_tok
        )
""",
    ),
    (
        SGLANG / "models/kimi_k3.py",
        "        self.topk = TopK(\n            top_k=config.num_experts_per_token,\n",
        f"""        self.topk = TopK(
            top_k=config.num_experts_per_token,
            layer_id=self.layer_idx,  # {MARKER}
""",
    ),
)

for target, anchor, replacement in EDITS:
    if not target.exists():
        print(f"{target} not found; skipping K3 routed experts topk patch")
        continue
    src = target.read_text()
    if MARKER in src:
        print(f"K3 routed experts topk patch already applied to {target.name}")
        continue
    if src.count(anchor) != 1:
        raise SystemExit(
            f"K3 routed experts topk patch did not match in {target}; sglang has "
            "changed. Re-check the anchors before shipping."
        )
    target.write_text(src.replace(anchor, replacement))
    print(f"Patched {target.name} for K3 routed experts capture")
