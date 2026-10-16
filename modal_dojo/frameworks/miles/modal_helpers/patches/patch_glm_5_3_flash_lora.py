"""Port GLM-5.3 LoRA geometry and gate registration from SGLang #37753.

image: radixark/miles:dev-202610140350
sglang: f31278164939dd0746037abbb7566de4164e6767

The pinned SGLang already keeps KDA projections unfused with LoRA enabled
and handles dense/shared/routed MLP dimensions. Port the remaining per-layer
attention dimensions and replicated/attention-TP gate registrations.
TODO: remove once sgl-project/sglang#37753 (including #42795's geometry fix)
is included in the pinned SGLang source.
"""

from pathlib import Path

MARKER = "PATCHED_MODAL_DOJO_GLM53_LORA"
GATES = '"b_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj"'
GEOMETRY = f"""    supported_lora_modules = [
        "qkv_proj", "o_proj", {GATES},
        "fused_qkv_a_proj_with_mqa", "q_b_proj", "kv_b_proj",
        "gate_up_proj", "down_proj", "embed_tokens", "lm_head",
    ]

    def get_hidden_dim(self, module_name: str, layer_idx: int):
        from sglang.srt.lora.utils import get_default_hidden_dim

        cfg = self.config
        heads = cfg.linear_attn_config["num_heads"]
        head_dim = cfg.linear_attn_config["head_dim"]
        width = heads * head_dim
        if module_name == "qkv_proj":
            return cfg.hidden_size, 3 * width
        if module_name == "o_proj" and cfg.is_kda_layer(layer_idx):
            return width, cfg.hidden_size
        if module_name in ("f_a_proj", "g_a_proj"):
            return cfg.hidden_size, head_dim
        if module_name in ("f_b_proj", "g_b_proj"):
            return head_dim, width
        if module_name == "b_proj":
            return cfg.hidden_size, heads
        return get_default_hidden_dim(module_name, cfg, layer_idx)

"""
GATE_GEOMETRY = """    elif module_name in KDA_GATE_LORA_NAMES:
        linear = getattr(config, "linear_attn_config", None) or {}
        heads = linear.get("num_heads") or getattr(config, "linear_num_heads", None)
        head_dim = linear.get("head_dim") or getattr(config, "linear_head_dim", None)
        if heads is None or head_dim is None:
            raise NotImplementedError(f"{module_name} needs KDA head geometry")
        if module_name in ("f_a_proj", "g_a_proj"):
            return config.hidden_size, head_dim
        if module_name in ("f_b_proj", "g_b_proj"):
            return head_dim, heads * head_dim
        return config.hidden_size, heads
"""
REPLACEMENTS = {
    "models/glm5_next.py": [
        (
            "class Glm5NextForConditionalGeneration(nn.Module):\n",
            "class Glm5NextForConditionalGeneration(nn.Module):\n" + GEOMETRY,
        ),
    ],
    "lora/utils.py": [
        (
            '    elif module_name == "gate_up_proj_moe":\n',
            GATE_GEOMETRY + '    elif module_name == "gate_up_proj_moe":\n',
        ),
        (
            "REPLICATED_LINEAR_LORA_NAMES = [\n",
            f"KDA_GATE_LORA_NAMES = frozenset([{GATES}])\n"
            'REPLICATED_LINEAR_LORA_NAMES = [\n    "f_a_proj", "g_a_proj",\n',
        ),
        (
            "ATTN_TP_LORA_MODULE_NAMES = frozenset(\n    {\n",
            "ATTN_TP_LORA_MODULE_NAMES = frozenset(\n    {\n"
            '        "b_proj", "f_b_proj", "g_b_proj",\n',
        ),
        (
            "_KNOWN_LORA_TARGET_MODULES = frozenset(\n    {\n",
            f"_KNOWN_LORA_TARGET_MODULES = frozenset(\n    {{\n        {GATES},\n",
        ),
    ],
    "utils/common.py": [
        (
            "SUPPORTED_LORA_TARGET_MODULES = [\n",
            f"SUPPORTED_LORA_TARGET_MODULES = [\n    {GATES},\n",
        ),
    ],
}


def patch_source(source: str, name: str) -> str:
    if MARKER in source:
        return source
    for old, new in REPLACEMENTS[name]:
        if source.count(old) != 1:
            raise ValueError(f"GLM LoRA patch anchor changed: {name}: {old!r}")
        source = source.replace(old, new, 1)
    return f"# {MARKER}\n" + source


def main() -> None:
    root = Path("/sgl-workspace/sglang/python/sglang/srt")
    patched = {
        root / name: patch_source((root / name).read_text(), name)
        for name in REPLACEMENTS
    }
    for path, source in patched.items():
        compile(source, str(path), "exec")
    for path, source in patched.items():
        path.write_text(source)
        print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
