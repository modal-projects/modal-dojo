"""Allow training to opt out of TE's FA4 without disabling SGLang's FA4.

image: radixark/miles:muse-spark-open (digest pinned in muse_spark.py)
commit: https://github.com/radixark/TransformerEngine/commit/ae9d2cc80e2a73ddbae5b7c2d290bb644230b2f6
file: transformer_engine/pytorch/attention/dot_product_attention/utils.py

FA4 4.0.0b35 with quack 0.6.5 fails the Muse packed-BF16 attention reference
check on B300. The pinned TE fork has no per-version switch;
NVTE_FLASH_ATTN=0 would disable the working FA2 fallback too.

TODO: remove this patch and set NVTE_FLASH_ATTN_V4=0 in the recipe when the
Miles image includes NVIDIA/TransformerEngine#3204's native selector:
https://github.com/NVIDIA/TransformerEngine/pull/3204
Upstream also fixed FA4 window sentinels in #3540; rerun the Muse attention
forward/backward checks before enabling FA4 training:
https://github.com/NVIDIA/TransformerEngine/pull/3540
"""


def apply() -> None:
    import importlib.metadata
    import importlib.util
    from pathlib import Path

    assert importlib.metadata.version("transformer-engine") == "2.17.0+miles"
    spec = importlib.util.find_spec("transformer_engine")
    path = (
        Path(next(iter(spec.submodule_search_locations)))
        / "pytorch/attention/dot_product_attention/utils.py"
    )
    source = path.read_text()
    old = "    use_flash_attention_4 = use_flash_attention\n"
    new = (
        "    # Dojo Muse: the pinned FA4 fails the B300 training attention check.\n"
        "    # Keep FA4 installed for SGLang; allow only TE callers to opt out.\n"
        '    use_flash_attention_4 = use_flash_attention and int(os.getenv("DOJO_MUSE_TE_FA4", "1"))\n'
    )
    if new not in source:
        assert source.count(old) == 1, "TE FA4 selection source changed"
        source = source.replace(old, new)
        source = source.replace(
            'logger.debug("Disabling FlashAttention 4 due to NVTE_FLASH_ATTN=0")',
            'logger.debug("Disabling FlashAttention 4 due to NVTE_FLASH_ATTN=0 or DOJO_MUSE_TE_FA4=0")',
        )
        compile(source, str(path), "exec")
        path.write_text(source)
    print("Dojo Muse TE FA4 gate installed:", path)


if __name__ == "__main__":
    apply()
