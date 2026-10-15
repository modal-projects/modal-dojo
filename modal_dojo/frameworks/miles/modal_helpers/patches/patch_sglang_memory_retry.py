from pathlib import Path

TARGET = Path(
    "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/weight_updater.py"
)
REPLACEMENTS = (
    (
        """        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
""",
        """        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES
        tags = [tag for tag in dict.fromkeys(tags) if tag not in self.offload_tags]

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
""",
    ),
    (
        """        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)
""",
        """        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES
        tags = [tag for tag in dict.fromkeys(tags) if tag in self.offload_tags]

        for tag in tags:
            self.offload_tags.remove(tag)
""",
    ),
)


def patch_source(source: str) -> str:
    for old, new in REPLACEMENTS:
        if source.count(new) == 1:
            continue
        if source.count(old) != 1:
            raise ValueError("SGLang memory retry patch anchor changed")
        source = source.replace(old, new, 1)
    compile(source, str(TARGET), "exec")
    return source


if __name__ == "__main__":
    TARGET.write_text(patch_source(TARGET.read_text()))
    print(f"Patched {TARGET}: make memory release and resume retries idempotent")
