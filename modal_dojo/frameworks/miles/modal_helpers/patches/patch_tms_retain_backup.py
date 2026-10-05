"""Retain selected pinned host buffers without ever reusing stale contents.

Source: fzyzcjy/torch_memory_saver@b5588e83de86412a48689a6583a4b567e75f7acc.
Opt in with DOJO_TMS_RETAIN_BACKUP_TAG; the default allocator is unchanged.
Both pause and resume still copy all bytes. Only cudaMallocHost/cudaFreeHost
churn is removed, so mutable buffers and weight updates remain correct.
"""

from pathlib import Path

TARGET = Path("/tmp/dojo-tms/csrc/core.cpp")
MARKER = "DOJO_TMS_RETAIN_BACKUP_TAG"
ANCHOR = """            // TODO may provide a flag to choose whether to free immediately
            // (users may want to lazily free to reduce re-alloc time)
            CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
            metadata.cpu_backup = nullptr;
"""
REPLACEMENT = """            // Keep the allocation, not its contents: pause always refreshes it.
            const char* retained_tag = std::getenv("DOJO_TMS_RETAIN_BACKUP_TAG");
            if (retained_tag == nullptr || metadata.tag != retained_tag) {
                CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
                metadata.cpu_backup = nullptr;
            }
"""


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        return
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: unexpected memory-saver source")
    target.write_text("#include <cstdlib>\n" + source.replace(ANCHOR, REPLACEMENT))
    print(f"Patched {target}: opt-in retained host allocations; copies unchanged")


if __name__ == "__main__":
    apply()
