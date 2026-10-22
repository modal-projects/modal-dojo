"""Opt-in, read-only CUDA snapshots for native-LoRA serving weights.

Built only into the pinned DeepSeek serving environment. Runtime remains unchanged
unless DOJO_REUSE_FROZEN_BASE_BACKUP=1. The first offload writes a snapshot; later
pauses reuse it. Every resume still reads it. CUDA write protection makes a stale
snapshot caused by an unexpected base-weight mutation a fatal error.
"""

import hashlib
import io
import os
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

TMS_REVISION = "b5588e83de86412a48689a6583a4b567e75f7acc"
TMS_ARCHIVE_SHA256 = "e0ad776c9586d9e6ded1c1edff0df03d785130799f1b89de57fa7c797235646e"
TMS_ARCHIVE_URL = (
    f"https://api.github.com/repos/fzyzcjy/torch_memory_saver/tarball/{TMS_REVISION}"
)
VERSION = "0.0.9.post1+dojo.frozen1"
SG_ADAPTER = Path(
    "/sgl-workspace/sglang/python/sglang/srt/utils/torch_memory_saver_adapter.py"
)
SG_PYTHON = Path("/opt/sglang/bin/python")
MARKER = "# Modal Dojo: opt-in immutable serving-weight disk snapshots."


def _replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Frozen-backup source anchor changed: {old[:100]!r}")
    return text.replace(old, new)


def patch_source(root):
    """Patch the verified source; validate every anchor before writing any file."""
    root = Path(root)
    changes = {}

    def edit(name, old, new):
        changes[name] = _replace(changes.get(name, (root / name).read_text()), old, new)

    edit(
        "csrc/core.h",
        "    CUmemGenericAllocationHandle allocHandle;\n#endif\n};",
        "    CUmemGenericAllocationHandle allocHandle;\n#endif\n"
        "    bool disk_backup_readonly = false;\n    bool disk_backup_valid = false;\n};",
    )
    edit(
        "csrc/core.h",
        "    void pause(const std::string& tag);",
        "    uint64_t seal_disk_backup(const std::string& tag);\n    void pause(const std::string& tag);",
    )
    edit(
        "csrc/core.cpp",
        "void TorchMemorySaver::pause(const std::string& tag) {",
        SEAL_CPP + "\nvoid TorchMemorySaver::pause(const std::string& tag) {",
    )
    edit(
        "csrc/core.cpp",
        "            disk_backend_.offload(ptr, metadata.raw_size, metadata.disk);",
        """            if (metadata.disk_backup_readonly && metadata.disk_backup_valid) {
                reused_bytes += metadata.raw_size;
            } else {
                disk_backend_.offload(ptr, metadata.raw_size, metadata.disk);
                written_bytes += metadata.raw_size;
                metadata.disk_backup_valid = metadata.disk_backup_readonly;
            }""",
    )
    edit(
        "csrc/core.cpp",
        "void TorchMemorySaver::pause(const std::string& tag) {\n",
        "void TorchMemorySaver::pause(const std::string& tag) {\n    uint64_t written_bytes = 0, reused_bytes = 0;\n",
    )
    edit(
        "csrc/core.cpp",
        "    }\n#endif\n}\n\nvoid TorchMemorySaver::resume",
        """    }
    if (written_bytes || reused_bytes) {
        std::cout << "[dojo_frozen_backup] tag=" << tag
                  << " written_bytes=" << written_bytes
                  << " reused_bytes=" << reused_bytes << std::endl;
    }
#endif
}

void TorchMemorySaver::resume""",
    )
    edit(
        "csrc/core.cpp",
        "            disk_backend_.onload(ptr, metadata.raw_size, metadata.disk);",
        """            disk_backend_.onload(ptr, metadata.raw_size, metadata.disk);
#ifdef USE_CUDA
            if (metadata.disk_backup_readonly) {
                CUDAUtils::cu_mem_set_readonly(ptr, metadata.allocation_size, metadata.device);
            }
#endif""",
    )
    edit(
        "csrc/utils.h",
        "    static CUdevice cu_ctx_get_device() {\n        CUdevice ans;",
        """    static void cu_mem_set_readonly(void *ptr, size_t size, CUdevice device) {
        CUmemAccessDesc access_desc = {};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
        CURESULT_CHECK(cuMemSetAccess((CUdeviceptr)ptr, size, &access_desc, 1));
    }

    static CUdevice cu_ctx_get_device() {
        CUdevice ans;""",
    )
    edit(
        "csrc/entrypoint.cpp",
        "void tms_pause(const char* tag) {",
        """uint64_t tms_seal_disk_backup(const char* tag) {
    SIMPLE_CHECK(tag != nullptr, "seal requires a nonempty tag");
    return TorchMemorySaver::instance().seal_disk_backup(std::string(tag));
}

void tms_pause(const char* tag) {""",
    )
    edit(
        "csrc/disk_backend.cpp",
        "    fsync(fd);",
        '    SIMPLE_CHECK(fsync(fd) == 0, "disk backup fsync failed");',
    )
    edit(
        "torch_memory_saver/binary_wrapper.py",
        "    cdll.tms_pause.argtypes = [ctypes.c_char_p]",
        "    cdll.tms_seal_disk_backup.argtypes = [ctypes.c_char_p]\n"
        "    cdll.tms_seal_disk_backup.restype = ctypes.c_uint64\n"
        "    cdll.tms_pause.argtypes = [ctypes.c_char_p]",
    )
    edit(
        "torch_memory_saver/entrypoint.py",
        "    def pause(self, tag: Optional[str] = None):",
        '''    def seal_disk_backup(self, tag: str) -> int:
        """Permanently write-protect current disk-backed allocations in this tag.

        Invoke only while the serving worker is quiescent. Unsupported backends,
        empty tags, mixed backup modes and cross-device pools fail closed.
        New allocations must be sealed again before their first reusable offload.
        """
        if not isinstance(tag, str) or not tag or "\\0" in tag:
            raise ValueError("seal requires a nonempty tag without NUL")
        if torch.version.hip or not torch.version.cuda:
            raise RuntimeError("immutable disk snapshots require CUDA")
        self._ensure_initialized()
        return self._impl._binary_wrapper.cdll.tms_seal_disk_backup(tag.encode("utf-8"))

    def pause(self, tag: Optional[str] = None):''',
    )
    edit("setup.py", "version='0.0.9.post1'", f"version='{VERSION}'")
    for name, text in changes.items():
        if name.endswith(".py"):
            compile(text, name, "exec")
    for name, text in changes.items():
        (root / name).write_text(text)


SEAL_CPP = r"""
uint64_t TorchMemorySaver::seal_disk_backup(const std::string& tag) {
#ifdef USE_CUDA
    SIMPLE_CHECK(!tag.empty(), "seal requires a nonempty tag");
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    const CUdevice current_device = CUDAUtils::cu_ctx_get_device();
    uint64_t bytes = 0;
    // Validate the entire pool before changing any access permissions.
    for (const auto& item : allocation_metadata_) {
        const auto& m = item.second;
        if (m.tag != tag) continue;
        SIMPLE_CHECK(m.state == AllocationState::ACTIVE, "seal requires active allocations");
        SIMPLE_CHECK(m.enable_disk_backup && !m.enable_cpu_backup,
                     "seal requires exclusively disk-backed allocations");
        SIMPLE_CHECK(m.device == current_device, "seal requires a single-device pool");
        bytes += m.raw_size;
    }
    SIMPLE_CHECK(bytes > 0, "seal found no allocations");
    for (auto& item : allocation_metadata_) {
        auto& m = item.second;
        if (m.tag != tag || m.disk_backup_readonly) continue;
        CUDAUtils::cu_mem_set_readonly(item.first, m.allocation_size, m.device);
        m.disk_backup_readonly = true;
        // A snapshot taken before the seal could already be stale.
        m.disk_backup_valid = false;
    }
    return bytes;
#else
    SIMPLE_CHECK(false, "immutable disk snapshots require CUDA");
    return 0;
#endif
}
"""


def patch_adapter(path=SG_ADAPTER):
    path = Path(path)
    text = path.read_text()
    if MARKER in text:
        return
    if "PATCHED_MODAL_DOJO_SGLANG_WEIGHTS_DISK_BACKUP" not in text:
        raise ValueError("Pinned Dojo disk-backup patch must be installed first")
    text = _replace(
        text,
        "    def pause(self, tag: str):\n        return _memory_saver.pause(tag=tag)",
        f"""    def pause(self, tag: str):
        {MARKER}
        import os

        if os.environ.get("DOJO_REUSE_FROZEN_BASE_BACKUP", "0") == "1" and tag == "weights":
            from sglang.srt.runtime_context import get_lora, get_model

            if not get_lora().enable_lora or get_model().weight_cache_mode != "off":
                raise RuntimeError("Frozen backup reuse requires native LoRA without IPC weight caching")
            if not os.environ.get("DOJO_SGLANG_WEIGHTS_DISK_BACKUP_DIR"):
                raise RuntimeError("Frozen backup reuse requires disk-backed serving weights")
            sealed_bytes = _memory_saver.seal_disk_backup(tag=tag)
            logger.info("[dojo_frozen_backup] sealed_bytes=%s", sealed_bytes)
        return _memory_saver.pause(tag=tag)""",
    )
    compile(text, str(path), "exec")
    path.write_text(text)


def install(python=SG_PYTHON, adapter=SG_ADAPTER):
    """Build the pinned CUDA extension in the serving venv, without a GPU."""
    python = str(python)
    version = subprocess.check_output(
        [
            python,
            "-c",
            "from importlib.metadata import version; print(version('torch-memory-saver'))",
        ],
        text=True,
    ).strip()
    if version != VERSION:
        if version != "0.0.9.post1":
            raise ValueError(f"Unexpected torch-memory-saver version: {version}")
        major = subprocess.check_output(
            [
                python,
                "-c",
                "import torch; assert not torch.version.hip; print(torch.version.cuda.split('.')[0])",
            ],
            text=True,
        ).strip()
        with urllib.request.urlopen(TMS_ARCHIVE_URL, timeout=120) as response:
            archive = response.read()
        if hashlib.sha256(archive).hexdigest() != TMS_ARCHIVE_SHA256:
            raise ValueError("Pinned torch-memory-saver archive checksum mismatch")
        with tempfile.TemporaryDirectory(prefix="dojo-frozen-backup-") as tmp:
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
                bundle.extractall(tmp, filter="data")
            roots = list(Path(tmp).iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise ValueError("Unexpected torch-memory-saver archive layout")
            patch_source(roots[0])
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    python,
                    "--no-deps",
                    str(roots[0]),
                ],
                env={**os.environ, "TMS_CUDA_MAJOR": major},
                check=True,
            )
    patch_adapter(adapter)


if __name__ == "__main__":
    install()
