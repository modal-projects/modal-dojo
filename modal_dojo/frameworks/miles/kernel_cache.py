"""Node-local kernel caches with best-effort persistent snapshots.

Compile/cache lookup never writes small files directly to the shared Volume.
Each node publishes its own archive; a stable local path preserves absolute
paths embedded in Triton group manifests across container restarts.
"""

import json
import logging
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import threading
import time

logger = logging.getLogger(__name__)
CACHE_KEYS = (
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TILELANG_CACHE_DIR",
    "SGLANG_CACHE_DIR",
)


class KernelCache:
    def __init__(self, environment, *, node_rank, gpu_type):
        self.environment = dict(environment)
        self.root = Path(environment["DOJO_LOCAL_KERNEL_CACHE"]) / gpu_type
        self.sources = {k: Path(environment[k]) for k in CACHE_KEYS if k in environment}
        persistent = Path(os.path.commonpath([str(p) for p in self.sources.values()]))
        # Snapshot namespace is distinct from the existing raw cache trees.
        self.archive = (
            persistent / "node-snapshots" / gpu_type / f"node-{node_rank}.tar"
        )
        self._lock = threading.Lock()
        self._fingerprint = None
        self._stop = threading.Event()
        self._thread = None

    def prepare(self):
        started = time.monotonic()
        self.root.mkdir(parents=True, exist_ok=True)
        restored = False
        if self.archive.exists():
            try:
                with tempfile.TemporaryDirectory(dir=self.root.parent) as staging:
                    with tarfile.open(self.archive) as archive:
                        archive.extractall(staging, filter="data")
                    shutil.copytree(staging, self.root, dirs_exist_ok=True)
                restored = True
            except (OSError, tarfile.TarError):
                logger.exception(
                    "Kernel cache snapshot unreadable; seeding legacy cache"
                )
        for key, source in self.sources.items():
            local = self.root / key.lower()
            if not restored and source.exists():
                shutil.copytree(
                    source,
                    local,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(
                        "tmp.*", "*.lock", "lock", "node-snapshots"
                    ),
                )
                # Legacy Triton manifests name the old Volume paths. Rewrite
                # only child_paths, never arbitrary compiler metadata.
                if key == "TRITON_CACHE_DIR":
                    for manifest in local.rglob("__grp__*"):
                        try:
                            data = json.loads(manifest.read_text())
                            data["child_paths"] = {
                                name: str(local / Path(path).relative_to(source))
                                for name, path in data["child_paths"].items()
                            }
                            manifest.write_text(json.dumps(data))
                        except (OSError, ValueError, KeyError, TypeError):
                            # An incomplete legacy group is a cache miss.
                            manifest.unlink(missing_ok=True)
            local.mkdir(parents=True, exist_ok=True)
            self.environment[key] = str(local)
        print(
            f"DOJO_KERNEL_CACHE restore={restored} "
            f"elapsed_s={time.monotonic() - started:.3f} root={self.root}",
            flush=True,
        )
        return self.environment

    def snapshot(self):
        with self._lock:
            files = sorted(
                p
                for p in self.root.rglob("*")
                if p.is_file()
                and not p.is_symlink()
                and not any(
                    part.startswith("tmp.") or part.endswith(".lock") or part == "lock"
                    for part in p.relative_to(self.root).parts
                )
            )
            fingerprint = [
                (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files
            ]
            if fingerprint == self._fingerprint:
                return
            # Build on local disk; only the final archive crosses the Volume.
            temporary = self.root.with_suffix(".tar.tmp")
            with tarfile.open(temporary, "w") as archive:
                for path in files:
                    archive.add(
                        path, arcname=str(path.relative_to(self.root)), recursive=False
                    )
            self.archive.parent.mkdir(parents=True, exist_ok=True)
            dest = self.archive.with_suffix(f".tmp.{os.getpid()}")
            shutil.copyfile(temporary, dest)
            os.replace(dest, self.archive)
            temporary.unlink()
            self._fingerprint = fingerprint
            print(
                f"DOJO_KERNEL_CACHE saved files={len(files)} archive={self.archive}",
                flush=True,
            )

    def start(self):
        def publish():
            while not self._stop.wait(300):
                try:
                    self.snapshot()
                except Exception:
                    logger.exception("Kernel cache snapshot failed; training continues")

        self._thread = threading.Thread(
            target=publish, daemon=True, name="kernel-cache"
        )
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        try:
            self.snapshot()
        except Exception:
            logger.exception("Final kernel cache snapshot failed; training continues")


def prepare_kernel_cache(environment, *, node_rank, gpu_type):
    if not environment.get("DOJO_LOCAL_KERNEL_CACHE"):
        return dict(environment), None
    try:
        cache = KernelCache(environment, node_rank=node_rank, gpu_type=gpu_type)
        mapped = cache.prepare()
    except Exception:
        logger.exception(
            "Local kernel cache preparation failed; using configured caches"
        )
        return dict(environment), None
    cache.start()
    return mapped, cache
