"""
image: radixark/miles:dev-202609251434
commit: https://github.com/sgl-project/sglang/commit/880e3d2453eb7ef1738350e8c35ba2b956cc93a9
file: python/sglang/srt/entrypoints/http_server.py::_freeze_gc_after_server_warmup
"""

from pathlib import Path

TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py")
MARKER = "PATCHED_SGLANG_FREEZE_GC_READY"
ANCHOR = """    try:
        res = requests.post(
            server_args.url() + "/freeze_gc",
            headers=freeze_headers,
            timeout=10,
            verify=ssl_verify_of(server_args),
        )
        res.raise_for_status()
    except requests.exceptions.RequestException:
        logger.warning("post-warmup freeze_gc failed", exc_info=True)
"""
REPLACEMENT = f"""    # {MARKER}
    # Skipping generation warmup can reach here before Uvicorn binds its socket.
    # /ready checks HTTP readiness without issuing a cold generation request.
    import time as _gc_time

    deadline = _gc_time.monotonic() + 30
    while True:
        try:
            remaining = max(0.01, deadline - _gc_time.monotonic())
            ready = requests.get(
                server_args.url() + "/ready",
                headers=freeze_headers,
                timeout=min(2, remaining),
                verify=ssl_verify_of(server_args),
            )
            ready.raise_for_status()
            res = requests.post(
                server_args.url() + "/freeze_gc",
                headers=freeze_headers,
                timeout=min(10, max(0.01, deadline - _gc_time.monotonic())),
                verify=ssl_verify_of(server_args),
            )
            res.raise_for_status()
            return
        except requests.exceptions.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403, 404) or _gc_time.monotonic() >= deadline:
                logger.warning("post-warmup freeze_gc failed", exc_info=True)
                return
            _gc_time.sleep(min(0.25, max(0, deadline - _gc_time.monotonic())))
"""


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        if source.count(REPLACEMENT) != 1:
            raise RuntimeError(f"{target}: unexpected patched freeze_gc source")
        return
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: expected one freeze_gc request")
    source = source.replace(ANCHOR, REPLACEMENT)
    compile(source, str(target), "exec")
    target.write_text(source)


if __name__ == "__main__":
    apply()
