"""Wait for asynchronous SGLang router registration before serving a cell.

Miles' SGLangRouterApiClient treats HTTP 202 as completed registration. The
router is still discovering engine metadata at that point; a following weight
checksum can block discovery until the job fails, leaving no usable workers.
Only asynchronous registrations need this barrier. Legacy/synchronous router
responses retain their existing behavior.
"""

import ast
from pathlib import Path

TARGET = Path("/root/miles/miles/backends/sglang_utils/sglang_router_api_client.py")
MARKER = "MODAL_DOJO_ROUTER_REGISTRATION_READY_V1"
GUARD = """
        if not use_legacy_api and response.status_code == 202:
            await _dojo_wait_for_router_worker(
                self.router_url, worker_url, response.json().get("worker_id")
            )
"""
HELPER = """

# MODAL_DOJO_ROUTER_REGISTRATION_READY_V1
_DOJO_ROUTER_REGISTRATION_TIMEOUT = 120.0


async def _dojo_wait_for_router_worker(router_url, worker_url, worker_id):
    import asyncio

    last_state = "worker not listed"
    try:
        async with asyncio.timeout(_DOJO_ROUTER_REGISTRATION_TIMEOUT):
            while True:
                response = await GeneralHttpClientProvider.client().get(
                    f"{router_url}/workers", timeout=ROUTER_REQUEST_TIMEOUT
                )
                response.raise_for_status()
                workers = response.json()["workers"]
                matching = [
                    worker for worker in workers
                    if worker.get("url") == worker_url
                    and (worker_id is None or worker.get("id") == worker_id)
                ]
                if any(worker.get("is_healthy") is True for worker in matching):
                    return
                last_state = (
                    f"worker listed but unhealthy (id={worker_id})"
                    if matching else f"worker not listed (id={worker_id})"
                )
                await asyncio.sleep(0.5)
    except TimeoutError as exc:
        raise TimeoutError(
            f"Router {router_url} accepted registration for {worker_url}, but "
            f"it did not become healthy within {_DOJO_ROUTER_REGISTRATION_TIMEOUT}s: "
            f"{last_state}. Check the router AddWorker/discover_metadata logs."
        ) from exc
"""


def apply(path: Path = TARGET) -> bool:
    if not path.exists():
        print(f"{path} not found; skipping router registration patch")
        return False
    source = path.read_text()
    if MARKER in source:
        return False
    tree = ast.parse(source)
    methods = [
        method
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SGLangRouterApiClient"
        for method in node.body
        if isinstance(method, ast.AsyncFunctionDef) and method.name == "add_worker"
    ]
    if (
        len(methods) != 1
        or ast.unparse(methods[0].body[-1]) != "response.raise_for_status()"
    ):
        raise ValueError(f"Unexpected Miles router registration implementation: {path}")
    end = methods[0].end_lineno
    lines = source.splitlines(keepends=True)
    patched = "".join(lines[:end]) + GUARD + "".join(lines[end:]) + HELPER
    compile(patched, str(path), "exec")
    path.write_text(patched)
    print(f"Patched {path} to wait for router registration")
    return True


if __name__ == "__main__":
    apply()
