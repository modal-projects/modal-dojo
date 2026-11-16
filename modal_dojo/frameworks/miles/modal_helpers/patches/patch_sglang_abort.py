"""Patch Miles SGLang abort cleanup to tolerate transient router reads.

Rollout can complete successfully and then fail during cleanup if the
SGLang router briefly refuses/interrupts the request that lists its workers
for the abort fan-out. In that case there may be no remaining pending tasks
to abort, and treating the router read as fatal kills the whole Ray job
after useful rollout data has already been collected.

Miles has listed the workers in two shapes, in ``sglang_rollout.abort`` and
(from when it exists) ``inference_rollout_train.get_worker_urls``: by router
version (``/list_workers`` or ``/workers``), and later ``/workers`` alone.
Each known shape is guarded; a file that reads the worker list in any other
shape fails the build.

Executed at image-build time via ``python3 <this file>``.
"""

from __future__ import annotations

import pathlib

MARKER = "Failed to query SGLang workers for abort; continuing without abort fanout"
ROOT = pathlib.Path("/root/miles")
SGLANG_ROLLOUT = "miles/rollout/sglang_rollout.py"
INFERENCE_ROLLOUT_TRAIN = "miles/rollout/inference_rollout/inference_rollout_train.py"
_ROUTER = 'f"http://{args.sglang_router_ip}:{args.sglang_router_port}'
_WARN = f'logger.warning(f"{MARKER}: {{exc}}")'

# By router version, in sglang_rollout.abort and later in get_worker_urls.
_BY_ROUTER_VERSION = (
    f"""    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
        response = await get({_ROUTER}/list_workers")
        urls = response["urls"]
    else:
        response = await get({_ROUTER}/workers")
        urls = [worker["url"] for worker in response["workers"]]
""",
    f"""    try:
        if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
            response = await get({_ROUTER}/list_workers")
            urls = response["urls"]
        else:
            response = await get({_ROUTER}/workers")
            urls = [worker["url"] for worker in response["workers"]]
    except Exception as exc:
        {_WARN}
        urls = []
""",
)

_SHAPES = {
    SGLANG_ROLLOUT: [
        _BY_ROUTER_VERSION,
        (
            f"""    response = await get({_ROUTER}/workers")
    urls = router_worker_base_urls([worker["url"] for worker in response["workers"]])
""",
            f"""    try:
        response = await get({_ROUTER}/workers")
        urls = router_worker_base_urls([worker["url"] for worker in response["workers"]])
    except Exception as exc:
        {_WARN}
        urls = []
""",
        ),
    ],
    INFERENCE_ROLLOUT_TRAIN: [
        _BY_ROUTER_VERSION,
        (
            f"""async def get_worker_urls(args: Namespace):
    response = await get({_ROUTER}/workers")
    return router_worker_base_urls([worker["url"] for worker in response["workers"]])
""",
            f"""async def get_worker_urls(args: Namespace):
    try:
        response = await get({_ROUTER}/workers")
    except Exception as exc:
        {_WARN}
        return []
    return router_worker_base_urls([worker["url"] for worker in response["workers"]])
""",
        ),
    ],
}


def _patch_file(path: pathlib.Path, shapes: list[tuple[str, str]]) -> str:
    src = path.read_text()
    if MARKER in src:
        return "already applied"
    for old, new in shapes:
        if old in src:
            path.write_text(src.replace(old, new, 1))
            return "applied"
    if '/workers")' in src or "/list_workers" in src:
        raise RuntimeError(
            f"{path} reads the SGLang worker list in an unknown shape; "
            "update patch_sglang_abort.py for this miles"
        )
    return "not needed (no worker listing)"


def apply(root: pathlib.Path = ROOT) -> dict[str, str]:
    if not (root / SGLANG_ROLLOUT).exists():
        print(f"{root / SGLANG_ROLLOUT} not found; skipping SGLang abort patch")
        return {}
    outcomes = {
        name: _patch_file(root / name, shapes)
        for name, shapes in _SHAPES.items()
        if (root / name).exists()
    }
    for name, outcome in outcomes.items():
        print(f"SGLang abort patch {outcome}: {name}")
    return outcomes


if __name__ == "__main__":
    apply()
