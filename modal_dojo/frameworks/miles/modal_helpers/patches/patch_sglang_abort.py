"""Patch Miles SGLang abort cleanup to tolerate transient router reads.

Rollout can complete successfully and then fail during cleanup if the
SGLang router briefly refuses/interrupts the ``/list_workers`` request
used to fan out aborts. In that case there may be no remaining pending
tasks to abort, and treating the router read as fatal kills the whole
Ray job after useful rollout data has already been collected.

Executed at image-build time via ``python3 <this file>``.
"""

from pathlib import Path

path = Path("/root/miles/miles/rollout/sglang_rollout.py")
marker = "Failed to query SGLang workers for abort; continuing without abort fanout"
if not path.exists():
    print(f"{path} not found; skipping SGLang abort patch")
    raise SystemExit(0)

src = path.read_text()
old = """    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]
"""
new = """    try:
        if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
            response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
            urls = response["urls"]
        else:
            response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
            urls = [worker["url"] for worker in response["workers"]]
    except Exception as exc:
        logger.warning(f"Failed to query SGLang workers for abort; continuing without abort fanout: {exc}")
        urls = []
"""

if marker in src:
    print("SGLang abort patch already applied")
elif old in src:
    path.write_text(src.replace(old, new, 1))
    print("Patched Miles SGLang abort cleanup")
else:
    raise RuntimeError("Could not find Miles SGLang abort block to patch")
