"""Golden-file tests for the miles rollout-status and advantage patchers."""

from __future__ import annotations

from pathlib import Path

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_advantage_distribution as advantage_patcher,
    patch_qkvr_cpu_merge as qkvr_patcher,
    patch_rollout_status_reporting as rollout_patcher,
)
from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_sglang_abort as abort_patcher,
)

TESTDATA = Path(__file__).parent / "testdata" / "miles"

STATUS_PATCH_FILES = ("train.py", "train_async.py", "log_utils.py")


@pytest.fixture(scope="session")
def miles_inputs() -> dict[str, str]:
    inputs = [TESTDATA / f"{name}.input" for name in STATUS_PATCH_FILES]
    assert inputs
    return {path.name.removesuffix(".input"): path.read_text() for path in inputs}


def test_all_miles_goldens_compile():
    for golden in TESTDATA.glob("*.output"):
        compile(golden.read_text(), str(golden), "exec")


def _apply_patcher(name: str, work: Path) -> None:
    if name == "log_utils.py":
        advantage_patcher._patch_file(work)
    else:
        rollout_patcher._patch_file(work)


def test_missing_patch_targets_are_skipped(tmp_path, capsys):
    missing = tmp_path / "train.py"
    rollout_patcher._patch_file(missing)
    assert not missing.exists()
    assert "not found, skipping rollout-status patch" in capsys.readouterr().out

    advantage_patcher._patch_file(missing)
    assert not missing.exists()
    assert "not found, skipping advantage-distribution patch" in capsys.readouterr().out


def test_patch_matches_golden(miles_inputs, tmp_path, request):
    rewrite_goldens = request.config.getoption("--rewrite")
    for name, source in miles_inputs.items():
        golden_path = TESTDATA / (
            f"{name}.advantage.output"
            if name == "log_utils.py"
            else f"{name}.status.output"
        )
        work = tmp_path / name
        work.write_text(source)
        _apply_patcher(name, work)
        actual = work.read_text()

        if rewrite_goldens:
            golden_path.write_text(actual)
            continue

        assert golden_path.exists(), (
            f"Golden output file does not exist: {golden_path}. "
            "Regenerate and review the expected patch output with "
            "uv run pytest tests/test_miles_patches.py --rewrite."
        )
        expected = golden_path.read_text()
        assert actual == expected, (
            f"golden mismatch for {name}; rerun with --rewrite to accept"
        )


def test_qkvr_cpu_merge_rewrites_factory(tmp_path, capsys):
    # Indent matches the pinned miles image (nested under `if qkey in sd`).
    work = tmp_path / "layers.py"
    work.write_text(
        "class InklingSelfAttention:\n"
        "    def sharded_state_dict(self):\n"
        "        if True:\n"
        "            def _qkvr_merge(sub):\n"
        "                return torch.cat(list(sub), dim=0)\n"
        "\n"
        "            return _qkvr_merge\n"
    )

    assert qkvr_patcher.apply(work) == 1
    patched = work.read_text()
    assert qkvr_patcher.MARKER in patched
    assert "return torch.cat(list(sub), dim=0)" not in patched
    assert "resize_" not in patched
    assert ".to(device)" not in patched
    compile(patched, str(work), "exec")
    assert qkvr_patcher.apply(work) == 0
    assert "already applied" in capsys.readouterr().out


# The abort fan-out's worker listing as Miles has written it, by router version
# (24bd5fb716 and earlier) and /workers alone (c2d56413f5).
_BY_ROUTER_VERSION = """
async def abort(args, rollout_id):
    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]
    return urls
"""
_GET_WORKER_URLS_BY_ROUTER_VERSION = """
async def get_worker_urls(args: Namespace):
    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]
    return router_worker_base_urls(urls)
"""
_WORKERS_ONLY = """
async def abort(args, rollout_id):
    response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
    urls = router_worker_base_urls([worker["url"] for worker in response["workers"]])
    return urls
"""
_GET_WORKER_URLS_WORKERS_ONLY = """
async def get_worker_urls(args: Namespace):
    response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
    return router_worker_base_urls([worker["url"] for worker in response["workers"]])
"""


def _miles_tree(
    root: Path, sglang_rollout: str, inference_rollout_train: str | None
) -> Path:
    (root / abort_patcher.SGLANG_ROLLOUT).parent.mkdir(parents=True)
    (root / abort_patcher.INFERENCE_ROLLOUT_TRAIN).parent.mkdir(parents=True)
    (root / abort_patcher.SGLANG_ROLLOUT).write_text(sglang_rollout)
    if inference_rollout_train is not None:
        (root / abort_patcher.INFERENCE_ROLLOUT_TRAIN).write_text(
            inference_rollout_train
        )
    return root


@pytest.mark.parametrize(
    ("sglang_rollout", "inference_rollout_train"),
    [
        (_BY_ROUTER_VERSION, None),
        (_BY_ROUTER_VERSION, _GET_WORKER_URLS_BY_ROUTER_VERSION),
        (_WORKERS_ONLY, _GET_WORKER_URLS_WORKERS_ONLY),
    ],
    ids=["router-version", "router-version-with-get-worker-urls", "workers-only"],
)
def test_sglang_abort_patch_guards_every_known_worker_listing(
    tmp_path, sglang_rollout, inference_rollout_train
):
    root = _miles_tree(tmp_path, sglang_rollout, inference_rollout_train)
    outcomes = abort_patcher.apply(root)
    assert set(outcomes.values()) == {"applied"}
    for name in outcomes:
        patched = (root / name).read_text()
        assert abort_patcher.MARKER in patched
        assert "except Exception as exc:" in patched
        compile(patched, name, "exec")
    assert set(abort_patcher.apply(root).values()) == {"already applied"}


def test_sglang_abort_patch_fails_on_an_unknown_worker_listing(tmp_path):
    unknown = """
async def abort(args, rollout_id):
    workers = (await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers"))["workers"]
    return workers
"""
    root = _miles_tree(tmp_path, unknown, None)
    with pytest.raises(RuntimeError, match="unknown shape"):
        abort_patcher.apply(root)


def test_sglang_abort_patch_leaves_files_without_a_worker_listing(tmp_path):
    root = _miles_tree(
        tmp_path, _WORKERS_ONLY, "async def get_worker_urls(args):\n    return []\n"
    )
    outcomes = abort_patcher.apply(root)
    assert (
        outcomes[abort_patcher.INFERENCE_ROLLOUT_TRAIN]
        == "not needed (no worker listing)"
    )


def test_sglang_abort_patch_skips_a_tree_without_sglang_rollout(tmp_path, capsys):
    assert abort_patcher.apply(tmp_path) == {}
    assert "skipping SGLang abort patch" in capsys.readouterr().out
