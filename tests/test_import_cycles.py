"""Guard against import cycles anywhere in ``modal_dojo``.

Every module is imported *first*, alone, in a fresh interpreter. That ordering is
the whole point: a cycle only raises for whoever enters it first, and nothing
else in CI ever does. ``modal_dojo/__init__.py`` is a lazy
``__getattr__`` shim, so ``import modal_dojo`` pulls in no submodule at
all; ``compileall`` never executes an import; and by the time a test touches
``common.config`` the module is already in ``sys.modules``, which masks the cycle
no matter which side is broken.

That is how the ``common.config`` -> ``cli.setup`` -> ``_dashboard`` ->
``common.config`` cycle shipped green: importing either ``common.config`` or
``_dashboard`` first raised ``ImportError: cannot import name
'get_dashboard_url' from partially initialized module``, but no CI check imported
either one first.

Cycles in this package are avoided by deferring the import into the function body
that needs it (see ``common.config.get_dashboard_proxy_auth``), so a failure here
usually means "move this module-level import inside the function".
"""

from __future__ import annotations

import concurrent.futures
import pathlib
import subprocess
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "modal_dojo"

# Modules that import slime / Megatron-LM / miles / torch at module scope, or
# read a file that only exists in the training image. They cannot be imported
# outside that image, so their import is only checked for *cycles* and not for
# success. Entries must still name real modules, or the staleness test below
# fails on a rename or deletion.
REMOTE_ONLY = frozenset(
    {
        "modal_dojo.common.megatron_patches.patch_checkpoint_save",
        "modal_dojo.common.megatron_patches.patch_dist_ckpt_nofork",
        "modal_dojo.common.megatron_patches.patch_dist_ckpt_quantized",
        "modal_dojo.common.megatron_patches.patch_gdn_packed_seq",
        "modal_dojo.common.megatron_patches.patch_torch_load",
        "modal_dojo.common.megatron_patches.patch_validation",
        "modal_dojo.frameworks.miles.modal_helpers.hf_block_dequant",
        "modal_dojo.frameworks.miles.modal_helpers.patches.patch_deepseek_v41_chat_template",
        "modal_dojo.frameworks.miles.modal_helpers.patches.patch_deepseek_v41_fp4_dequant_block",
        "modal_dojo.frameworks.miles.modal_helpers.patches.patch_deepseek_v41_fp8_hopper_gemm",
        "modal_dojo.frameworks.miles.modal_helpers.patches.patch_deepseek_v41_processor_tokenizer",
        "modal_dojo.frameworks.miles.modal_helpers.patches.patch_deepseek_v41_sglang_tree",
        "modal_dojo.frameworks.miles.modal_helpers.patches.patch_deepseek_v41_vision_topk_capture",
        "modal_dojo.frameworks.miles.modal_helpers.patches.patch_gemma4_vl_rollout_text",
        "modal_dojo.frameworks.miles.modal_helpers.patches.patch_router_startup_timeout",
        "modal_dojo.frameworks.miles.modal_helpers.patches.patch_sglang_abort",
        "modal_dojo.frameworks.slime.modal_helpers.patches.patch_advantages",
        "modal_dojo.frameworks.slime.modal_helpers.patches.patch_bridge_provider_per_token_loss",
        "modal_dojo.frameworks.slime.modal_helpers.patches.patch_megatron_bridge",
        "modal_dojo.frameworks.slime.modal_helpers.patches.patch_stop_token_diagnostic",
        "modal_dojo.frameworks.slime.modal_helpers.patches.patch_zero_std_metrics",
        "modal_dojo.frameworks.slime.opd_reward",
    }
)

# CPython's wording for a cycle, from ``_handle_fromlist`` / ``_find_and_load``.
_CYCLE_MARKERS = (
    "partially initialized module",
    "most likely due to a circular import",
)


def discover_modules() -> list[str]:
    """Every module in the package as a dotted name, packages included.

    A filesystem walk rather than ``pkgutil.walk_packages``: walk_packages
    imports each package to read its ``__path__``, which both defeats the point
    (the parent is in ``sys.modules`` before the module under test loads) and
    dies on the remote-only subpackages.
    """
    modules = []
    for path in sorted(PKG_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()  # the package itself, not a submodule
        if any(not part.isidentifier() for part in parts):
            continue
        modules.append(".".join(parts))
    return modules


MODULES = discover_modules()


def _import_alone(module: str) -> str | None:
    """Import ``module`` first in a fresh interpreter; return stderr on failure."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return None if result.returncode == 0 else result.stderr


@pytest.fixture(scope="session")
def import_results() -> dict[str, str | None]:
    """Import every module in its own interpreter, in parallel (~3s for all of them)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        return dict(zip(MODULES, pool.map(_import_alone, MODULES)))


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_alone(
    module: str, import_results: dict[str, str | None]
) -> None:
    stderr = import_results[module]
    if stderr is None:
        return

    if any(marker in stderr for marker in _CYCLE_MARKERS):
        pytest.fail(
            f"circular import: {module} cannot be imported first.\n"
            f"Move the offending module-level import into the function that uses it.\n\n"
            f"{stderr}"
        )

    if module in REMOTE_ONLY:
        # Expected: needs slime / Megatron-LM / miles / torch from the training image.
        # Checked for cycles above, which is all this test can assert here.
        return

    pytest.fail(f"importing {module} first fails:\n{stderr}")


def test_no_stale_remote_only_entries() -> None:
    """``REMOTE_ONLY`` must not accumulate entries for modules that no longer exist."""
    assert not (REMOTE_ONLY - set(MODULES)), (
        "REMOTE_ONLY names modules that no longer exist; delete these entries"
    )


def test_discovery_covers_the_package() -> None:
    """A walk that silently stops short would make every check above vacuous."""
    assert "modal_dojo" in MODULES
    assert "modal_dojo.common.config" in MODULES  # nested module
    assert "modal_dojo.frameworks.slime.launcher" in MODULES  # deeply nested

    subpackages = {
        f"modal_dojo.{path.name}"
        for path in PKG_ROOT.iterdir()
        if (path / "__init__.py").exists()
    }
    assert subpackages <= set(MODULES)
