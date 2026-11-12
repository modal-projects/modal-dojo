"""Keep Modal's private SDK surface out of the repo.

Modal Dojo is a consumer of the Modal SDK, so it should break only when
Modal's documented API changes. Private modules (``modal._*``,
``modal_proto``), private classes (``_Client``), raw gRPC stubs, and
``_experimental_*`` methods carry no such promise.

``modal_dojo/common/_modal_compat.py`` is the single place allowed to use
them, for operations Modal has no public API for yet. The other entries in
``ALLOWED`` are tests that need a private name to do their job; each says
why. Delete entries, do not add them.
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCANNED_DIRS = (
    "modal_dojo",
    "scripts",
    "tests",
    "synthetic_monitoring",
    "dashboards",
    "tutorials",
    "docs-next",
)

PRIVATE_PATTERNS = {
    "private modal module": re.compile(r"\bmodal\._[a-z]"),
    "modal_proto": re.compile(r"\bmodal_proto\b"),
    "modal CLI internals": re.compile(r"\bmodal\.cli\b"),
    "private modal class": re.compile(
        r"\b_(Client|App|Function|FunctionCall|Cls|Image|Volume|Secret|Sandbox|Object)\b"
    ),
    "raw gRPC stub": re.compile(r"\.stub\.[A-Z]"),
    "_experimental_ method": re.compile(r"\._experimental_[a-z]"),
    "private handle attribute": re.compile(
        r"_sync_original|_load_context_overrides|\b_object_id\b"
    ),
}

ALLOWED = {
    "modal_dojo/common/_modal_compat.py": "the compat module itself",
    "tests/test_modal_private_api.py": "this guard names the patterns it bans",
    "tests/test_endpoint.py": (
        "contract test: pins the `modal endpoint create` flags that "
        "Endpoint.launch() passes to the CLI"
    ),
    "tests/test_modal_refs.py": (
        "builds fake and deliberately broken handles to test the reducers that "
        "sit on _modal_compat.describe_handle"
    ),
    "scripts/validate_model_configs.py": (
        "registers the validation dataset's module for by-value pickling with "
        "Modal's vendored cloudpickle, which is what serialized=True pickles with"
    ),
    "tests/test_model_validation_registry.py": (
        "pickles with Modal's serializer to mirror serialized=True for the entry above"
    ),
}


def _python_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for name in SCANNED_DIRS:
        base = ROOT / name
        if not base.is_dir():
            continue
        files.extend(
            path
            for path in base.rglob("*.py")
            if not {"node_modules", ".venv"} & set(path.parts)
        )
    return sorted(files)


def _violations(path: pathlib.Path) -> list[str]:
    found: list[str] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(line):
                found.append(f"{lineno}: {label}: {line.strip()}")
    return found


def test_private_modal_api_is_confined_to_the_compat_module() -> None:
    offenders = {
        str(path.relative_to(ROOT)): hits
        for path in _python_files()
        if str(path.relative_to(ROOT)) not in ALLOWED and (hits := _violations(path))
    }
    if offenders:
        report = "\n".join(
            f"{name}\n  " + "\n  ".join(hits) for name, hits in offenders.items()
        )
        pytest.fail(
            "Private Modal API used outside modal_dojo/common/_modal_compat.py. "
            "Use the public API, or add a helper to the compat module:\n" + report
        )


def test_allowlist_has_no_stale_entries() -> None:
    stale = [
        name
        for name in ALLOWED
        if name != "tests/test_modal_private_api.py"
        and not ((ROOT / name).is_file() and _violations(ROOT / name))
    ]
    assert not stale, f"Remove entries that no longer need an exemption: {stale}"
