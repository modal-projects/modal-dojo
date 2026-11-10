from __future__ import annotations

import inspect
from pathlib import Path
import sysconfig
from types import ModuleType
from typing import TYPE_CHECKING

from enum import Enum

if TYPE_CHECKING:
    from modal import Image


# Canonical mount point for `modal_dojo/tools/` inside every framework
# image. Framework-agnostic helpers shell out to scripts at this path, so it
# must exist on every launcher's image.
TOOLS_REMOTE_PATH = "/opt/modal-dojo/tools"

# Local path to the tools directory. Shipped with the package so
# `mount_tools_dir` works from an editable install or a `pip install`ed copy.
TOOLS_LOCAL_PATH = Path(__file__).resolve().parent.parent / "tools"


def mount_tools_dir(image: "Image") -> "Image":
    """Attach `modal_dojo/tools/` to a Modal image at `TOOLS_REMOTE_PATH`.

    Every framework launcher should call this on its training image(s) so
    cross-framework scripts resolve at a predictable remote path regardless of
    the framework image's layout.
    """
    return image.add_local_dir(
        TOOLS_LOCAL_PATH,
        remote_path=TOOLS_REMOTE_PATH,
        copy=True,
    )


def _installed_module_dirs() -> list[Path]:
    return [
        Path(path).resolve()
        for key in ("stdlib", "platstdlib")
        if (path := sysconfig.get_path(key))
    ]


def _is_installed_module_file(module_file: str, stdlib_dirs: list[Path]) -> bool:
    """Whether a module file belongs to the standard library or an installed
    distribution, as opposed to the user's own source tree."""
    module_path = Path(module_file).resolve()
    if {"site-packages", "dist-packages"} & set(module_path.parts):
        return True
    return any(module_path.is_relative_to(stdlib_dir) for stdlib_dir in stdlib_dirs)


def is_user_module(module: ModuleType, skip_prefix: str = "modal_dojo") -> bool:
    """Whether ``module`` is the user's own code, absent from a training image.

    Images carry Modal Dojo and whatever is pip-installed in them, so those
    modules are importable remotely. A module from the user's source tree is
    not, and neither is ``__main__``.
    """
    name = getattr(module, "__name__", "")
    if not name or name.split(".")[0] == skip_prefix:
        return False
    if name == "__main__":
        return True
    module_file = getattr(module, "__file__", None)
    if not module_file:  # builtin or namespace package
        return False
    return not _is_installed_module_file(module_file, _installed_module_dirs())


def resolve_caller_module(
    skip_prefix: str = "modal_dojo",
) -> ModuleType | None:
    """Return the first stack frame's module whose name is outside this package.

    Framework configs (e.g. `TrainConfig.build_app`) delegate to launcher
    factories (e.g. `build_slime_app`) on behalf of user tutorials, so the
    immediate caller of the factory is a framework module — not the tutorial.
    `cloudpickle.register_pickle_by_value(module)` needs the tutorial module
    to inline the user's inline classes; walking past framework frames
    recovers it.
    """
    stdlib_dirs = _installed_module_dirs()
    for frame_info in inspect.stack()[1:]:
        module = inspect.getmodule(frame_info.frame)
        if module is None:
            continue
        name = getattr(module, "__name__", "")
        if not name or name.startswith(skip_prefix):
            continue
        if name.startswith(
            (
                "IPython",
                "ipykernel",
                "asyncio",
                "traitlets",
                "contextlib",
                "runpy",
                "importlib",
            )
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file and _is_installed_module_file(module_file, stdlib_dirs):
            continue
        return module
    return None


class Framework(str, Enum):
    # ``str`` mixin so a Framework serializes straight to its value in plain
    # ``json.dumps`` (dashboard blobs do not coerce enums), matching the
    # SlimeStatus/MilesStatus convention.
    SLIME = "slime"
    MILES = "miles"
