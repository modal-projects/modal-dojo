"""What a training run's remote functions receive, and how it gets there.

A run's Modal functions (``download``, ``prepare_dataset``,
``resolve_checkpoint``, ``convert_checkpoint``, ``train``) are module-level
functions, registered without ``serialized=True``. Everything they need about
one particular run is a :class:`RunSpec`, passed as their first argument.

``RunSpec.pack()`` is the one place Modal Dojo pickles on the user's behalf.
The spec holds the user's recipe, model, and datasets, whose classes may be
defined inline in the launching script. Those classes do not exist in the
image, so their definitions have to travel by value, which only cloudpickle
can do. Everything else in the spec is plain data.

By-value pickling embeds Python bytecode, which is only valid for the Python
minor version that produced it. That is why the launching Python must match
the image's when a run ships inline classes, and ``RunSpec.unpack()`` says so
when the two differ instead of surfacing a bare unpickling error.
"""

from __future__ import annotations

import itertools
import pickle
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from modal_dojo.common.errors import DojoError
from modal_dojo.common.framework import is_user_module

# How deep to look inside the spec's objects for instances of user classes.
# A dataset wrapped by the launcher sits one level down; a user object holding
# another user object sits two.
_USER_MODULE_SEARCH_DEPTH = 4
# Collections are sampled, not scanned: a dataset may hold millions of rows,
# and the classes worth finding show up in the first few elements.
_USER_MODULE_SEARCH_WIDTH = 32


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _user_modules(root: Any) -> set[ModuleType]:
    """Top-level modules defining classes of objects reachable from ``root``,
    limited to modules that are absent from the image."""
    found: set[ModuleType] = set()
    seen: set[int] = set()

    def visit(obj: Any, depth: int) -> None:
        if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
            return
        if id(obj) in seen or depth > _USER_MODULE_SEARCH_DEPTH:
            return
        seen.add(id(obj))

        for cls in type(obj).__mro__:
            module = sys.modules.get(cls.__module__)
            if module is not None and is_user_module(module):
                # Registering the top-level package covers its submodules, so
                # a class and the sibling helpers it uses travel together.
                top = sys.modules.get(module.__name__.split(".")[0], module)
                found.add(top)

        if isinstance(obj, dict):
            children: Any = obj.values()
        elif isinstance(obj, (list, tuple, set, frozenset)):
            children = obj
        else:
            children = getattr(obj, "__dict__", {}).values()
        for child in itertools.islice(children, _USER_MODULE_SEARCH_WIDTH):
            visit(child, depth + 1)

    visit(root, 0)
    return found


@dataclass
class RunSpec:
    """Everything one training run's remote functions need to know."""

    training_run_id: str
    app_name: str
    recipe: Any
    model: Any
    dataset: Any
    eval_dataset: Any
    dataset_path: str
    eval_dataset_path: str | None
    checkpoint_dir: str
    hf_cache_volume_name: str
    data_volume_name: str
    checkpoints_volume_name: str
    checkpoints_mount_path: str
    group_id: str | None = None

    # Volumes are named rather than carried as handles so the spec stays plain
    # data. Inside a container these resolve to the volumes mounted on the
    # function, which is what ``reload()`` and ``commit()`` act on.

    def hf_cache_volume(self) -> Any:
        import modal

        return modal.Volume.from_name(self.hf_cache_volume_name)

    def data_volume(self) -> Any:
        import modal

        return modal.Volume.from_name(self.data_volume_name)

    def checkpoints_volume(self) -> Any:
        import modal

        return modal.Volume.from_name(self.checkpoints_volume_name)

    def pack(self) -> dict[str, Any]:
        """Serialize the spec into a payload Modal can pass as an argument."""
        import cloudpickle

        from modal_dojo.common.modal_refs import register_modal_cloudpickle_reducers

        # Classes from modules the image lacks must be pickled by value.
        # ``__main__`` always is; other user modules have to be registered.
        for module in _user_modules(self):
            if module.__name__ != "__main__":
                cloudpickle.register_pickle_by_value(module)
        register_modal_cloudpickle_reducers()

        return {"python": _python_version(), "spec": cloudpickle.dumps(self)}

    @staticmethod
    def unpack(payload: dict[str, Any]) -> "RunSpec":
        """Rebuild the spec inside a container."""
        try:
            return pickle.loads(payload["spec"])
        except Exception as exc:
            launched_from = payload.get("python")
            running = _python_version()
            if launched_from != running:
                raise DojoError(
                    f"This run was launched from Python {launched_from}, but its "
                    f"image runs Python {running}. A run that defines classes or "
                    "functions inline in the launching script ships them by "
                    "value, which only works when both sides use the same Python "
                    "minor version. Launch from a matching Python, or move the "
                    "definitions into a module the image can import."
                ) from exc
            raise
