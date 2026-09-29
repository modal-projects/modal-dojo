from __future__ import annotations

import ast
import copyreg
import pickle
from types import FunctionType
from typing import Any


class ModalCaptureError(TypeError):
    """Raised when a cloudpickled user callback captures live Modal state."""


_REDUCERS_REGISTERED = False


def register_modal_cloudpickle_reducers() -> None:
    """Make Modal's reference handles cloudpickle by reference.

    Modal's lazy ``from_name``/``from_id`` handles close over local loader
    functions that the standard pickler cannot serialize. Training Gym still
    needs cloudpickle for inline user callbacks, so register reducers that
    preserve normal Modal syntax while reconstructing those handles from
    Modal's public APIs inside the remote container.
    """

    global _REDUCERS_REGISTERED
    if _REDUCERS_REGISTERED:
        return

    try:
        import modal
    except ImportError:
        return

    copyreg.pickle(modal.Function, _reduce_modal_app_named_handle)
    copyreg.pickle(modal.Cls, _reduce_modal_app_named_handle)
    copyreg.pickle(modal.Sandbox, _reduce_modal_sandbox)
    copyreg.pickle(modal.Dict, _reduce_modal_name_or_id_handle)
    copyreg.pickle(modal.Queue, _reduce_modal_name_or_id_handle)
    copyreg.pickle(modal.Volume, _reduce_modal_volume)
    copyreg.pickle(modal.Secret, _reduce_modal_secret)
    copyreg.pickle(modal.NetworkFileSystem, _reduce_modal_named_handle)
    copyreg.pickle(modal.Proxy, _reduce_modal_named_handle)
    copyreg.pickle(modal.FunctionCall, _reduce_modal_id_handle)
    copyreg.pickle(modal.Image, _reduce_modal_id_handle)
    copyreg.pickle(modal.SandboxSnapshot, _reduce_modal_id_handle)
    _REDUCERS_REGISTERED = True


def _reduce_modal_app_named_handle(value: Any):
    original = _sync_original(value)
    local_reduce = _reduce_from_local_handle(value, original)
    if local_reduce is not None:
        return local_reduce
    closure = _load_closure(original)
    app_name = closure.get("app_name")
    name = closure.get("name")
    if not isinstance(app_name, str) or not isinstance(name, str):
        raise ModalCaptureError(
            f"Only name-based {type(value).__name__} handles can be captured in "
            f"Training Gym callbacks. Use modal.{type(value).__name__}.from_name(...) "
            "or create the handle inside the callback."
        )
    return _restore_modal_app_named_handle, (
        _public_modal_class_name(value),
        app_name,
        name,
        _environment_name(original),
    )


def _reduce_modal_named_handle(value: Any):
    original = _sync_original(value)
    local_reduce = _reduce_from_local_handle(value, original)
    if local_reduce is not None:
        return local_reduce
    closure = _load_closure(original)
    name = getattr(original, "_name", None) or closure.get("name")
    if not isinstance(name, str):
        raise ModalCaptureError(
            f"Only name-based {type(value).__name__} handles can be captured in "
            f"Training Gym callbacks. Use modal.{type(value).__name__}.from_name(...) "
            "or create the handle inside the callback."
        )
    return _restore_modal_named_handle, (
        _public_modal_class_name(value),
        name,
        _environment_name(original),
        _create_if_missing(closure),
    )


def _reduce_modal_name_or_id_handle(value: Any):
    original = _sync_original(value)
    local_reduce = _reduce_from_local_handle(value, original)
    if local_reduce is not None:
        return local_reduce
    closure = _load_closure(original)
    object_id = _object_id(original) or _single_closure_id(closure)
    if isinstance(object_id, str):
        return _restore_modal_id_handle, (_public_modal_class_name(value), object_id)
    return _reduce_modal_named_handle(value)


def _reduce_modal_volume(volume: Any):
    original = _sync_original(volume)
    local_reduce = _reduce_from_local_handle(volume, original)
    if local_reduce is not None:
        return local_reduce
    closure = _load_closure(original)
    object_id = _object_id(original) or _single_closure_id(closure)
    if isinstance(object_id, str):
        return _restore_modal_id_handle, (_public_modal_class_name(volume), object_id)
    name = getattr(original, "_name", None) or closure.get("name")
    if not isinstance(name, str):
        raise ModalCaptureError(
            "Only name-based or id-based Volume handles can be captured in "
            "Training Gym callbacks. Use modal.Volume.from_name(...), "
            "modal.Volume.from_id(...), or create the Volume inside the callback."
        )
    create_if_missing = bool(closure.get("create_if_missing", False))
    version = closure.get("version")
    return _restore_modal_volume, (
        name,
        _environment_name(original),
        create_if_missing,
        version,
    )


def _reduce_modal_secret(secret: Any):
    original = _sync_original(secret)
    local_reduce = _reduce_from_local_handle(secret, original)
    if local_reduce is not None:
        return local_reduce
    closure = _load_closure(original)
    name = getattr(original, "_name", None) or closure.get("name")
    if not isinstance(name, str):
        raise ModalCaptureError(
            "Only name-based Secret handles can be captured in Training Gym "
            "callbacks. Use modal.Secret.from_name(...) or create the Secret "
            "inside the callback. Secret.from_dict(...) and "
            "Secret.from_local_environ(...) are intentionally not serialized "
            "because that would embed local secret values in the pickle payload."
        )
    required_keys = closure.get("required_keys", [])
    if not isinstance(required_keys, list):
        required_keys = []
    return _restore_modal_secret, (name, _environment_name(original), required_keys)


def _reduce_modal_sandbox(sandbox: Any):
    original = _sync_original(sandbox)
    closure = _load_closure(original)
    sandbox_id = _object_id(original) or closure.get("sandbox_id")
    if not isinstance(sandbox_id, str):
        raise ModalCaptureError(
            "Only id-based Sandbox handles can be captured in Training Gym "
            "callbacks. Use modal.Sandbox.from_id(...), capture a running "
            "Sandbox returned by Modal, or create the Sandbox inside the callback."
        )
    return _restore_modal_id_handle, ("Sandbox", sandbox_id)


def _reduce_modal_id_handle(value: Any):
    original = _sync_original(value)
    local_reduce = _reduce_from_local_handle(value, original)
    if local_reduce is not None:
        return local_reduce
    closure = _load_closure(original)
    object_id = _object_id(original) or _single_closure_id(closure)
    if not isinstance(object_id, str):
        raise ModalCaptureError(
            f"Only id-based {type(value).__name__} handles can be captured in "
            f"Training Gym callbacks. Use modal.{type(value).__name__}.from_id(...) "
            "or create the handle inside the callback."
        )
    return _restore_modal_id_handle, (_public_modal_class_name(value), object_id)


def _restore_modal_app_named_handle(
    class_name: str,
    app_name: str,
    name: str,
    environment_name: str | None,
):
    import modal

    kwargs: dict[str, Any] = {}
    if environment_name is not None:
        kwargs["environment_name"] = environment_name
    return getattr(modal, class_name).from_name(app_name, name, **kwargs)


def _restore_modal_named_handle(
    class_name: str,
    name: str,
    environment_name: str | None,
    create_if_missing: bool,
):
    import modal

    kwargs: dict[str, Any] = {}
    if environment_name is not None:
        kwargs["environment_name"] = environment_name
    if class_name in {"Dict", "Queue", "NetworkFileSystem"}:
        kwargs["create_if_missing"] = create_if_missing
    return getattr(modal, class_name).from_name(name, **kwargs)


def _restore_modal_id_handle(class_name: str, object_id: str):
    import modal

    return getattr(modal, class_name).from_id(object_id)


def _restore_modal_volume(
    name: str,
    environment_name: str | None,
    create_if_missing: bool,
    version: Any,
):
    import modal

    kwargs: dict[str, Any] = {"create_if_missing": create_if_missing}
    if environment_name is not None:
        kwargs["environment_name"] = environment_name
    if version is not None:
        kwargs["version"] = version
    return modal.Volume.from_name(name, **kwargs)


def _restore_modal_secret(
    name: str,
    environment_name: str | None,
    required_keys: list[str],
):
    import modal

    kwargs: dict[str, Any] = {"required_keys": required_keys}
    if environment_name is not None:
        kwargs["environment_name"] = environment_name
    return modal.Secret.from_name(name, **kwargs)


def _sync_original(value: Any) -> Any:
    for attr_name, attr_value in getattr(value, "__dict__", {}).items():
        if attr_name.startswith("_sync_original"):
            return attr_value
    return value


def _public_modal_class_name(value: Any) -> str:
    return type(value).__name__.lstrip("_")


def _object_id(original: Any) -> str | None:
    object_id = getattr(original, "_object_id", None)
    return object_id if isinstance(object_id, str) else None


def _reduce_from_local_handle(value: Any, original: Any):
    load = getattr(original, "_load", None)
    qualname = getattr(load, "__qualname__", "")
    if not isinstance(qualname, str) or "from_local" not in qualname:
        return None
    return value.__reduce_ex__(pickle.HIGHEST_PROTOCOL)


def _single_closure_id(closure: dict[str, Any]) -> str | None:
    if len(closure) != 1:
        return None
    value = next(iter(closure.values()))
    return value if isinstance(value, str) else None


def _create_if_missing(closure: dict[str, Any]) -> bool:
    return bool(closure.get("create_if_missing", False))


def _environment_name(original: Any) -> str | None:
    load_context = getattr(original, "_load_context_overrides", None)
    environment_name = getattr(load_context, "_environment_name", None)
    if isinstance(environment_name, str):
        return environment_name

    kwargs = _repr_kwargs(original)
    environment_name = kwargs.get("environment_name")
    return environment_name if isinstance(environment_name, str) else None


def _load_closure(original: Any) -> dict[str, Any]:
    load = getattr(original, "_load", None)
    if not isinstance(load, FunctionType):
        return {}
    closure = load.__closure__ or ()
    return {
        name: cell.cell_contents
        for name, cell in zip(load.__code__.co_freevars, closure, strict=False)
    }


def _repr_kwargs(original: Any) -> dict[str, Any]:
    rep = getattr(original, "_rep", "")
    if not isinstance(rep, str):
        return {}
    try:
        expr = ast.parse(rep, mode="eval").body
    except SyntaxError:
        return {}
    if not isinstance(expr, ast.Call):
        return {}
    kwargs: dict[str, Any] = {}
    for keyword in expr.keywords:
        if keyword.arg is None:
            continue
        try:
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        except ValueError:
            continue
    return kwargs
