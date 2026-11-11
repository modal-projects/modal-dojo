from __future__ import annotations

import copyreg
import pickle
from typing import Any

from modal_dojo.common._modal_compat import HandleDescription, describe_handle


class ModalCaptureError(TypeError):
    """Raised when a cloudpickled user callback captures live Modal state."""


_REDUCERS_REGISTERED = False


def register_modal_cloudpickle_reducers() -> None:
    """Make Modal's reference handles cloudpickle by reference.

    Modal's lazy ``from_name``/``from_id`` handles close over local loader
    functions that the standard pickler cannot serialize. Modal Dojo still
    needs cloudpickle for inline user callbacks, so register reducers that
    preserve normal Modal syntax while reconstructing those handles from
    Modal's public APIs inside the remote container.

    Reading back how a handle was constructed is the only part that needs
    Modal's private attributes; it lives in ``_modal_compat.describe_handle``.
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
    handle = describe_handle(value)
    if handle.is_local:
        return _default_reduce(value)
    app_name = handle.loader_vars.get("app_name")
    name = handle.loader_vars.get("name")
    if not isinstance(app_name, str) or not isinstance(name, str):
        raise ModalCaptureError(
            f"Only name-based {type(value).__name__} handles can be captured in "
            f"Modal Dojo callbacks. Use modal.{type(value).__name__}.from_name(...) "
            "or create the handle inside the callback."
        )
    return _restore_modal_app_named_handle, (
        _public_modal_class_name(value),
        app_name,
        name,
        handle.environment_name,
    )


def _reduce_modal_named_handle(value: Any):
    handle = describe_handle(value)
    if handle.is_local:
        return _default_reduce(value)
    name = handle.name or handle.loader_vars.get("name")
    if not isinstance(name, str):
        raise ModalCaptureError(
            f"Only name-based {type(value).__name__} handles can be captured in "
            f"Modal Dojo callbacks. Use modal.{type(value).__name__}.from_name(...) "
            "or create the handle inside the callback."
        )
    return _restore_modal_named_handle, (
        _public_modal_class_name(value),
        name,
        handle.environment_name,
        bool(handle.loader_vars.get("create_if_missing", False)),
    )


def _reduce_modal_name_or_id_handle(value: Any):
    handle = describe_handle(value)
    if handle.is_local:
        return _default_reduce(value)
    object_id = _handle_object_id(handle)
    if object_id is not None:
        return _restore_modal_id_handle, (_public_modal_class_name(value), object_id)
    return _reduce_modal_named_handle(value)


def _reduce_modal_volume(volume: Any):
    handle = describe_handle(volume)
    if handle.is_local:
        return _default_reduce(volume)
    object_id = _handle_object_id(handle)
    if object_id is not None:
        return _restore_modal_id_handle, (_public_modal_class_name(volume), object_id)
    name = handle.name or handle.loader_vars.get("name")
    if not isinstance(name, str):
        raise ModalCaptureError(
            "Only name-based or id-based Volume handles can be captured in "
            "Modal Dojo callbacks. Use modal.Volume.from_name(...), "
            "modal.Volume.from_id(...), or create the Volume inside the callback."
        )
    return _restore_modal_volume, (
        name,
        handle.environment_name,
        bool(handle.loader_vars.get("create_if_missing", False)),
        handle.loader_vars.get("version"),
    )


def _reduce_modal_secret(secret: Any):
    handle = describe_handle(secret)
    if handle.is_local:
        return _default_reduce(secret)
    name = handle.name or handle.loader_vars.get("name")
    if not isinstance(name, str):
        raise ModalCaptureError(
            "Only name-based Secret handles can be captured in Modal Dojo "
            "callbacks. Use modal.Secret.from_name(...) or create the Secret "
            "inside the callback. Secret.from_dict(...) and "
            "Secret.from_local_environ(...) are intentionally not serialized "
            "because that would embed local secret values in the pickle payload."
        )
    required_keys = handle.loader_vars.get("required_keys", [])
    if not isinstance(required_keys, list):
        required_keys = []
    return _restore_modal_secret, (name, handle.environment_name, required_keys)


def _reduce_modal_sandbox(sandbox: Any):
    handle = describe_handle(sandbox)
    sandbox_id = handle.object_id or handle.loader_vars.get("sandbox_id")
    if not isinstance(sandbox_id, str):
        raise ModalCaptureError(
            "Only id-based Sandbox handles can be captured in Modal Dojo "
            "callbacks. Use modal.Sandbox.from_id(...), capture a running "
            "Sandbox returned by Modal, or create the Sandbox inside the callback."
        )
    return _restore_modal_id_handle, ("Sandbox", sandbox_id)


def _reduce_modal_id_handle(value: Any):
    handle = describe_handle(value)
    if handle.is_local:
        return _default_reduce(value)
    object_id = _handle_object_id(handle)
    if object_id is None:
        raise ModalCaptureError(
            f"Only id-based {type(value).__name__} handles can be captured in "
            f"Modal Dojo callbacks. Use modal.{type(value).__name__}.from_id(...) "
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


def _public_modal_class_name(value: Any) -> str:
    return type(value).__name__.lstrip("_")


def _default_reduce(value: Any):
    """Pickle a handle built from local state the way pickle would without us."""
    return value.__reduce_ex__(pickle.HIGHEST_PROTOCOL)


def _handle_object_id(handle: HandleDescription) -> str | None:
    """The handle's object id: its own if hydrated, else the single id a
    ``from_id`` loader was given."""
    if handle.object_id is not None:
        return handle.object_id
    if len(handle.loader_vars) != 1:
        return None
    value = next(iter(handle.loader_vars.values()))
    return value if isinstance(value, str) else None
