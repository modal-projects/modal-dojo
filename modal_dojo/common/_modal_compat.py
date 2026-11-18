"""The one module allowed to reach into Modal's private SDK surface.

Everything else in the repo uses Modal's public API (``modal.*`` and
``modal.experimental.*``). The helpers here cover operations Modal does not
expose publicly yet; each one names the public API that would replace it, so
this file shrinks as the SDK grows. ``tests/test_modal_private_api.py`` fails
when a private Modal import or attribute appears anywhere else.

Keep the helpers thin: build the private request, make the call, and hand
back plain values or the response as-is. Callers must not import
``modal_proto`` or ``modal._*`` themselves.
"""

from __future__ import annotations

import ast
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import FunctionType
from typing import Any


def stop_app_by_id(app_id: str) -> None:
    """Stop the Modal App with this id, ephemeral or deployed.

    Private because ``modal.experimental.stop_app`` only resolves a *deployed*
    App by name, and training runs live in detached ephemeral Apps known by
    id. Replace with a public id-based stop once Modal has one.
    """
    from modal._utils.async_utils import synchronizer
    from modal.client import _Client
    from modal_proto import api_pb2

    async def _stop() -> None:
        client = await _Client.from_env()
        await client.stub.AppStop(
            api_pb2.AppStopRequest(
                app_id=app_id,
                source=api_pb2.APP_STOP_SOURCE_PYTHON_CLIENT,
            )
        )

    synchronizer.create_blocking(_stop)()


async def open_async_client(token_id: str, token_secret: str) -> Any:
    """Open a Modal client that lives on the caller's event loop.

    Private because the public ``modal.Client`` runs its RPCs on Modal's own
    synchronizer loop, and the log helpers below need the raw stub on the
    loop that serves the request. Goes away with them.
    """
    from modal.client import _Client

    return await _Client.from_credentials(token_id, token_secret)


def to_proto_timestamp(secs: float) -> Any:
    """Convert epoch seconds (with fractional part) to a protobuf ``Timestamp``."""
    from google.protobuf.timestamp_pb2 import Timestamp

    ts = Timestamp()
    ts.seconds = int(secs)
    ts.nanos = int((secs - int(secs)) * 1e9)
    return ts


async def fetch_app_logs(
    client: Any,
    *,
    app_id: str,
    since: float,
    until: float,
    limit: int,
    search_text: str,
) -> Any:
    """Return the ``TaskLogsBatch`` sequence for one historical log window.

    Private because the public ``App.logs.fetch()`` needs an ``App`` handle,
    and there is no public way to build one from an app id.
    """
    from modal_proto import api_pb2

    resp = await client.stub.AppFetchLogs(
        api_pb2.AppFetchLogsRequest(
            app_id=app_id,
            since=to_proto_timestamp(since),
            until=to_proto_timestamp(until),
            limit=limit,
            search_text=search_text,
        )
    )
    return resp.batches


def stream_app_logs(
    client: Any,
    *,
    app_id: str,
    last_entry_id: str,
    timeout: float,
) -> AsyncIterator[Any]:
    """Open one long-poll log stream and yield its ``TaskLogsBatch`` messages.

    Private for the same reason as :func:`fetch_app_logs`; the public
    equivalent is ``App.logs.stream()``.
    """
    from modal_proto import api_pb2

    return client.stub.AppGetLogs.unary_stream(
        api_pb2.AppGetLogsRequest(
            app_id=app_id,
            timeout=timeout,
            last_entry_id=last_entry_id,
        )
    )


@dataclass(frozen=True)
class HandleDescription:
    """How a Modal object handle was constructed, as far as Modal records it."""

    # Set for hydrated handles and for handles made with ``from_id``.
    object_id: str | None
    # The handle's own name, when it has one (``Volume``, ``Secret``, ...).
    name: str | None
    environment_name: str | None
    # Free variables of the loader behind a lazy handle, e.g. ``name``,
    # ``app_name``, ``create_if_missing``, ``version``, ``required_keys``.
    loader_vars: dict[str, Any]
    # Built from local state (``Secret.from_local_environ``, ...), so there
    # is no remote object to refer to.
    is_local: bool


def describe_handle(value: Any) -> HandleDescription:
    """Read back the arguments a Modal handle was constructed with.

    Private because lazy ``from_name``/``from_id`` handles expose no public
    record of their arguments: ``repr()`` omits ``create_if_missing``,
    ``version``, and ``required_keys``, and ``Function``/``Cls`` handles have
    no public name at all. ``modal_refs`` needs them to pickle a handle by
    reference. Goes away once Modal handles are picklable themselves.
    """
    original = value
    for attr_name, attr_value in getattr(value, "__dict__", {}).items():
        if attr_name.startswith("_sync_original"):
            original = attr_value
            break

    load = getattr(original, "_load", None)
    loader_vars: dict[str, Any] = {}
    if isinstance(load, FunctionType):
        loader_vars = {
            name: cell.cell_contents
            for name, cell in zip(
                load.__code__.co_freevars, load.__closure__ or (), strict=False
            )
        }
    qualname = getattr(load, "__qualname__", "")

    object_id = getattr(original, "_object_id", None)
    name = getattr(original, "_name", None)
    return HandleDescription(
        object_id=object_id if isinstance(object_id, str) else None,
        name=name if isinstance(name, str) else None,
        environment_name=_handle_environment_name(original),
        loader_vars=loader_vars,
        is_local=isinstance(qualname, str) and "from_local" in qualname,
    )


def _handle_environment_name(original: Any) -> str | None:
    load_context = getattr(original, "_load_context_overrides", None)
    environment_name = getattr(load_context, "_environment_name", None)
    if isinstance(environment_name, str):
        return environment_name

    # Some handles only record the environment in their repr.
    rep = getattr(original, "_rep", "")
    if not isinstance(rep, str):
        return None
    try:
        expr = ast.parse(rep, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expr, ast.Call):
        return None
    for keyword in expr.keywords:
        if keyword.arg == "environment_name":
            try:
                value = ast.literal_eval(keyword.value)
            except ValueError:
                return None
            return value if isinstance(value, str) else None
    return None
