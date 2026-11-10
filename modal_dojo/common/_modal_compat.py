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

from collections.abc import AsyncIterator
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
