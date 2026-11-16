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
    synchronizer loop, and :func:`app_logs` needs a client on the loop that
    serves the request.
    """
    from modal.client import _Client

    return await _Client.from_credentials(token_id, token_secret)


def app_logs(client: Any, app_id: str) -> Any:
    """Return Modal's public ``App.logs`` manager for an app id.

    Private because there is no public way to get an ``App`` from an id, and
    training runs live in detached ephemeral Apps that ``App.lookup`` can't
    find by name. ``_new_remote`` builds the handle without an RPC; callers
    then use the public ``fetch()`` / ``stream()``. Replace with a public
    id-based ``App`` constructor once Modal has one.
    """
    from modal.app import _App

    app = _App._new_remote(None, app_id, environment_name="", client=client)
    return app.logs
