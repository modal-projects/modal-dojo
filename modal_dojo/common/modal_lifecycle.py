"""Modal app lifecycle helpers."""

from __future__ import annotations

from collections.abc import Callable

_LIVE_APP_STATES: frozenset[int] | None = None


def live_app_states() -> frozenset[int]:
    global _LIVE_APP_STATES
    if _LIVE_APP_STATES is None:
        from modal_proto import api_pb2

        _LIVE_APP_STATES = frozenset(
            {
                api_pb2.APP_STATE_EPHEMERAL,
                api_pb2.APP_STATE_DETACHED,
                api_pb2.APP_STATE_DETACHED_DISCONNECTED,
                api_pb2.APP_STATE_INITIALIZING,
                api_pb2.APP_STATE_DEPLOYED,
                api_pb2.APP_STATE_DERIVED,
            }
        )
    return _LIVE_APP_STATES


def get_app_lifecycle_state(app_id: str) -> int | None:
    """Return Modal ``app_state`` enum value, or ``None`` on error."""
    if not app_id:
        return None
    try:
        from modal._utils.async_utils import synchronizer
        from modal.client import _Client
        from modal_proto import api_pb2

        async def _lifecycle_state() -> int:
            client = await _Client.from_env()
            resp = await client.stub.AppGetLifecycle(
                api_pb2.AppGetLifecycleRequest(app_id=app_id)
            )
            return resp.lifecycle.app_state

        return synchronizer.create_blocking(_lifecycle_state)()
    except Exception:
        return None


def app_live_from_state(state: int | None) -> bool | None:
    """Map a lifecycle state enum to live/unknown."""
    if state is None:
        return None
    return state in live_app_states()


def app_live_status(app_id: str) -> bool | None:
    """Return whether the Modal app is live, or ``None`` if state is unknown."""
    return app_live_from_state(get_app_lifecycle_state(app_id))


def resolve_app_liveness(
    app_id: str,
    *,
    get_lifecycle_state: Callable[[str], int | None] | None = None,
) -> tuple[int | None, bool | None]:
    """Fetch lifecycle once and derive live status from that state."""
    if not app_id:
        return None, None
    get_state = get_lifecycle_state or get_app_lifecycle_state
    modal_app_state = get_state(app_id)
    return modal_app_state, app_live_from_state(modal_app_state)


def app_is_live(app_id: str) -> bool:
    """Best-effort check whether a Modal app is still alive on the server."""
    status = app_live_status(app_id)
    if status is None:
        return False
    return status


def stop_app(app_id: str) -> None:
    """Stop a detached Modal app (best effort). Never raises."""
    if not app_id:
        return
    try:
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
    except Exception as exc:  # noqa: BLE001 — auto-stop is best-effort
        print(f"WARNING: could not auto-stop app {app_id}: {exc!r}")
