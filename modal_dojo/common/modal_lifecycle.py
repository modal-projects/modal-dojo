"""Modal app lifecycle helpers."""

from __future__ import annotations


def app_live_status(app_id: str) -> bool | None:
    """Return whether the Modal app is live, or ``None`` if that is unknown.

    An app is live until Modal records a stop time for it. Lookup failures
    (network errors, an app id Modal no longer knows) are reported as unknown
    rather than dead, so callers never terminalize a run on a failed query.
    """
    if not app_id:
        return None
    try:
        from modal.experimental import get_app_lifecycle

        return get_app_lifecycle(app_id).stopped_at is None
    except Exception:
        return None


def app_is_live(app_id: str) -> bool:
    """Best-effort check whether a Modal app is still alive on the server."""
    return app_live_status(app_id) is True


def stop_app(app_id: str) -> None:
    """Stop a detached Modal app (best effort). Never raises."""
    if not app_id:
        return
    try:
        from modal_dojo.common._modal_compat import stop_app_by_id

        stop_app_by_id(app_id)
    except Exception as exc:  # noqa: BLE001 — auto-stop is best-effort
        print(f"WARNING: could not auto-stop app {app_id}: {exc!r}")
