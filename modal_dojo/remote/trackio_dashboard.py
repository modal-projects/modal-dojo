"""The Trackio dashboard web app deployed by ``modal_dojo.common.trackio``."""

from __future__ import annotations

import os
import secrets


def dashboard():
    """Build the Trackio ASGI app, gated by the optional dashboard password."""
    import base64
    import binascii

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response
    from trackio.server import build_starlette_app_only

    starlette_app, _ = build_starlette_app_only()
    dashboard_password = os.environ.get("DASHBOARD_PASSWORD", "")
    write_token = os.environ.get("TRACKIO_WRITE_TOKEN", "")

    def _password_ok(authorization: str | None) -> bool:
        scheme, _, encoded = (authorization or "").partition(" ")
        if scheme.lower() != "basic" or not encoded:
            return False
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            return False
        _user, _, supplied = decoded.partition(":")
        return secrets.compare_digest(supplied, dashboard_password)

    def _write_token_ok(supplied: str) -> bool:
        return bool(write_token) and secrets.compare_digest(supplied, write_token)

    async def require_password(request, call_next):
        if dashboard_password and not (
            _password_ok(request.headers.get("Authorization"))
            or _write_token_ok(request.headers.get("X-Trackio-Write-Token", ""))
        ):
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="modal-dojo"'},
            )
        return await call_next(request)

    starlette_app.add_middleware(BaseHTTPMiddleware, dispatch=require_password)
    return starlette_app
