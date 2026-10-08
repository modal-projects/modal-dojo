"""Shared HTTP client for the deployed modal-dojo dashboard."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlsplit

import httpx

from modal_dojo.common.config import (
    get_dashboard_url,
    modal_proxy_auth_headers,
)

from .errors import CLIError, ExitCode


DASHBOARD_PASSWORD_ENV = "MODAL_DOJO_DASHBOARD_PASSWORD"
DEFAULT_TIMEOUT_SECONDS = 10.0
QueryParams = Mapping[str, str | int | float | bool | None]


def _strip_proxy_auth_on_cross_origin_redirect(response: httpx.Response) -> None:
    """Prevent Modal proxy credentials from following redirects to another origin."""
    location = response.headers.get("location")
    if not response.is_redirect or not location:
        return

    source = response.request.url
    target = source.join(location)
    if (
        source.scheme,
        source.host,
        source.port,
    ) != (
        target.scheme,
        target.host,
        target.port,
    ):
        response.request.headers.pop("Modal-Key", None)
        response.request.headers.pop("Modal-Secret", None)


class DashboardClient:
    """Transport shared by dashboard-backed commands."""

    def __init__(
        self,
        *,
        password: str | None = None,
    ) -> None:
        configured_url = (get_dashboard_url() or "").strip()
        if not configured_url:
            raise CLIError(
                "Dashboard URL is not configured.",
                error="dashboard_not_configured",
                exit_code=ExitCode.BACKEND,
                hint="modal-dojo setup",
            )

        parsed = urlsplit(configured_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise CLIError(
                "Configured dashboard URL must use HTTP or HTTPS.",
                error="dashboard_configuration_invalid",
                exit_code=ExitCode.BACKEND,
                hint="modal-dojo setup",
            )

        dashboard_password = (
            os.environ.get(DASHBOARD_PASSWORD_ENV, "") if password is None else password
        )
        auth = (
            httpx.BasicAuth("modal-dojo", dashboard_password)
            if dashboard_password
            else None
        )
        self._client = httpx.Client(
            base_url=configured_url.rstrip("/") + "/",
            auth=auth,
            headers=modal_proxy_auth_headers(),
            event_hooks={"response": [_strip_proxy_auth_on_cross_origin_redirect]},
            timeout=DEFAULT_TIMEOUT_SECONDS,
            follow_redirects=True,
        )

    def get_json(
        self,
        path: str,
        *,
        params: QueryParams | None = None,
        not_found_error: CLIError | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Any:
        """GET a dashboard-relative path and decode its JSON response."""
        parsed_path = urlsplit(path)
        if parsed_path.scheme or parsed_path.netloc:
            raise CLIError(
                "Dashboard request path must be relative.",
                error="invalid_dashboard_path",
            )

        query = (
            {key: value for key, value in params.items() if value is not None}
            if params
            else None
        )
        try:
            response = self._client.get(
                path.lstrip("/"),
                params=query,
                timeout=DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout,
            )
        except httpx.TimeoutException as exc:
            raise CLIError(
                "Dashboard request timed out.",
                error="dashboard_timeout",
                exit_code=ExitCode.BACKEND,
            ) from exc
        except httpx.RequestError as exc:
            raise CLIError(
                "Could not connect to the dashboard.",
                error="dashboard_unreachable",
                exit_code=ExitCode.BACKEND,
                hint="modal-dojo setup",
            ) from exc

        if response.status_code == 404 and not_found_error is not None:
            raise not_found_error

        self._raise_for_status(response)
        try:
            return response.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise CLIError(
                "Dashboard returned malformed JSON.",
                error="invalid_dashboard_response",
                exit_code=ExitCode.BACKEND,
            ) from exc

    def post_json(
        self,
        path: str,
        body: Any,
        *,
        timeout: float | httpx.Timeout | None = None,
    ) -> Any:
        """POST a JSON body to a dashboard-relative path and decode the response."""
        parsed_path = urlsplit(path)
        if parsed_path.scheme or parsed_path.netloc:
            raise CLIError(
                "Dashboard request path must be relative.",
                error="invalid_dashboard_path",
            )
        try:
            response = self._client.post(
                path.lstrip("/"),
                json=body,
                timeout=DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout,
            )
        except httpx.TimeoutException as exc:
            raise CLIError(
                "Dashboard request timed out.",
                error="dashboard_timeout",
                exit_code=ExitCode.BACKEND,
            ) from exc
        except httpx.RequestError as exc:
            raise CLIError(
                "Could not connect to the dashboard.",
                error="dashboard_unreachable",
                exit_code=ExitCode.BACKEND,
                hint="modal-dojo setup",
            ) from exc

        self._raise_for_status(response)
        try:
            return response.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise CLIError(
                "Dashboard returned malformed JSON.",
                error="invalid_dashboard_response",
                exit_code=ExitCode.BACKEND,
            ) from exc

    def iter_event_stream(
        self,
        path: str,
        *,
        params: QueryParams | None = None,
        not_found_error: CLIError | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Stream ``(event, data)`` pairs from a dashboard SSE endpoint."""
        parsed_path = urlsplit(path)
        if parsed_path.scheme or parsed_path.netloc:
            raise CLIError(
                "Dashboard request path must be relative.",
                error="invalid_dashboard_path",
            )

        query = (
            {key: value for key, value in params.items() if value is not None}
            if params
            else None
        )
        try:
            with self._client.stream(
                "GET",
                path.lstrip("/"),
                params=query,
                timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS, read=None),
            ) as response:
                if response.status_code == 404 and not_found_error is not None:
                    raise not_found_error
                if response.status_code >= 400:
                    response.read()
                self._raise_for_status(response)

                event = "message"
                data_lines: list[str] = []
                for line in response.iter_lines():
                    if not line:
                        if data_lines:
                            yield event, "\n".join(data_lines)
                        event = "message"
                        data_lines = []
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event = line.removeprefix("event:").lstrip() or "message"
                    elif line.startswith("data:"):
                        data_lines.append(line.removeprefix("data:").lstrip())

                if data_lines:
                    yield event, "\n".join(data_lines)
        except httpx.TimeoutException as exc:
            raise CLIError(
                "Dashboard log stream timed out.",
                error="dashboard_timeout",
                exit_code=ExitCode.BACKEND,
                hint="Re-run the command to retry.",
            ) from exc
        except httpx.RequestError as exc:
            raise CLIError(
                "Dashboard log stream disconnected.",
                error="dashboard_unreachable",
                exit_code=ExitCode.BACKEND,
                hint="Re-run the command to reconnect.",
            ) from exc

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        status_code = response.status_code
        if status_code in {401, 403}:
            raise CLIError(
                "Dashboard authentication was rejected.",
                error="authentication_failed",
                exit_code=ExitCode.AUTH,
                hint=(
                    "Run `modal-dojo set-proxy-auth` for Modal proxy auth, "
                    "or `modal-dojo set-password` for dashboard Basic Auth."
                ),
            )
        if status_code == 404:
            raise CLIError(
                "The deployed dashboard does not support this resource.",
                error="dashboard_resource_not_found",
                exit_code=ExitCode.BACKEND,
                hint="modal-dojo setup",
            )
        if status_code >= 500:
            raise CLIError(
                f"Dashboard returned HTTP {status_code}.",
                error="dashboard_server_error",
                exit_code=ExitCode.BACKEND,
                status_code=status_code,
            )
        if status_code >= 400:
            try:
                payload = response.json()
                detail = payload.get("detail") if isinstance(payload, dict) else None
            except (ValueError, UnicodeDecodeError, httpx.ResponseNotRead):
                detail = None
            raise CLIError(
                (
                    detail
                    if isinstance(detail, str)
                    else f"Dashboard returned HTTP {status_code}."
                ),
                error="dashboard_request_failed",
                exit_code=ExitCode.BACKEND,
                status_code=status_code,
            )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
