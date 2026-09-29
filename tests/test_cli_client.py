from __future__ import annotations

import base64

import httpx
import pytest

from modal_dojo.cli import client as client_module
from modal_dojo.cli.client import (
    DEFAULT_TIMEOUT_SECONDS,
    DashboardClient,
)
from modal_dojo.cli.errors import CLIError, ExitCode


@pytest.fixture(autouse=True)
def configured_dashboard_url(monkeypatch):
    monkeypatch.setattr(
        client_module,
        "get_dashboard_url",
        lambda: "https://example.test",
    )
    monkeypatch.setattr(client_module, "modal_proxy_auth_headers", lambda: {})


@pytest.fixture
def mock_transport(monkeypatch):
    real_client = httpx.Client

    def install(handler):
        monkeypatch.setattr(
            client_module.httpx,
            "Client",
            lambda **kwargs: real_client(
                **kwargs,
                transport=httpx.MockTransport(handler),
            ),
        )

    return install


def test_uses_configured_url_and_encodes_query(monkeypatch, mock_transport):
    monkeypatch.setattr(
        client_module, "get_dashboard_url", lambda: "https://example.test/root/"
    )
    seen = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    mock_transport(respond)
    with DashboardClient() as client:
        result = client.get_json(
            "/api/items",
            params={"model": "a b", "limit": 2, "unused": None},
        )

    assert result == {"ok": True}
    assert str(seen[0].url) == ("https://example.test/root/api/items?model=a+b&limit=2")
    assert seen[0].extensions["timeout"]["read"] == DEFAULT_TIMEOUT_SECONDS


def test_get_json_supports_per_request_timeout(mock_transport):
    seen = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    mock_transport(respond)
    with DashboardClient() as client:
        assert client.get_json("/api/items", timeout=120.0) == {"ok": True}

    assert seen[0].extensions["timeout"] == {
        "connect": 120.0,
        "read": 120.0,
        "write": 120.0,
        "pool": 120.0,
    }


def test_sends_basic_auth_when_password_exists(monkeypatch, mock_transport):
    monkeypatch.setenv("TRAINING_GYM_DASHBOARD_PASSWORD", "secret")
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    mock_transport(respond)
    with DashboardClient() as client:
        client.get_json("/api/items")

    expected = base64.b64encode(b"training-gym:secret").decode()
    assert requests[0].headers["authorization"] == f"Basic {expected}"


def test_sends_modal_proxy_auth_headers(monkeypatch, mock_transport):
    monkeypatch.setattr(
        client_module,
        "modal_proxy_auth_headers",
        lambda: {"Modal-Key": "wk-test", "Modal-Secret": "ws-test"},
    )
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    mock_transport(respond)
    with DashboardClient() as client:
        client.get_json("/api/items")

    assert requests[0].headers["modal-key"] == "wk-test"
    assert requests[0].headers["modal-secret"] == "ws-test"


def test_does_not_forward_proxy_auth_to_redirected_host(monkeypatch, mock_transport):
    monkeypatch.setattr(
        client_module,
        "modal_proxy_auth_headers",
        lambda: {"Modal-Key": "wk-test", "Modal-Secret": "ws-test"},
    )
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append((request.url, dict(request.headers)))
        if request.url.host == "example.test":
            return httpx.Response(
                302,
                headers={"location": "https://other.test/api/items"},
            )
        return httpx.Response(200, json={})

    mock_transport(respond)
    with DashboardClient() as client:
        client.get_json("/api/items")

    assert requests[0][1]["modal-key"] == "wk-test"
    assert requests[0][1]["modal-secret"] == "ws-test"
    assert "modal-key" not in requests[1][1]
    assert "modal-secret" not in requests[1][1]


def test_omits_auth_when_password_is_absent(monkeypatch, mock_transport):
    monkeypatch.delenv("TRAINING_GYM_DASHBOARD_PASSWORD", raising=False)
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    mock_transport(respond)
    with DashboardClient() as client:
        client.get_json("/api/items")

    assert "authorization" not in requests[0].headers


@pytest.mark.parametrize(
    ("status_code", "error", "exit_code"),
    [
        (401, "authentication_failed", ExitCode.AUTH),
        (403, "authentication_failed", ExitCode.AUTH),
        (404, "dashboard_resource_not_found", ExitCode.BACKEND),
        (500, "dashboard_server_error", ExitCode.BACKEND),
        (503, "dashboard_server_error", ExitCode.BACKEND),
        (400, "dashboard_request_failed", ExitCode.BACKEND),
    ],
)
def test_maps_http_errors(status_code, error, exit_code, mock_transport):
    mock_transport(lambda _request: httpx.Response(status_code, text="ignored"))
    with DashboardClient() as client:
        with pytest.raises(CLIError) as exc_info:
            client.get_json("/api/items")

    assert exc_info.value.error == error
    assert exc_info.value.exit_code == exit_code


def test_surfaces_dashboard_detail_for_bad_request(mock_transport):
    mock_transport(
        lambda _request: httpx.Response(
            400,
            json={
                "detail": (
                    "since must be epoch seconds, ISO 8601, "
                    "or a relative time such as 24h"
                )
            },
        )
    )

    with DashboardClient() as client:
        with pytest.raises(CLIError) as exc_info:
            client.get_json("/api/runs/run-1/logs")

    assert str(exc_info.value) == (
        "since must be epoch seconds, ISO 8601, or a relative time such as 24h"
    )
    assert exc_info.value.error == "dashboard_request_failed"


def test_uses_command_specific_not_found_error(mock_transport):
    mock_transport(lambda _request: httpx.Response(404, text="ignored"))
    not_found = CLIError(
        "Run not found.",
        error="run_not_found",
        exit_code=ExitCode.NOT_FOUND,
        run_id="run-1",
    )

    with DashboardClient() as client:
        with pytest.raises(CLIError) as exc_info:
            client.get_json("/api/runs/run-1", not_found_error=not_found)

    assert exc_info.value is not_found
    assert exc_info.value.error == "run_not_found"
    assert exc_info.value.exit_code == ExitCode.NOT_FOUND


def test_maps_timeout_without_leaking_transport_details(mock_transport):
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret transport detail", request=request)

    mock_transport(timeout)
    with DashboardClient() as client:
        with pytest.raises(CLIError) as exc_info:
            client.get_json("/api/items")

    assert exc_info.value.error == "dashboard_timeout"
    assert exc_info.value.exit_code == ExitCode.BACKEND
    assert "secret transport detail" not in str(exc_info.value)


def test_maps_connection_failure(mock_transport):
    def disconnect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("host detail", request=request)

    mock_transport(disconnect)
    with DashboardClient() as client:
        with pytest.raises(CLIError) as exc_info:
            client.get_json("/api/items")

    assert exc_info.value.error == "dashboard_unreachable"
    assert exc_info.value.exit_code == ExitCode.BACKEND


def test_rejects_malformed_json(mock_transport):
    mock_transport(lambda _request: httpx.Response(200, text="<html>not json</html>"))

    with DashboardClient() as client:
        with pytest.raises(CLIError) as exc_info:
            client.get_json("/api/items")

    assert exc_info.value.error == "invalid_dashboard_response"
    assert exc_info.value.exit_code == ExitCode.BACKEND


@pytest.mark.parametrize(
    "url",
    [None, "", "not-a-url", "ftp://example.test"],
)
def test_rejects_missing_or_invalid_configuration(monkeypatch, url):
    monkeypatch.setattr(client_module, "get_dashboard_url", lambda: url)

    with pytest.raises(CLIError) as exc_info:
        DashboardClient()

    assert exc_info.value.exit_code == ExitCode.BACKEND


def test_rejects_absolute_request_path():
    with DashboardClient() as client:
        with pytest.raises(CLIError) as exc_info:
            client.get_json("https://other.test/api/items")

    assert exc_info.value.error == "invalid_dashboard_path"
    assert exc_info.value.exit_code == ExitCode.ERROR


def test_iter_event_stream_parses_named_and_default_events(mock_transport):
    mock_transport(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                ": keepalive\n\n"
                'data: {"line":"hello"}\n\n'
                "event: dropped\n"
                'data: {"dropped":2}\n\n'
                "event: reconnect\n"
                'data: {"reason":"temporary error"}\n\n'
                "event: done\n"
                "data: {}\n\n"
            ),
        )
    )

    with DashboardClient() as client:
        events = list(client.iter_event_stream("/api/logs/stream"))

    assert events == [
        ("message", '{"line":"hello"}'),
        ("dropped", '{"dropped":2}'),
        ("reconnect", '{"reason":"temporary error"}'),
        ("done", "{}"),
    ]


def test_iter_event_stream_disables_read_timeout(mock_transport):
    seen = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="event: done\ndata: {}\n\n")

    mock_transport(respond)
    with DashboardClient() as client:
        assert list(client.iter_event_stream("/api/logs/stream")) == [("done", "{}")]

    timeout = seen[0].extensions["timeout"]
    assert timeout == {
        "connect": DEFAULT_TIMEOUT_SECONDS,
        "read": None,
        "write": DEFAULT_TIMEOUT_SECONDS,
        "pool": DEFAULT_TIMEOUT_SECONDS,
    }
