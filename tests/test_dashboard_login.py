"""Dashboard login page and session cookie."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from modal_dojo import _dashboard
from modal_dojo import _dashboard_login as login
from modal_dojo.cli import setup as cli_setup
from modal_dojo.cli.errors import CLIError
from modal_dojo.common import config

PASSWORD = "correct horse"
HTML = {"Accept": "text/html"}


@pytest.fixture
def dashboard(monkeypatch, tmp_path):
    def make(password: str = PASSWORD) -> TestClient:
        (tmp_path / "assets").mkdir(exist_ok=True)
        (tmp_path / "index.html").write_text("spa")
        (tmp_path / "favicon.svg").write_text("<svg/>")
        monkeypatch.setattr(_dashboard, "STATIC_DIR", str(tmp_path))
        monkeypatch.setattr(login, "FAILED_LOGIN_DELAY_SECONDS", 0)
        monkeypatch.setenv("DASHBOARD_PASSWORD", password)
        # https so the Secure cookie round-trips.
        return TestClient(
            _dashboard.fastapi_app.local(),
            base_url="https://testserver",
            follow_redirects=False,
        )

    return make


def _basic(password: str) -> dict[str, str]:
    token = base64.b64encode(f"modal-dojo:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_open_dashboard_needs_no_login(dashboard):
    with dashboard(password="") as client:
        assert client.get("/", headers=HTML).text == "spa"
        assert client.get("/login?next=/runs").headers["location"] == "/runs"
        assert client.get(config.DASHBOARD_LOGIN_LINK_PATH).json() == "/"


def test_unauthenticated_requests(dashboard):
    with dashboard() as client:
        page = client.get("/runs/abc?tab=logs", headers=HTML)
        api = client.get("/api/runs")

    assert page.status_code == 303
    assert page.headers["location"] == "/login?next=%2Fruns%2Fabc%3Ftab%3Dlogs"
    assert api.status_code == 401
    assert "www-authenticate" not in api.headers
    assert api.headers[login.LOGIN_REQUIRED_HEADER] == login.LOGIN_PATH


def test_login_logout(dashboard):
    with dashboard() as client:
        page = client.get("/login?next=/runs", headers=HTML)
        assert 'name="next" value="/runs"' in page.text
        assert page.headers["x-frame-options"] == "DENY"

        wrong = client.post("/login", data={"password": "nope", "next": "/runs"})
        assert wrong.status_code == 401
        assert "Incorrect password" in wrong.text

        right = client.post("/login", data={"password": PASSWORD, "next": "/runs"})
        assert right.headers["location"] == "/runs"
        assert client.get("/", headers=HTML).text == "spa"

        client.get("/logout")
        assert client.get("/", headers=HTML).text == "spa"
        client.post("/logout")
        assert client.get("/", headers=HTML).status_code == 303


def test_basic_auth_still_works_for_cli(dashboard):
    with dashboard() as client:
        assert client.get("/", headers=_basic(PASSWORD)).text == "spa"
        assert client.get("/", headers=_basic("wrong")).status_code == 401


def test_cli_token(dashboard, monkeypatch):
    monkeypatch.setenv("DASHBOARD_CLI_TOKEN", "cli-token")
    with dashboard() as client:
        ok = client.get("/", headers={"Authorization": "Bearer cli-token"})
        wrong = client.get("/", headers={"Authorization": "Bearer wrong"})

    assert ok.text == "spa"
    assert wrong.status_code == 401


@pytest.mark.parametrize("password", [PASSWORD, ""])
def test_password_auth_status_needs_no_login(dashboard, password):
    with dashboard(password=password) as client:
        assert client.get(config.DASHBOARD_PASSWORD_AUTH_PATH).json() is bool(password)


@pytest.mark.parametrize(
    "next_path", ["//evil.test", "https://evil.test", "/\\evil.test", "/login"]
)
def test_login_never_redirects_off_site(next_path):
    assert login.safe_next_path(next_path) == "/"


def test_session_token():
    token = login.make_session_token(PASSWORD, now=1000)
    expires, _, signature = token.partition(".")

    assert login.session_token_ok(token, PASSWORD, now=1000)
    assert not login.session_token_ok(token, "new password", now=1000)
    assert not login.session_token_ok(
        token, PASSWORD, now=1001 + login.SESSION_TTL_SECONDS
    )
    assert not login.session_token_ok(f"{int(expires) + 1}.{signature}", PASSWORD)
    assert not login.session_token_ok("9999999999.é", PASSWORD, now=0)


def test_login_link_token():
    token = login.make_login_link_token(PASSWORD, now=1000)

    assert login.login_link_token_ok(token, PASSWORD, now=1000)
    assert not login.login_link_token_ok(
        token, PASSWORD, now=1001 + login.LOGIN_LINK_TTL_SECONDS
    )
    # Neither kind of token passes for the other.
    assert not login.session_token_ok(token, PASSWORD, now=1000)
    session = login.make_session_token(PASSWORD, now=1000)
    assert not login.login_link_token_ok(session, PASSWORD, now=1000)


def test_login_link(dashboard, monkeypatch):
    monkeypatch.setenv("DASHBOARD_CLI_TOKEN", "cli-token")
    cli = {"Authorization": "Bearer cli-token"}
    with dashboard() as client:
        assert client.get(config.DASHBOARD_LOGIN_LINK_PATH).status_code == 401
        link = client.get(config.DASHBOARD_LOGIN_LINK_PATH, headers=cli).json()
        assert client.get("/login?token=forged", headers=HTML).status_code == 200

        assert client.get(link).headers["location"] == "/"
        assert client.get("/", headers=HTML).text == "spa"


@pytest.mark.parametrize(
    "link", ["/login?token=abc", None, OSError("disk full"), "//evil.test/login"]
)
def test_open_uses_a_login_link(monkeypatch, capsys, link):
    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

        def get_json(self, _path):
            if link is None:
                raise CLIError("Dashboard authentication was rejected.", error="x")
            if isinstance(link, Exception):
                raise link
            return link

    opened = []
    monkeypatch.setattr(cli_setup, "deployed_dashboard_url", lambda: "https://d.test")
    monkeypatch.setattr(config, "get_dashboard_url", lambda: "https://d.test")
    monkeypatch.setattr("modal_dojo.cli.client.DashboardClient", FakeClient)
    monkeypatch.setattr(cli_setup.webbrowser, "open", opened.append)

    cli_setup.open_dashboard()

    if link == "/login?token=abc":
        assert opened == ["https://d.test/login?token=abc"]
    else:
        assert opened == ["https://d.test"]
    assert "token" not in capsys.readouterr().out
