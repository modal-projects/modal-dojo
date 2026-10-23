"""Dashboard login page and session cookie."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from modal_dojo import _dashboard
from modal_dojo import _dashboard_login as login

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
        assert client.get("/", headers=HTML).status_code == 303


def test_basic_auth_still_works_for_cli(dashboard):
    with dashboard() as client:
        assert client.get("/", headers=_basic(PASSWORD)).text == "spa"
        assert client.get("/", headers=_basic("wrong")).status_code == 401


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
