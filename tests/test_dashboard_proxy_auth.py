from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from modal_dojo.cli.setup import ProxyAuthMode
from fastapi.testclient import TestClient

from modal_dojo import _dashboard
from modal_dojo.cli import setup as cli_setup_module
from modal_dojo.common import config
from modal_dojo.common import status_reporter
from modal_dojo.common import reporting
from modal_dojo.common.dashboard import (
    DASHBOARD_VERSION,
    DashboardLookupUnknown,
    current_dashboard_version,
    deployed_dashboard_url,
    is_dashboard_upgrade,
)


class _Response:
    def __init__(self, body: bytes = b""):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.body


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / ".training-gym.toml"
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    return path


def test_dashboard_proxy_auth_mode_is_persisted(config_path, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise URLError("unavailable")

    monkeypatch.setattr(config, "urlopen", unavailable)
    config.save_dashboard_url("https://dashboard.test", proxy_auth=True)

    assert config.get_dashboard_url() == "https://dashboard.test"
    assert config.get_dashboard_proxy_auth("https://dashboard.test") is True
    assert "proxy_auth = true" in config_path.read_text()


@pytest.mark.parametrize(("body", "expected"), [(b"true", True), (b"false", False)])
def test_live_dashboard_proxy_auth_mode_is_authoritative(
    config_path, monkeypatch, body, expected
):
    config.save_dashboard_url("https://dashboard.test", proxy_auth=not expected)
    monkeypatch.setattr(config, "urlopen", lambda *_args, **_kwargs: _Response(body))

    assert config.get_dashboard_proxy_auth("https://dashboard.test") is expected


def test_dashboard_proxy_auth_treats_403_as_enabled(config_path, monkeypatch):
    config.save_dashboard_url("https://dashboard.test", proxy_auth=False)

    def forbidden(*_args, **_kwargs):
        raise HTTPError("https://dashboard.test", 403, "Forbidden", {}, None)

    monkeypatch.setattr(config, "urlopen", forbidden)

    assert config.get_dashboard_proxy_auth("https://dashboard.test") is True


def test_proxy_auth_status_does_not_require_basic_auth(monkeypatch, tmp_path):
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("ok")
    (static / "favicon.svg").write_text("<svg/>")
    monkeypatch.setattr(_dashboard, "STATIC_DIR", str(static))
    monkeypatch.setattr(config, "_dashboard_requires_proxy_auth", False)
    monkeypatch.setenv("DASHBOARD_PASSWORD", "password")

    with TestClient(_dashboard.fastapi_app.local()) as client:
        response = client.get(config.DASHBOARD_PROXY_AUTH_PATH)

    assert response.status_code == 200
    assert response.json() is False


def test_dashboard_import_sets_proxy_auth_mode(monkeypatch):
    dashboard = SimpleNamespace()
    observed = []

    def reload_module(module):
        observed.append(config.dashboard_requires_proxy_auth())
        return module

    monkeypatch.setattr(config, "_dashboard_requires_proxy_auth", False)
    monkeypatch.setitem(sys.modules, "modal_dojo._dashboard", dashboard)
    monkeypatch.setattr(importlib, "reload", reload_module)

    loaded = cli_setup_module._load_dashboard_for_deploy(True)

    assert loaded is dashboard
    assert observed == [True]
    assert config.dashboard_requires_proxy_auth() is True


@pytest.mark.parametrize(
    ("last_proxy_auth", "expected"),
    [(True, True), (False, False), (None, False)],
)
def test_auto_deploy_reuses_proxy_auth_mode(monkeypatch, last_proxy_auth, expected):
    calls = []
    monkeypatch.setattr(
        cli_setup_module,
        "deployed_dashboard_url",
        lambda: None,
    )
    monkeypatch.setattr(
        config,
        "get_dashboard_proxy_auth",
        lambda: last_proxy_auth,
    )

    def setup(**kwargs):
        calls.append(kwargs)
        return "https://dashboard.test"

    monkeypatch.setattr(cli_setup_module, "setup", setup)

    assert cli_setup_module.ensure_dashboard_deployed() == "https://dashboard.test"
    assert calls == [{"interactive": False, "proxy_auth": ProxyAuthMode.UNSPECIFIED}]


class _ModalFn:
    def __init__(self, *, hydrate_error=None, web_url="https://dashboard.test"):
        self._hydrate_error = hydrate_error
        self._web_url = web_url

    def hydrate(self):
        if self._hydrate_error is not None:
            raise self._hydrate_error

    def get_web_url(self):
        return self._web_url


def _stub_modal_function(monkeypatch, fn):
    import modal

    monkeypatch.setattr(modal.Function, "from_name", lambda *_args, **_kwargs: fn)


def test_deployed_dashboard_url_not_found_is_missing(monkeypatch):
    from modal.exception import NotFoundError

    _stub_modal_function(monkeypatch, _ModalFn(hydrate_error=NotFoundError("gone")))

    assert deployed_dashboard_url() is None


def test_deployed_dashboard_url_lookup_failure_is_unknown(monkeypatch):
    _stub_modal_function(monkeypatch, _ModalFn(hydrate_error=TimeoutError("timed out")))

    with pytest.raises(DashboardLookupUnknown):
        deployed_dashboard_url()


def test_deployed_dashboard_url_empty_web_url_is_unknown(monkeypatch):
    _stub_modal_function(monkeypatch, _ModalFn(web_url=None))

    with pytest.raises(DashboardLookupUnknown):
        deployed_dashboard_url()


def test_auto_deploy_skips_when_dashboard_lookup_is_unknown(config_path, monkeypatch):
    config.save_dashboard_url("https://cached.test")

    def unknown():
        raise DashboardLookupUnknown("blip")

    monkeypatch.setattr(cli_setup_module, "deployed_dashboard_url", unknown)
    calls = _record_setup(monkeypatch)

    assert cli_setup_module.ensure_dashboard_deployed() == "https://cached.test"
    assert calls == []


def _deployed_dashboard(monkeypatch, *, live_version, local_version):
    monkeypatch.setattr(
        cli_setup_module, "deployed_dashboard_url", lambda: "https://dashboard.test"
    )
    monkeypatch.setattr(
        cli_setup_module, "current_dashboard_version", lambda: local_version
    )
    monkeypatch.setattr(config, "get_dashboard_version", lambda _url: live_version)
    monkeypatch.setattr(config, "get_dashboard_proxy_auth", lambda: False)


def _record_setup(monkeypatch):
    calls = []

    def setup(**kwargs):
        calls.append(kwargs)
        return "https://dashboard.test"

    monkeypatch.setattr(cli_setup_module, "setup", setup)
    return calls


def test_auto_deploy_skips_when_version_is_not_newer(config_path, monkeypatch):
    _deployed_dashboard(monkeypatch, live_version="1", local_version="1")
    calls = _record_setup(monkeypatch)

    assert cli_setup_module.ensure_dashboard_deployed() == "https://dashboard.test"
    assert calls == []


def test_auto_deploy_redeploys_when_incoming_version_is_newer(config_path, monkeypatch):
    _deployed_dashboard(monkeypatch, live_version=1, local_version=2)
    calls = _record_setup(monkeypatch)

    assert cli_setup_module.ensure_dashboard_deployed() == "https://dashboard.test"
    assert calls == [{"interactive": False, "proxy_auth": ProxyAuthMode.UNSPECIFIED}]


def _raise_version_error(error):
    def urlopen(*_args, **_kwargs):
        raise error

    return urlopen


def test_auto_deploy_skips_when_version_is_unknown(config_path, monkeypatch):
    monkeypatch.setattr(
        cli_setup_module, "deployed_dashboard_url", lambda: "https://dashboard.test"
    )
    monkeypatch.setattr(cli_setup_module, "current_dashboard_version", lambda: "2")
    monkeypatch.setattr(config, "get_dashboard_proxy_auth", lambda: False)
    monkeypatch.setattr(
        config,
        "urlopen",
        _raise_version_error(
            HTTPError(
                "https://dashboard.test/api/version", 401, "Unauthorized", {}, None
            )
        ),
    )
    calls = _record_setup(monkeypatch)

    assert cli_setup_module.ensure_dashboard_deployed() == "https://dashboard.test"
    assert calls == []


@pytest.mark.parametrize(
    ("incoming", "deployed", "expected"),
    [
        ("2", None, True),
        ("2", "1", True),
        ("1", "1", False),
        ("1", "deadbeef", True),
    ],
)
def test_is_dashboard_upgrade(incoming, deployed, expected):
    assert is_dashboard_upgrade(incoming, deployed) is expected


def test_current_dashboard_version_uses_source_constant(monkeypatch):
    monkeypatch.delenv(_dashboard.DASHBOARD_VERSION_ENV_KEY, raising=False)
    assert current_dashboard_version() == str(DASHBOARD_VERSION)


def test_current_dashboard_version_uses_baked_env(monkeypatch):
    monkeypatch.setenv(_dashboard.DASHBOARD_VERSION_ENV_KEY, "9")

    assert current_dashboard_version() == "9"


def test_get_dashboard_version_reads_success_bodies(monkeypatch):
    monkeypatch.setattr(config, "urlopen", lambda *_args, **_kwargs: _Response(b'"2"'))
    assert config.get_dashboard_version("https://dashboard.test") == "2"


def test_get_dashboard_version_404_is_unversioned(monkeypatch):
    monkeypatch.setattr(
        config,
        "urlopen",
        _raise_version_error(
            HTTPError("https://dashboard.test/api/version", 404, "Not Found", {}, None)
        ),
    )
    assert config.get_dashboard_version("https://dashboard.test") is None


def test_get_dashboard_version_unread_is_unknown(monkeypatch):
    monkeypatch.setattr(
        config,
        "urlopen",
        _raise_version_error(
            HTTPError(
                "https://dashboard.test/api/version", 401, "Unauthorized", {}, None
            )
        ),
    )
    with pytest.raises(config.DashboardVersionUnknown):
        config.get_dashboard_version("https://dashboard.test")


def _capture_report(reporter, monkeypatch):
    requests = []

    def urlopen(request, **_kwargs):
        requests.append(request)
        return _Response()

    monkeypatch.setattr(reporter, "urlopen", urlopen)
    return requests


def test_status_reporting_posts_include_proxy_auth_headers(config_path, monkeypatch):
    config.save_proxy_auth("wk-test", "ws-test")
    monkeypatch.delenv("MODAL_KEY", raising=False)
    monkeypatch.delenv("MODAL_SECRET", raising=False)
    requests = _capture_report(status_reporter, monkeypatch)
    status_reporter.post_item(
        {
            "_url": "https://dashboard.test/api/framework-status",
            "_timeout": 1,
            "_token": "run-token",
            "training_run_id": "run-1",
            "phase": "training",
        }
    )

    headers = dict(requests[0].header_items())
    assert headers["Modal-key"] == "wk-test"
    assert headers["Modal-secret"] == "ws-test"
    assert headers["Authorization"] == "Bearer run-token"


def test_slime_reporting_posts_include_proxy_auth_headers(config_path, monkeypatch):
    config.save_proxy_auth("wk-test", "ws-test")
    monkeypatch.delenv("MODAL_KEY", raising=False)
    monkeypatch.delenv("MODAL_SECRET", raising=False)
    monkeypatch.setenv("MODAL_DOJO_FRAMEWORK_STATUS_TOKEN", "run-token")
    requests = _capture_report(reporting, monkeypatch)
    reporting._post(
        {
            "_url": "https://dashboard.test/api/framework-status",
            "_timeout": 1,
            "training_run_id": "run-1",
            "phase": "training",
        }
    )

    headers = dict(requests[0].header_items())
    assert headers["Modal-key"] == "wk-test"
    assert headers["Modal-secret"] == "ws-test"
    assert headers["Authorization"] == "Bearer run-token"
