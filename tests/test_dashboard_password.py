"""Dashboard password setup: auth mode, password changes, and the CLI token."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import modal
import pytest

from modal_dojo import _dashboard
from modal_dojo.cli import setup as cli_setup
from modal_dojo.common import config


@pytest.fixture
def secret(monkeypatch, tmp_path):
    """Fake the password Secret; returns the ``Secret.objects`` mock."""
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / ".modal-dojo.toml")
    objects = Mock()
    monkeypatch.setattr(modal.Secret, "objects", objects)
    return objects


def _created_env(objects: Mock) -> dict[str, str]:
    return objects.create.call_args.args[1]


def test_password_change_replaces_the_secret(secret):
    token = _dashboard.set_dashboard_password("new")

    assert [call[0] for call in secret.mock_calls] == ["delete", "create"]
    assert _created_env(secret) == {
        "DASHBOARD_PASSWORD": "new",
        "DASHBOARD_CLI_TOKEN": token,
    }
    assert _dashboard.set_dashboard_password("") == ""
    assert _created_env(secret) == {"DASHBOARD_PASSWORD": ""}


def _update(
    password=None, *, configured=False, live_url=None, attended=False, proxy=False
):
    return cli_setup._update_password(
        password,
        configured=configured,
        live_url=live_url,
        attended=attended,
        require_proxy_auth=proxy,
    )


def test_first_deploy_generates_password_when_unattended(secret, capsys):
    assert _update() is True

    env = _created_env(secret)
    assert f"Dashboard password: {env['DASHBOARD_PASSWORD']}" in capsys.readouterr().out
    assert config.get_cli_token_from_config() == env["DASHBOARD_CLI_TOKEN"]


def test_first_deploy_prompts_until_passwords_match(secret, monkeypatch, capsys):
    answers = iter(["mine", "typo", "mine", "mine"])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: next(answers))

    assert _update(attended=True) is True

    assert _created_env(secret)["DASHBOARD_PASSWORD"] == "mine"
    assert "mine" not in capsys.readouterr().out


def test_existing_password_is_kept(secret):
    assert _update(configured=True) is None
    secret.create.assert_not_called()


@pytest.mark.parametrize(("proxy", "warns"), [(False, True), (True, False)])
def test_unattended_redeploy_of_open_dashboard_only_warns(secret, capsys, proxy, warns):
    assert _update(live_url="https://dash.test", proxy=proxy) is None

    secret.create.assert_not_called()
    assert ("has no password" in capsys.readouterr().out) is warns


def test_no_password_clears_it(secret):
    config.save_cli_token_to_config("old")

    assert _update(False, configured=True) is False

    assert _created_env(secret) == {"DASHBOARD_PASSWORD": ""}
    assert config.get_cli_token_from_config() == ""


@pytest.mark.parametrize(
    ("live", "creates"), [(False, True), (True, False), (None, False)]
)
def test_password_flag_only_replaces_a_password_that_is_off(
    secret, monkeypatch, live, creates
):
    monkeypatch.setattr(config, "get_dashboard_password_auth", lambda _url: live)

    _update(True, configured=True, live_url="https://dash.test")

    assert secret.create.called is creates


@pytest.mark.parametrize(
    ("answers", "mode"),
    [([""], (False, True)), (["3"], (True, False)), (["9", "4"], (False, False))],
)
def test_choose_auth_mode(monkeypatch, answers, mode):
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(replies))

    assert cli_setup._choose_auth_mode() == mode


@pytest.fixture
def stub_deploy(monkeypatch):
    """Run setup() without Modal; returns the mocks it decides with."""
    monkeypatch.setattr("modal_dojo.cli.skills.warn_legacy_skills", lambda: None)
    monkeypatch.setattr(config, "require_migrated_config", lambda: None)
    monkeypatch.setattr(config, "get_dashboard_url", lambda: "https://saved.test")
    monkeypatch.setattr(config, "get_dashboard_proxy_auth", lambda _url: False)
    monkeypatch.setattr(config, "get_dashboard_trajectory_viewer", lambda: None)
    monkeypatch.setattr(config, "save_dashboard_url", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_setup, "ensure_modal_setup", lambda: None)
    monkeypatch.setattr(cli_setup, "_password_secret_exists", lambda: False)
    mocks = SimpleNamespace(
        live_url=None,
        update=Mock(return_value=None),
        choose=Mock(return_value=(True, True)),
        password_auth=Mock(return_value=None),
        app=Mock(),
    )
    monkeypatch.setattr(cli_setup, "deployed_dashboard_url", lambda: mocks.live_url)
    monkeypatch.setattr(cli_setup, "_update_password", mocks.update)
    monkeypatch.setattr(cli_setup, "_choose_auth_mode", mocks.choose)
    monkeypatch.setattr(config, "get_dashboard_password_auth", mocks.password_auth)
    monkeypatch.setattr(cli_setup, "ensure_proxy_auth", lambda **_k: True)
    dashboard = SimpleNamespace(
        app=mocks.app,
        fastapi_app=Mock(get_web_url=lambda: "https://dash.test"),
        ensure_creds_secret=lambda **_k: True,
        MODAL_CREDS_SECRET_NAME="creds",
    )
    monkeypatch.setattr(
        cli_setup, "_load_dashboard_for_deploy", lambda *_a, **_k: dashboard
    )
    return mocks


def test_stale_saved_url_is_not_a_live_dashboard(stub_deploy):
    cli_setup.setup(interactive=False)

    assert stub_deploy.update.call_args.kwargs["live_url"] is None


def test_first_interactive_setup_asks_for_the_auth_mode(stub_deploy, monkeypatch):
    monkeypatch.setattr(cli_setup.sys.stdin, "isatty", lambda: True)

    cli_setup.setup()

    assert stub_deploy.update.call_args.args == (True,)
    assert stub_deploy.update.call_args.kwargs["require_proxy_auth"] is True


@pytest.mark.parametrize(
    ("changed", "live", "restart", "line", "strategy"),
    [
        (True, None, False, "Auth: password on, proxy auth off.", "recreate"),
        (None, False, False, "Auth: password off, proxy auth off.", "rolling"),
        (None, None, False, None, "rolling"),
        # set-password changed the Secret, so the running dashboard is stale.
        (None, False, True, None, "recreate"),
    ],
)
def test_auth_line_reflects_the_live_dashboard(
    stub_deploy, capsys, changed, live, restart, line, strategy
):
    stub_deploy.live_url = "https://dash.test"
    stub_deploy.update.return_value = changed
    stub_deploy.password_auth.return_value = live

    cli_setup.setup(interactive=False, always_restart=restart)

    out = capsys.readouterr().out
    assert (line in out) if line else ("Auth:" not in out)
    assert stub_deploy.app.deploy.call_args.kwargs["strategy"] == strategy
