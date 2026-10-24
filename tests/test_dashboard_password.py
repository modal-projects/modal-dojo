"""Dashboard password setup: first-deploy password and password changes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import modal
import pytest
from modal.exception import NotFoundError

from modal_dojo import _dashboard
from modal_dojo.cli import setup as cli_setup
from modal_dojo.common import config


def test_password_change_replaces_the_secret(monkeypatch):
    objects = Mock()
    monkeypatch.setattr(modal.Secret, "objects", objects)

    _dashboard.set_dashboard_password("new")

    assert [call[0] for call in objects.mock_calls] == ["delete", "create"]


@pytest.fixture
def secret(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / ".modal-dojo.toml")
    create = Mock()
    monkeypatch.setattr(modal.Secret, "objects", Mock(create=create))

    def install(*, exists: bool) -> Mock:
        handle = Mock()
        if not exists:
            handle.hydrate.side_effect = NotFoundError("missing")
        monkeypatch.setattr(modal.Secret, "from_name", lambda *_a, **_k: handle)
        return create

    return install


def test_first_deploy_generates_password_when_unattended(secret):
    create = secret(exists=False)

    assert cli_setup.ensure_dashboard_password(interactive=False) == "created"

    generated = create.call_args.args[1]["DASHBOARD_PASSWORD"]
    assert config.get_dashboard_password() == generated


def test_first_deploy_prompts_until_passwords_match(secret, monkeypatch):
    create = secret(exists=False)
    monkeypatch.setattr(cli_setup.sys.stdin, "isatty", lambda: True)
    answers = iter(["mine", "typo", "mine", "mine"])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: next(answers))

    cli_setup.ensure_dashboard_password(interactive=True)

    assert create.call_args.args[1] == {"DASHBOARD_PASSWORD": "mine"}


def test_existing_secret_is_left_alone(secret):
    create = secret(exists=True)

    assert cli_setup.ensure_dashboard_password(interactive=False) == "existing"
    create.assert_not_called()


def test_unattended_redeploy_of_open_dashboard_only_warns(secret):
    create = secret(exists=False)

    state = cli_setup.ensure_dashboard_password(
        interactive=False, existing_dashboard=True
    )

    assert state == "missing"
    create.assert_not_called()


@pytest.fixture
def stub_deploy(monkeypatch):
    """Run setup() without Modal; returns the ensure_dashboard_password mock."""
    monkeypatch.setattr("modal_dojo.cli.skills.warn_legacy_skills", lambda: None)
    monkeypatch.setattr(config, "require_migrated_config", lambda: None)
    monkeypatch.setattr(config, "get_dashboard_url", lambda: "https://saved.test")
    monkeypatch.setattr(config, "get_dashboard_proxy_auth", lambda _url: False)
    monkeypatch.setattr(config, "get_dashboard_trajectory_viewer", lambda: None)
    monkeypatch.setattr(config, "get_dashboard_password", lambda: "")
    monkeypatch.setattr(config, "save_dashboard_url", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_setup, "deployed_dashboard_url", lambda: None)
    monkeypatch.setattr(cli_setup, "ensure_proxy_auth", lambda **_k: False)
    ensure = Mock(return_value="existing")
    monkeypatch.setattr(cli_setup, "ensure_dashboard_password", ensure)
    dashboard = SimpleNamespace(
        app=Mock(),
        fastapi_app=Mock(get_web_url=lambda: "https://dash.test"),
        ensure_creds_secret=lambda **_k: True,
        MODAL_CREDS_SECRET_NAME="creds",
    )
    monkeypatch.setattr(
        cli_setup, "_load_dashboard_for_deploy", lambda *_a, **_k: dashboard
    )
    return ensure


def test_stale_saved_url_is_not_an_existing_dashboard(stub_deploy):
    cli_setup.setup(interactive=False)

    assert stub_deploy.call_args.kwargs["existing_dashboard"] is False


@pytest.mark.parametrize(
    ("state", "protected", "label"),
    [
        ("created", None, " (password protected)"),
        ("missing", None, " (no password)"),
        ("existing", None, ""),
        ("existing", False, " (no password)"),
    ],
)
def test_deploy_line_says_whether_a_password_is_on(
    stub_deploy, capsys, state, protected, label
):
    stub_deploy.return_value = state

    cli_setup.setup(interactive=False, password_protected=protected)

    assert f"Dashboard deployed: https://dash.test{label}\n" in capsys.readouterr().out
