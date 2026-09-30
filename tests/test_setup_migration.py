from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import re

from modal_dojo.cli.setup import ProxyAuthMode

from modal_dojo.common import config
from modal_dojo.common.dashboard import (
    DashboardLookupUnknown,
    LEGACY_DASHBOARD_APP_NAME,
)
from modal_dojo.cli import setup as cli_setup_module


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / ".modal-dojo.toml")
    monkeypatch.setattr(config, "LEGACY_CONFIG_PATH", tmp_path / ".training-gym.toml")
    monkeypatch.setattr(config, "_legacy_warning_printed", False)
    return config.CONFIG_PATH, config.LEGACY_CONFIG_PATH


def test_read_warns_once_without_writing(paths, capsys):
    new, old = paths
    old.write_text('[dashboard]\nurl="https://old.test"\n')
    assert config.load_config() == config.load_config()
    output = capsys.readouterr()
    assert not output.out
    assert output.err.count("Warning:") == 1
    assert "modal-dojo setup" in output.err
    assert not new.exists()


def test_copy_verbatim_and_precedence(paths, capsys):
    new, old = paths
    raw = b'# my settings\n[dashboard]\nurl="https://old.test"\n'
    old.write_bytes(raw)
    assert config.migrate_config()
    assert new.read_bytes() == old.read_bytes() == raw
    assert new.stat().st_mode & 0o777 == 0o600
    assert "Note:" in capsys.readouterr().err
    old.write_text("invalid toml")
    assert not config.migrate_config()
    config.save_dashboard_url("https://new.test")
    assert config.get_dashboard_url() == "https://new.test"
    assert old.read_text() == "invalid toml"
    assert new.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("target", [0, 1])
def test_malformed_config_is_not_overwritten(paths, target):
    paths[target].write_text("invalid toml")
    with pytest.raises(ValueError, match="Cannot read configuration"):
        config.migrate_config()
    assert paths[target].read_text() == "invalid toml"
    if target == 0:
        assert config.load_config() == {}
    else:
        assert not paths[0].exists()


def test_unreadable_config_is_not_overwritten(paths, monkeypatch):
    paths[1].write_text("[dashboard]\n")
    original = type(paths[1]).open

    def denied(path, *args, **kwargs):
        if path == paths[1]:
            raise PermissionError("denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(paths[1]), "open", denied)
    with pytest.raises(ValueError, match="Cannot read configuration"):
        config.migrate_config()
    assert not paths[0].exists()


@pytest.fixture
def deployment(paths, monkeypatch):
    import modal

    seen = []
    dashboard = SimpleNamespace(
        ensure_creds_secret=lambda **kwargs: True,
        app=SimpleNamespace(deploy=lambda: None),
        fastapi_app=SimpleNamespace(get_web_url=lambda: "https://new.test"),
    )

    def load(auth, trajectory_viewer=None):
        seen.append((auth, trajectory_viewer))
        return dashboard

    monkeypatch.setattr(cli_setup_module, "_load_dashboard_for_deploy", load)
    monkeypatch.setattr(cli_setup_module, "ensure_proxy_auth", lambda **kwargs: True)
    monkeypatch.setattr(modal, "enable_output", nullcontext)
    monkeypatch.setattr(cli_setup_module, "deployed_dashboard_url", lambda *args: None)
    monkeypatch.setattr(config, "get_dashboard_proxy_auth", lambda url: None)
    return dashboard, seen


@pytest.mark.parametrize("auth", [True, False])
def test_migration_inherits_settings_and_prefers_new_app(
    paths, deployment, monkeypatch, capsys, auth
):
    dashboard, seen = deployment
    paths[1].write_text(
        '[dashboard]\nurl="https://old.test"\ntrajectory_viewer="/my/viewer.svelte"\n[proxy_auth]\nkey="wk-test"\nsecret="ws-test"\n'
    )
    monkeypatch.setattr(
        cli_setup_module,
        "deployed_dashboard_url",
        lambda *args: "https://old.test" if args else "https://new.test",
    )
    probed = []

    def mode(url):
        probed.append(url)
        return auth

    monkeypatch.setattr(config, "get_dashboard_proxy_auth", mode)
    cli_setup_module.setup(interactive=False)
    assert probed == ["https://new.test"]
    assert seen == [(auth, "/my/viewer.svelte")]
    assert config.get_proxy_auth() == ("wk-test", "ws-test")
    assert (
        re.search(
            r"modal[ \n]app[ \n]stop[ \n]training-gym-dashboard",
            capsys.readouterr().err,
        )
        is not None
    )
    assert config.get_dashboard_url() == "https://new.test"


def test_failed_deploy_keeps_old_url_then_retry(paths, deployment, monkeypatch):
    dashboard, seen = deployment
    paths[1].write_text('[dashboard]\nurl="https://old.test"\n')

    def fail():
        raise RuntimeError("deployment failed")

    dashboard.app.deploy = fail
    with pytest.raises(RuntimeError):
        cli_setup_module.setup(proxy_auth=ProxyAuthMode.REQUIRE, interactive=False)
    assert config.get_dashboard_url() == "https://old.test"
    dashboard.app.deploy = lambda: None
    cli_setup_module.setup(proxy_auth=ProxyAuthMode.DISABLE, interactive=False)
    assert config.get_dashboard_url() == "https://new.test"
    assert seen == [(True, None), (False, None)]


def test_unknown_auth_requires_explicit_flag(paths, deployment):
    paths[1].write_text('[dashboard]\nurl="https://old.test"\n')
    with pytest.raises(ValueError, match="--proxy-auth"):
        cli_setup_module.setup(interactive=False)
    assert not deployment[1]


def test_legacy_app_without_config_is_detected(paths, deployment, monkeypatch):
    calls = []

    def lookup(*args):
        calls.append(args)
        return "https://old.test" if args else None

    monkeypatch.setattr(cli_setup_module, "deployed_dashboard_url", lookup)
    monkeypatch.setattr(config, "get_dashboard_proxy_auth", lambda url: True)
    cli_setup_module.setup(interactive=False)
    assert (LEGACY_DASHBOARD_APP_NAME,) in calls
    assert deployment[1] == [(True, None)]


def test_fresh_install_defaults_open(paths, deployment):
    cli_setup_module.setup(interactive=False)
    assert deployment[1] == [(False, None)]


def test_unknown_discovery_never_deploys(paths, deployment, monkeypatch):
    def unknown(*args):
        raise DashboardLookupUnknown()

    monkeypatch.setattr(cli_setup_module, "deployed_dashboard_url", unknown)
    with pytest.raises(ValueError, match="discover"):
        cli_setup_module.setup(interactive=False)
    assert not deployment[1]


def test_metrics_defaults_and_persistent_identifiers():
    from modal_dojo.common.metric_mirror import DashboardMetricConfig
    from modal_dojo.common.dashboard_components import DASHBOARD_OVERLAY_VOLUME_NAME
    from modal_dojo.utils.metadata import METADATA_VOLUME_NAME
    from modal_dojo.common.dashboard import DASHBOARD_APP_NAME, DASHBOARD_VERSION

    assert DashboardMetricConfig().project == "modal-dojo"
    assert DashboardMetricConfig(project="training-gym").project == "training-gym"
    assert METADATA_VOLUME_NAME == "training-gym-metadata"
    assert DASHBOARD_OVERLAY_VOLUME_NAME == "training-gym-dashboard-overlay"
    assert config.DASHBOARD_PASSWORD_SECRET_NAME == "_training-gym-dashboard-password"
    assert DASHBOARD_APP_NAME == "dojo-dashboard"
    assert DASHBOARD_VERSION == 6


@pytest.mark.parametrize(
    ("flag", "expected"),
    [(None, None), ("--proxy-auth", True), ("--no-proxy-auth", False)],
)
def test_cli_requires_auth_choice_when_existing_mode_unknown(
    paths, deployment, flag, expected
):
    from click.testing import CliRunner
    from modal_dojo.cli import entrypoint_cli

    paths[1].write_text('[dashboard]\nurl="https://old.test"\n')
    result = CliRunner().invoke(entrypoint_cli, ["setup", *([flag] if flag else [])])
    if flag is None:
        assert result.exit_code != 0
        assert "--proxy-auth or --no-proxy-auth" in str(result.exception)
        assert not deployment[1]
        assert config.get_dashboard_url() == "https://old.test"
    else:
        assert result.exit_code == 0, result.exception
        assert deployment[1] == [(expected, None)]


def test_auth_lookup_uses_only_explicit_url(paths, monkeypatch):
    config.save_dashboard_url("https://cached.test", proxy_auth=True)
    requested = []

    def unavailable(request, **kwargs):
        from urllib.error import URLError

        requested.append(request.full_url)
        raise URLError("offline")

    monkeypatch.setattr(config, "urlopen", unavailable)
    assert config.get_dashboard_proxy_auth("https://explicit.test") is True
    assert requested == ["https://explicit.test/api/proxy-auth"]
    requested.clear()
    assert config.get_dashboard_proxy_auth(None) is True
    assert not requested
    with pytest.raises(TypeError):
        config.get_dashboard_proxy_auth()


def test_setup_warns_about_old_skills_without_modifying_them(
    paths, deployment, monkeypatch, capsys
):
    root = paths[0].parent
    (root / ".git").mkdir()
    old = root / ".agents/skills/training-gym-overview"
    old.mkdir(parents=True)
    (old / "SKILL.md").write_text("old instructions")
    nested = root / "src"
    nested.mkdir()
    monkeypatch.chdir(nested)
    cli_setup_module.setup(interactive=False)
    assert "modal-dojo skills install --force" in capsys.readouterr().err
    assert (old / "SKILL.md").read_text() == "old instructions"
    assert not (root / ".agents/skills/modal-dojo-overview").exists()


def test_current_dashboard_does_not_require_legacy_lookup(
    paths, deployment, monkeypatch
):
    looked_up = []

    def lookup(*args):
        looked_up.append(args)
        if args:
            raise DashboardLookupUnknown()
        return "https://new.test"

    monkeypatch.setattr(cli_setup_module, "deployed_dashboard_url", lookup)
    monkeypatch.setattr(config, "get_dashboard_proxy_auth", lambda url: True)
    assert cli_setup_module.setup(interactive=False) == "https://new.test"
    assert looked_up == [()]
    assert deployment[1] == [(True, None)]


def test_unknown_legacy_lookup_without_current_dashboard_blocks_deploy(
    paths, deployment, monkeypatch
):
    def lookup(*args):
        if args:
            raise DashboardLookupUnknown()
        return None

    monkeypatch.setattr(cli_setup_module, "deployed_dashboard_url", lookup)
    with pytest.raises(ValueError, match="discover"):
        cli_setup_module.setup(interactive=False)
    assert not deployment[1]
