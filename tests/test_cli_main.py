from __future__ import annotations

import json
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from modal_dojo import cli as cli_module
from modal_dojo.cli.errors import CLIError, ExitCode
from modal_dojo.common.trackio import TrackioLookupUnknown


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_root_help_lists_existing_commands_by_panel(runner):
    result = runner.invoke(cli_module.entrypoint_cli, ["--help"])

    assert result.exit_code == 0
    assert "Configuration:" in result.stdout
    assert "Utilities:" in result.stdout
    assert "Training runs:" in result.stdout
    assert "Skills:" in result.stdout
    for command in (
        "run",
        "skills",
        "setup",
        "open",
        "set-password",
        "set-proxy-auth",
        "cleanup",
    ):
        assert command in result.stdout


def test_root_supports_short_help(runner):
    result = runner.invoke(cli_module.entrypoint_cli, ["-h"])

    assert result.exit_code == 0
    assert result.stdout.startswith("Usage:")


def test_root_without_command_shows_help(runner):
    result = runner.invoke(cli_module.entrypoint_cli, [])

    assert result.exit_code == 2
    assert "Usage:" in result.stderr


def test_main_returns_help_exit_code(capsys):
    assert cli_module.main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "Usage:" in captured.out
    assert captured.err == ""


def test_main_returns_no_command_exit_code(capsys):
    assert cli_module.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Usage:" in captured.err


def test_setup_dispatches_to_existing_function(runner, monkeypatch):
    setup = Mock()
    monkeypatch.setattr("modal_dojo.cli.setup.setup", setup)
    monkeypatch.setattr(
        "modal_dojo.common.config.get_dashboard_proxy_auth", lambda: False
    )

    result = runner.invoke(cli_module.entrypoint_cli, ["setup"])

    assert result.exit_code == 0
    assert result.stderr == ""
    setup.assert_called_once_with(require_proxy_auth=False)


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("--proxy-auth", True),
        ("--no-proxy-auth", False),
    ],
)
def test_setup_preserves_proxy_auth_choice(runner, monkeypatch, flag, expected):
    setup = Mock()
    monkeypatch.setattr("modal_dojo.cli.setup.setup", setup)

    result = runner.invoke(cli_module.entrypoint_cli, ["setup", flag])

    assert result.exit_code == 0
    setup.assert_called_once_with(require_proxy_auth=expected)


def test_setup_rejects_both_proxy_auth_flags(runner):
    result = runner.invoke(
        cli_module.entrypoint_cli,
        ["setup", "--proxy-auth", "--no-proxy-auth"],
    )

    assert result.exit_code == 2
    assert "cannot be used together" in result.stderr


def test_setup_prompts_for_explicit_choice_after_authenticated_deploy(
    runner, monkeypatch
):
    setup = Mock()
    monkeypatch.setattr("modal_dojo.cli.setup.setup", setup)
    monkeypatch.setattr(
        "modal_dojo.common.config.get_dashboard_proxy_auth", lambda: True
    )

    result = runner.invoke(cli_module.entrypoint_cli, ["setup"])

    assert result.exit_code == 2
    assert "--proxy-auth or --no-proxy-auth" in result.stderr
    setup.assert_not_called()


def test_open_dispatches_to_existing_function(runner, monkeypatch):
    open_dashboard = Mock()
    monkeypatch.setattr("modal_dojo.cli.setup.open_dashboard", open_dashboard)

    result = runner.invoke(cli_module.entrypoint_cli, ["open"])

    assert result.exit_code == 0
    open_dashboard.assert_called_once_with()


def test_set_proxy_auth_dispatches_to_existing_function(runner, monkeypatch):
    set_proxy_auth = Mock()
    monkeypatch.setattr("modal_dojo.cli.setup.set_proxy_auth", set_proxy_auth)

    result = runner.invoke(cli_module.entrypoint_cli, ["set-proxy-auth"])

    assert result.exit_code == 0
    set_proxy_auth.assert_called_once_with()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["set-password"], None),
        (["set-password", "--password", "secret"], "secret"),
        (["set-password", "--password", ""], ""),
    ],
)
def test_set_password_preserves_arguments(runner, monkeypatch, args, expected):
    set_password = Mock()
    monkeypatch.setattr("modal_dojo.cli.setup.set_password", set_password)

    result = runner.invoke(cli_module.entrypoint_cli, args)

    assert result.exit_code == 0
    set_password.assert_called_once_with(password=expected)


@pytest.mark.parametrize(
    ("lookup", "expect_old_password", "expect_could_not_check"),
    [
        ("https://example--training-gym-trackio.modal.run", True, False),
        (None, False, False),
        (TrackioLookupUnknown("timeout"), False, True),
    ],
)
def test_set_password_warns_about_stale_trackio_password(
    monkeypatch, capsys, lookup, expect_old_password, expect_could_not_check
):
    from modal_dojo.cli.setup import set_password

    monkeypatch.setattr("modal_dojo._dashboard.set_dashboard_password", Mock())
    monkeypatch.setattr("modal_dojo.cli.setup.setup", Mock())
    monkeypatch.setattr(
        "modal_dojo.common.config.get_dashboard_proxy_auth",
        lambda: False,
    )
    if isinstance(lookup, Exception):
        monkeypatch.setattr(
            "modal_dojo.common.trackio.lookup_trackio_url",
            Mock(side_effect=lookup),
        )
    else:
        monkeypatch.setattr(
            "modal_dojo.common.trackio.lookup_trackio_url",
            lambda app_name="training-gym-trackio": lookup,
        )

    set_password(password="secret")

    err = " ".join(capsys.readouterr().err.split())
    if expect_old_password:
        assert "Warning:" in err
        assert (
            "A Trackio dashboard is deployed on Modal and still has the old password."
            in err
        )
        assert "TrackioConfig.deploy_to_modal()" in err
        assert "Could not check" not in err
    elif expect_could_not_check:
        assert "Warning:" in err
        assert "Could not check whether a Trackio dashboard is deployed" in err
        assert "TrackioConfig.deploy_to_modal()" in err
        assert "still has the old password" not in err
    else:
        assert "Warning:" not in err


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["cleanup"], {"older_than_days": 7, "dry_run": False}),
        (
            ["cleanup", "--older-than-days", "3", "--dry-run"],
            {"older_than_days": 3, "dry_run": True},
        ),
    ],
)
def test_cleanup_preserves_arguments(runner, monkeypatch, args, expected):
    cleanup = Mock()
    monkeypatch.setattr("modal_dojo.cli.cleanup.cleanup", cleanup)

    result = runner.invoke(cli_module.entrypoint_cli, args)

    assert result.exit_code == 0
    cleanup.assert_called_once_with(**expected)


def test_click_usage_errors_use_exit_two_and_stderr(runner):
    result = runner.invoke(cli_module.entrypoint_cli, ["cleanup", "--unknown"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "No such option '--unknown'" in result.stderr


def test_expected_errors_use_declared_exit_code(runner, monkeypatch):
    def fail(**_kwargs):
        raise CLIError(
            "offline",
            error="dashboard_unreachable",
            exit_code=ExitCode.BACKEND,
            hint="modal-dojo open",
        )

    monkeypatch.setattr("modal_dojo.cli.setup.setup", fail)
    monkeypatch.setattr(
        "modal_dojo.common.config.get_dashboard_proxy_auth", lambda: False
    )

    result = runner.invoke(cli_module.entrypoint_cli, ["setup"])

    assert result.exit_code == ExitCode.BACKEND
    assert result.stdout == ""
    assert "offline" in result.stderr
    assert "modal-dojo open" in result.stderr


def test_main_renders_structured_json_errors(monkeypatch, capsys):
    def fail(**_kwargs):
        raise CLIError(
            "Run run_8f2a was not found.",
            error="run_not_found",
            exit_code=ExitCode.NOT_FOUND,
            hint="modal-dojo run list --since 7d",
            run_id="run_8f2a",
        )

    monkeypatch.setattr(cli_module.entrypoint_cli, "main", fail)

    assert cli_module.main(["run", "show", "run_8f2a", "--json"]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": "run_not_found",
        "run_id": "run_8f2a",
        "message": "Run run_8f2a was not found.",
        "hint": "modal-dojo run list --since 7d",
    }


def test_main_renders_usage_errors_as_json(capsys):
    assert cli_module.main(["cleanup", "--unknown", "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "usage_error"
    assert "No such option" in payload["message"]


def test_main_maps_unexpected_errors(monkeypatch, capsys):
    def fail(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_module.entrypoint_cli, "main", fail)

    assert cli_module.main(["setup"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Error: boom\n"


def test_main_maps_click_keyboard_interrupt_to_130(monkeypatch, capsys):
    def interrupt(**_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "modal_dojo.cli.setup.setup",
        interrupt,
    )
    monkeypatch.setattr(
        "modal_dojo.common.config.get_dashboard_proxy_auth", lambda: False
    )

    assert cli_module.main(["setup"]) == 130
    captured = capsys.readouterr()
    assert "Interrupted." in captured.err
    assert "Aborted!" not in captured.err
