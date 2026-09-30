from types import SimpleNamespace

import pytest

from modal_dojo.cli import migrate as migration
from modal_dojo.cli import setup as cli_setup_module
from modal_dojo.cli.setup import ProxyAuthMode
from modal_dojo.cli.errors import CLIError
from modal_dojo.common import config
from modal_dojo.common.errors import DojoConfigError


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / ".modal-dojo.toml")
    monkeypatch.setattr(config, "LEGACY_CONFIG_PATH", tmp_path / ".training-gym.toml")
    return config.CONFIG_PATH, config.LEGACY_CONFIG_PATH


class Resources:
    def __init__(self):
        self.volumes = {
            old: SimpleNamespace(object_id=old) for old, _ in migration.VOLUME_PAIRS
        }
        self.app_list = [
            migration.AppInfo(
                "dashboard", migration.LEGACY_DASHBOARD_APP_NAME, True, {}
            ),
            migration.AppInfo("trackio", "training-gym-trackio", True, {}),
        ]
        self.events = []
        self.rows = []
        self.liveness = False
        self.fail_rename = None

    def volume(self, name):
        return self.volumes.get(name)

    def apps(self, names):
        return [app for app in self.app_list if app.name in names]

    def records(self, volume):
        return self.rows

    def live(self, app_id):
        return self.liveness

    def stop(self, app_id):
        self.events.append(("stop", app_id))
        for app in self.app_list:
            if app.id == app_id:
                app.live = False

    def rename(self, old, new):
        if old == self.fail_rename:
            raise RuntimeError("rename failed")
        self.events.append(("rename", old))
        self.volumes[new] = self.volumes.pop(old)


@pytest.fixture
def harness(paths, monkeypatch):
    resources = Resources()
    paths[1].write_text(
        '# settings\n[dashboard]\nurl="https://old.test"\nproxy_auth=true\n[proxy_auth]\nkey="wk-test"\nsecret="ws-test"\n'
    )
    monkeypatch.setattr(
        migration, "deployed_dashboard_url", lambda *args: "https://old.test"
    )
    monkeypatch.setattr(config, "get_dashboard_proxy_auth", lambda url, **kwargs: True)
    monkeypatch.setattr(
        migration.TrackioConfig,
        "deploy_to_modal",
        lambda **kwargs: SimpleNamespace(server_url="https://trackio.test"),
    )

    def setup(**kwargs):
        resources.events.append(("setup", kwargs["proxy_auth"]))
        config.save_dashboard_url("https://new.test")
        return "https://new.test"

    monkeypatch.setattr(migration, "setup", setup)
    return resources


def test_loader_ignores_legacy_and_guards_setup(paths, harness):
    assert config.load_config() == {}
    with pytest.raises(DojoConfigError, match="modal-dojo migrate"):
        cli_setup_module.setup()
    assert not paths[0].exists()


@pytest.mark.parametrize("method", ["train", "launch"])
def test_training_guard_precedes_launch_work(paths, harness, method):
    from modal_dojo.common.train import TrainConfig

    with pytest.raises(DojoConfigError, match="modal-dojo migrate"):
        getattr(TrainConfig, method)(object.__new__(TrainConfig))


def test_migrate_moves_config_and_preserves_volume_ids(paths, harness):
    migration.migrate(resources=harness)
    assert paths[0].exists() and not paths[1].exists()
    assert paths[0].stat().st_mode & 0o777 == 0o600
    assert config.get_proxy_auth() == ("wk-test", "ws-test")
    for old, new in migration.VOLUME_PAIRS:
        assert harness.volumes[new].object_id == old
    assert harness.events[:2] == [("stop", "dashboard"), ("stop", "trackio")]
    assert harness.events[-1] == ("setup", ProxyAuthMode.REQUIRE)
    migration.migrate(resources=harness)


def test_conflicts_precede_service_shutdown(paths, harness):
    harness.volumes[migration.METADATA_VOLUME_NAME] = SimpleNamespace(object_id="other")
    with pytest.raises(CLIError, match="Both"):
        migration.migrate(resources=harness)
    assert not harness.events and paths[1].exists()


@pytest.mark.parametrize(
    "status",
    ["running", "initializing", "deploying_model", "running_eval", "unknown", None],
)
def test_active_or_unknown_metadata_blocks(paths, harness, status):
    harness.rows = [
        ("training-runs/run.json", {"status": status, "modal_app_id": "ap-run"})
    ]
    with pytest.raises(CLIError, match="ap-run"):
        migration.migrate(resources=harness)
    assert not harness.events


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "stopped"])
def test_terminal_metadata_never_checks_app_liveness(
    paths, harness, monkeypatch, status
):
    harness.rows = [
        ("training-runs/run.json", {"status": status, "modal_app_id": "ap-run"})
    ]

    def forbidden(*args):
        pytest.fail("Run checks must not query app liveness")

    monkeypatch.setattr(harness, "live", forbidden)
    migration.migrate(resources=harness)


@pytest.mark.parametrize("folder", ["evals", "eval-results"])
def test_evaluation_metadata_is_ignored(paths, harness, folder):
    harness.rows = [(f"{folder}/eval.json", {"status": "running_eval"})]
    migration.migrate(resources=harness)


def test_partial_rename_retry(paths, harness):
    harness.fail_rename = migration.VOLUME_PAIRS[1][0]
    with pytest.raises(CLIError, match="volume rename"):
        migration.migrate(resources=harness)
    assert paths[1].exists() and not paths[0].exists()
    harness.fail_rename = None
    migration.migrate(resources=harness)


def test_stop_failure_blocks_renames(paths, harness, monkeypatch):
    def fail(app_id):
        raise RuntimeError("stop failed")

    monkeypatch.setattr(harness, "stop", fail)
    with pytest.raises(CLIError, match="service shutdown"):
        migration.migrate(resources=harness)
    assert all(old in harness.volumes for old, _ in migration.VOLUME_PAIRS)


def test_both_configs_preserve_new_and_archive_old(paths, harness):
    paths[0].write_text(
        '[dashboard]\nurl="https://chosen.test"\n[proxy_auth]\nkey="new-key"\nsecret="new-secret"\n'
    )
    old_bytes = paths[1].read_bytes()
    migration.migrate(resources=harness)
    assert config.get_proxy_auth() == ("new-key", "new-secret")
    assert not paths[1].exists()
    assert next(paths[0].parent.glob("*.bak")).read_bytes() == old_bytes


@pytest.mark.parametrize("mode", list(ProxyAuthMode))
def test_auth_resolution_precedes_shutdown(paths, harness, monkeypatch, mode):
    def probe(url, **kwargs):
        assert not harness.events
        assert kwargs["headers"] == {"Modal-Key": "wk-test", "Modal-Secret": "ws-test"}
        return None

    monkeypatch.setattr(config, "get_dashboard_proxy_auth", probe)
    if mode is ProxyAuthMode.UNSPECIFIED:
        with pytest.raises(CLIError, match="--proxy-auth or --no-proxy-auth"):
            migration.migrate(resources=harness)
        assert not harness.events
    else:
        migration.migrate(resources=harness, proxy_auth=mode)
        assert harness.events[-1] == ("setup", mode)


def test_failed_deploy_after_move_is_retryable(paths, harness, monkeypatch):
    original = migration.setup

    def fail(**kwargs):
        raise RuntimeError("deployment failed")

    monkeypatch.setattr(migration, "setup", fail)
    with pytest.raises(CLIError, match="dashboard deployment"):
        migration.migrate(resources=harness)
    assert paths[0].exists() and not paths[1].exists()
    assert config.get_dashboard_url() == "https://old.test"
    monkeypatch.setattr(migration, "setup", original)
    migration.migrate(resources=harness)


def test_no_config_does_not_inspect_resources(paths):
    with pytest.raises(CLIError, match="modal-dojo setup"):
        migration.migrate(resources=object())


def test_invalid_config_does_not_mutate(paths, harness):
    paths[1].write_text("invalid toml")
    with pytest.raises(CLIError, match="configuration inspection"):
        migration.migrate(resources=harness)
    assert not harness.events


def test_app_discovery_failure_prevents_mutation(paths, harness, monkeypatch):
    def fail(names):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(harness, "apps", fail)
    with pytest.raises(CLIError, match="service discovery"):
        migration.migrate(resources=harness)
    assert not harness.events


@pytest.mark.parametrize(
    ("app_name", "volume_name", "expected"),
    [
        ("modal-dojo-trackio", "", "modal-dojo-trackio-data"),
        ("custom-trackio", "", "custom-trackio-data"),
        ("modal-dojo-trackio", "custom-data", "custom-data"),
    ],
)
def test_trackio_volume_defaults(monkeypatch, app_name, volume_name, expected):
    import modal_dojo.common.trackio as trackio

    calls = []

    def deploy(**kwargs):
        calls.append(kwargs)
        return "https://trackio.test"

    monkeypatch.setattr(trackio, "_deploy_modal_dashboard", deploy)
    trackio.TrackioConfig.deploy_to_modal(app_name=app_name, volume_name=volume_name)
    assert calls[0]["volume_name"] == expected


def test_cli_migrate_flags(monkeypatch):
    from click.testing import CliRunner
    from modal_dojo.cli import entrypoint_cli

    calls = []
    monkeypatch.setattr(migration, "migrate", lambda **kwargs: calls.append(kwargs))
    for flag, mode in [
        ([], ProxyAuthMode.UNSPECIFIED),
        (["--proxy-auth"], ProxyAuthMode.REQUIRE),
        (["--no-proxy-auth"], ProxyAuthMode.DISABLE),
    ]:
        result = CliRunner().invoke(entrypoint_cli, ["migrate", *flag], input="yes\n")
        assert result.exit_code == 0, result.exception
        assert calls[-1] == {"proxy_auth": mode, "force": False}
    assert (
        CliRunner()
        .invoke(entrypoint_cli, ["migrate", "--proxy-auth", "--no-proxy-auth"])
        .exit_code
        == 2
    )


def test_shutdown_waits_for_containers(monkeypatch):
    calls = []
    outputs = iter(['[{"Container ID": "ta-running"}]', "[]"])

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=next(outputs)) if "list" in command else None

    monkeypatch.setattr(migration.subprocess, "run", run)
    monkeypatch.setattr(migration.time, "sleep", lambda seconds: None)
    migration.ModalResources().stop("ap-service")
    assert calls[0][0] == [
        migration.sys.executable,
        "-m",
        "modal",
        "app",
        "stop",
        "ap-service",
        "--yes",
    ]
    assert [call[0][3:] for call in calls[1:]] == [
        ["container", "list", "--app-id", "ap-service", "--json"],
        ["container", "list", "--app-id", "ap-service", "--json"],
    ]
    assert all(kwargs["check"] for _, kwargs in calls)


@pytest.mark.parametrize("failure", ["stop", "list", "invalid-json"])
def test_shutdown_errors_propagate(monkeypatch, failure):
    def run(command, **kwargs):
        if (failure == "stop" and "stop" in command) or (
            failure == "list" and "list" in command
        ):
            raise migration.subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout="invalid")

    monkeypatch.setattr(migration.subprocess, "run", run)
    with pytest.raises((migration.subprocess.CalledProcessError, ValueError)):
        migration.ModalResources().stop("ap-service")


def test_migration_stops_both_trackio_names(paths, harness):
    harness.app_list.append(
        migration.AppInfo("new-trackio", "modal-dojo-trackio", True, {})
    )
    migration.migrate(resources=harness)
    assert ("stop", "trackio") in harness.events
    assert ("stop", "new-trackio") in harness.events


def test_default_trackio_deploy_uses_new_names(monkeypatch):
    import modal_dojo.common.trackio as trackio

    calls = []

    def deploy(**kwargs):
        calls.append(kwargs)
        return "https://new-trackio.test"

    monkeypatch.setattr(trackio, "_deploy_modal_dashboard", deploy)
    result = trackio.TrackioConfig.deploy_to_modal()
    assert calls[0]["app_name"] == "modal-dojo-trackio"
    assert calls[0]["volume_name"] == "modal-dojo-trackio-data"
    assert result.modal_secret_name == "_modal-dojo-trackio-write-token"


def test_migrate_warns_about_old_skills_without_changing_files(
    paths, harness, monkeypatch, capsys
):
    root = paths[0].parent
    (root / ".git").mkdir()
    old = root / ".agents/skills/training-gym-overview"
    old.mkdir(parents=True)
    (old / "SKILL.md").write_text("existing instructions")
    link = root / ".claude/skills/training-gym-overview"
    link.parent.mkdir(parents=True)
    link.symlink_to(old, target_is_directory=True)
    nested = root / "src"
    nested.mkdir()
    monkeypatch.chdir(nested)

    migration.migrate(resources=harness)

    assert "modal-dojo skills install --force" in capsys.readouterr().err
    assert (old / "SKILL.md").read_text() == "existing instructions"
    assert link.is_symlink() and link.resolve() == old
    assert not (old.parent / "modal-dojo-overview").exists()


def test_service_discovery_uses_public_lookup(monkeypatch):
    import modal
    from modal.exception import NotFoundError

    backend = migration.ModalResources()
    calls = []

    def lookup(name, *, create_if_missing):
        calls.append((name, create_if_missing))
        if name == "missing":
            raise NotFoundError("missing")
        return SimpleNamespace(app_id="ap-service")

    monkeypatch.setattr(modal.App, "lookup", lookup)
    assert not hasattr(backend, "rpc")
    apps = backend.apps({"present", "missing"})
    assert [(app.id, app.name) for app in apps] == [("ap-service", "present")]
    assert calls == [("missing", False), ("present", False)]


def test_run_check_uses_only_volume_records():
    resources = SimpleNamespace(
        records=lambda volume: [
            (
                "training-runs/run.json",
                {"status": "completed", "modal_app_id": "ap-old"},
            ),
            ("evals/historical.json", {}),
        ]
    )
    migration._check_runs(resources, [object()])


def test_metadata_read_failure_blocks_migration(paths, harness, monkeypatch):
    def fail(volume):
        raise OSError("metadata unavailable")

    monkeypatch.setattr(harness, "records", fail)
    with pytest.raises(CLIError, match="metadata unavailable"):
        migration.migrate(resources=harness)
    assert not harness.events


@pytest.mark.parametrize("wrapped", [False, True])
def test_training_summary_read_does_not_iterate_volume(wrapped):
    import json

    rows = [{"training_run_id": "run-a", "status": "running"}]
    payload = {"items": rows} if wrapped else rows
    reads = []

    def read_file(path):
        reads.append(path)
        return [json.dumps(payload).encode()]

    volume = SimpleNamespace(read_file=read_file)
    assert list(migration.ModalResources().records(volume)) == [
        ("training-runs/run-a", rows[0])
    ]
    assert reads == ["training-runs-summary/summary.json"]


@pytest.mark.parametrize(
    "payload", [b"bad json", b"{}", b'{"items": null}', b'{"items": [42]}']
)
def test_invalid_summary_blocks_without_iteration(payload):
    volume = SimpleNamespace(read_file=lambda path: [payload])
    with pytest.raises(
        ValueError,
        match="Manually check that no runs are active, then run with --force",
    ):
        list(migration.ModalResources().records(volume))


def test_missing_summary_blocks_without_iteration():
    def missing(path):
        raise FileNotFoundError(path)

    with pytest.raises(
        ValueError,
        match="Manually check that no runs are active, then run with --force",
    ):
        list(migration.ModalResources().records(SimpleNamespace(read_file=missing)))


def test_empty_summary_has_no_live_runs():
    volume = SimpleNamespace(read_file=lambda path: [b'{"items": []}'])
    assert list(migration.ModalResources().records(volume)) == []


def test_force_skips_all_run_checks(paths, harness, monkeypatch, capsys):
    def forbidden(*args):
        pytest.fail("Forced migration must not read run metadata")

    monkeypatch.setattr(harness, "records", forbidden)
    migration.migrate(resources=harness, force=True)
    assert "Skipping the live-run check" in capsys.readouterr().out


@pytest.mark.parametrize("unreadable", [True, False])
def test_live_check_failure_explains_force(paths, harness, monkeypatch, unreadable):
    if unreadable:

        def fail(volume):
            raise ValueError("summary unavailable")

        monkeypatch.setattr(harness, "records", fail)
    else:
        harness.rows = [("training-runs/run.json", {"status": "running"})]
    with pytest.raises(CLIError) as error:
        migration.migrate(resources=harness)
    assert "After verifying there are no active training runs" in error.value.hint
    assert "modal-dojo migrate --force" in error.value.hint
    assert not harness.events


def test_force_does_not_skip_volume_conflicts(paths, harness):
    harness.volumes[migration.METADATA_VOLUME_NAME] = SimpleNamespace(object_id="other")
    with pytest.raises(CLIError, match="Both") as error:
        migration.migrate(resources=harness, force=True)
    assert "--force" not in error.value.hint
    assert not harness.events


def test_cli_forwards_force(monkeypatch):
    from click.testing import CliRunner
    from modal_dojo.cli import entrypoint_cli

    calls = []
    monkeypatch.setattr(migration, "migrate", lambda **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(
        entrypoint_cli, ["migrate", "--force", "--proxy-auth"], input="y\n"
    )
    assert result.exit_code == 0, result.exception
    assert calls == [{"proxy_auth": ProxyAuthMode.REQUIRE, "force": True}]


@pytest.mark.parametrize("answer", ["y", "Y", "n", "N"])
def test_migration_confirmation(monkeypatch, answer):
    from click.testing import CliRunner
    from modal_dojo.cli import entrypoint_cli

    calls = []
    monkeypatch.setattr(migration, "migrate", lambda **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(
        entrypoint_cli, ["migrate", "--force"], input=answer + "\n"
    )
    assert result.exit_code == 0
    assert "no active runs on the Training Gym" in result.output
    assert bool(calls) is (answer.lower() == "y")


@pytest.mark.parametrize("answer", ["maybe", "true", "1", ""])
def test_invalid_confirmation_reprompts(monkeypatch, answer):
    from click.testing import CliRunner
    from modal_dojo.cli import entrypoint_cli

    calls = []
    monkeypatch.setattr(migration, "migrate", lambda **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(entrypoint_cli, ["migrate"], input=answer + "\nno\n")
    assert result.exit_code == 0
    assert not calls


def test_confirmation_eof_does_not_migrate(monkeypatch):
    from click.testing import CliRunner
    from modal_dojo.cli import entrypoint_cli

    calls = []
    monkeypatch.setattr(migration, "migrate", lambda **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(entrypoint_cli, ["migrate"], input="")
    assert result.exit_code != 0
    assert not calls
