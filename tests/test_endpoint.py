"""Unit tests for the managed Modal Endpoint wrapper in ``common/endpoint.py``."""

from __future__ import annotations

import inspect
import subprocess
from types import SimpleNamespace
from typing import Any

import httpx
import modal
import pytest
from modal.cli.endpoint import create as modal_endpoint_create

from modal_dojo.common import endpoint as endpoint_module
from modal_dojo.common.checkpoint import Checkpoint, CheckpointType
from modal_dojo.common.endpoint import Endpoint
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.models import ModelConfig


class _FakeClock:
    """Deterministic stand-in for the ``time`` module used by endpoint.py.

    ``sleep()`` advances the monotonic clock instead of blocking, so polling
    loops and their timeouts resolve immediately and can be asserted on.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class _FakeResponse:
    def __init__(
        self, status_code: int = 200, message: dict[str, Any] | None = None
    ) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._message = message or {"role": "assistant", "content": "ok"}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://ws--ep.modal.run")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": self._message}]}


class _FakeModalCli:
    """Records ``modal endpoint create`` invocations and scripts ``get_url()``."""

    DEFAULT_URL = "https://ws--ep-test.modal.run"

    def __init__(
        self,
        urls: list[str | None | BaseException] | None,
        stop_returncode: int = 0,
        stop_error: BaseException | None = None,
    ) -> None:
        self.commands: list[list[str]] = []
        self.run_kwargs: list[dict[str, Any]] = []
        self.servers: list[tuple[str, str, str | None]] = []
        self._urls = urls
        self._stop_returncode = stop_returncode
        self._stop_error = stop_error
        self._stop_stdout = ""
        self._stop_stderr = ""
        self._create_returncode = 0
        self._create_stdout = ""

    def run(self, command: list[str], **kwargs: Any) -> SimpleNamespace:
        self.commands.append(list(command))
        self.run_kwargs.append(kwargs)
        if "stop" in command:
            if self._stop_error is not None:
                raise self._stop_error
            result = SimpleNamespace(
                returncode=self._stop_returncode,
                stdout=self._stop_stdout,
                stderr=self._stop_stderr,
            )
            if kwargs.get("check") and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    command,
                    output=result.stdout,
                    stderr=result.stderr,
                )
            return result
        return SimpleNamespace(
            returncode=self._create_returncode,
            stdout=self._create_stdout,
            stderr="",
        )

    def from_name(
        self, app_name: str, cls_name: str, environment_name: str | None = None
    ) -> SimpleNamespace:
        self.servers.append((app_name, cls_name, environment_name))
        return SimpleNamespace(get_url=self.get_url)

    def get_url(self) -> str | None:
        if self._urls is None:
            return self.DEFAULT_URL
        value = self._urls.pop(0) if self._urls else None
        if isinstance(value, BaseException):
            raise value
        return value

    @property
    def last_command(self) -> list[str]:
        return self.commands[-1]

    def flag_value(self, flag: str) -> str:
        return self.last_command[self.last_command.index(flag) + 1]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(endpoint_module, "time", fake)
    return fake


@pytest.fixture
def fake_modal_cli(monkeypatch: pytest.MonkeyPatch, clock: _FakeClock):
    def _install(
        urls: list[str | None | BaseException] | None = None,
        stop_returncode: int = 0,
        stop_error: BaseException | None = None,
    ) -> _FakeModalCli:
        cli = _FakeModalCli(
            urls, stop_returncode=stop_returncode, stop_error=stop_error
        )
        monkeypatch.setattr(endpoint_module.subprocess, "run", cli.run)
        monkeypatch.setattr(
            endpoint_module,
            "modal",
            SimpleNamespace(
                Server=SimpleNamespace(from_name=cli.from_name),
                exception=modal.exception,
            ),
        )
        return cli

    return _install


def _endpoint(
    *, requires_proxy_auth: bool = False, environment: str | None = None
) -> Endpoint:
    return Endpoint(
        "https://ws--ep.modal.run",
        endpoint_name="my-ft",
        model_name="model",
        requires_proxy_auth=requires_proxy_auth,
        environment=environment,
    )


@pytest.mark.parametrize(
    "model", ["Qwen/Qwen3-4B", ModelConfig(model_name="Qwen/Qwen3-4B")]
)
def test_launch_accepts_strings_and_model_configs(
    fake_modal_cli, model: ModelConfig | str
) -> None:
    cli = fake_modal_cli()

    endpoint = Endpoint.launch(model, unauthenticated=True)

    assert cli.flag_value("--model") == "Qwen/Qwen3-4B"
    assert endpoint.model_name == "Qwen/Qwen3-4B"


def test_launch_creates_a_public_endpoint(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    endpoint = Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=True)

    assert cli.last_command[1:5] == ["-m", "modal", "endpoint", "create"]
    assert cli.flag_value("--model") == "Qwen/Qwen3-4B"
    assert cli.flag_value("--name") == endpoint.endpoint_name
    assert "--unauthenticated" in cli.last_command
    assert "--env" not in cli.last_command
    assert "--routing-region" not in cli.last_command
    assert "--custom-volume-name" not in cli.last_command
    assert "--colocate-compute" not in cli.last_command
    assert "--custom-hf-repo" not in cli.last_command
    assert "--custom-hf-revision" not in cli.last_command
    assert "--custom-hf-token" not in cli.last_command
    assert cli.run_kwargs[-1] == {
        "check": False,
        "timeout": 120,
        "capture_output": True,
        "text": True,
    }
    assert endpoint.model_name == "Qwen/Qwen3-4B"
    assert endpoint.requires_proxy_auth is False


def test_launch_omits_the_unauthenticated_flag_for_proxy_auth(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    endpoint = Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=False)

    assert "--unauthenticated" not in cli.last_command
    assert endpoint.requires_proxy_auth is True


def test_launch_forwards_environment_and_routing_region(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    Endpoint.launch(
        "Qwen/Qwen3-4B",
        unauthenticated=True,
        environment="dev",
        routing_region="us-east",
    )

    assert cli.flag_value("--env") == "dev"
    assert cli.flag_value("--routing-region") == "us-east"
    assert cli.servers[-1][2] == "dev"


def test_launch_appends_colocate_compute(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=True, colocate_compute=True)

    assert "--colocate-compute" in cli.last_command


def _checkpoint(
    path: str = "/checkpoints/run-1/iter_10_hf",
    mount_path: str = "/checkpoints",
) -> Checkpoint:
    return Checkpoint(
        checkpoint_type=CheckpointType.hf,
        name=path.rstrip("/").rsplit("/", 1)[-1],
        path=path,
        timestamp=0.0,
        checkpoints_volume_name="gym-checkpoints",
        checkpoints_mount_path=mount_path,
    )


@pytest.mark.parametrize(
    ("path", "mount_path", "expected"),
    [
        ("/checkpoints/run-1/iter_10_hf", "/checkpoints", "run-1/iter_10_hf"),
        ("/checkpoints/run-1/iter_10_hf/", "/checkpoints", "run-1/iter_10_hf"),
        ("/data/ckpt/iter_5_hf", "/data/ckpt", "iter_5_hf"),
        ("/checkpoints", "/checkpoints", ""),
    ],
)
def test_checkpoint_path_relative_to_volume_strips_the_mount_prefix(
    path: str, mount_path: str, expected: str
) -> None:
    assert _checkpoint(path, mount_path).path_relative_to_volume == expected


@pytest.mark.parametrize(
    ("path", "mount_path", "expected"),
    [
        ("/checkpoints/run-1/iter_10_hf", "/checkpoints", "run-1/iter_10_hf"),
        ("/checkpoints/run-1/iter_10_hf/", "/checkpoints", "run-1/iter_10_hf"),
        ("/data/ckpt/iter_5_hf", "/data/ckpt", "iter_5_hf"),
    ],
)
def test_launch_mounts_the_checkpoint_relative_to_the_volume_root(
    fake_modal_cli, path: str, mount_path: str, expected: str
) -> None:
    cli = fake_modal_cli()

    Endpoint.launch(
        "Qwen/Qwen3-4B", _checkpoint(path, mount_path), unauthenticated=True
    )

    assert cli.flag_value("--custom-volume-name") == "gym-checkpoints"
    assert cli.flag_value("--custom-volume-path") == expected


def test_launch_converts_megatron_checkpoints_before_create(
    fake_modal_cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = fake_modal_cli()
    megatron = Checkpoint(
        checkpoint_type=CheckpointType.megatron,
        name="iter_10",
        path="/checkpoints/run-1/iter_10",
        timestamp=0.0,
        training_run_id="run-1",
        checkpoints_volume_name="gym-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )
    converted = _checkpoint("/checkpoints/run-1/iter_10_hf")
    seen: dict[str, Any] = {}

    def convert(checkpoint, model, **kwargs):
        seen["checkpoint"] = checkpoint
        seen["model"] = model
        return converted

    monkeypatch.setattr(endpoint_module, "convert_megatron_checkpoint_to_hf", convert)

    Endpoint.launch("Qwen/Qwen3-4B", megatron, unauthenticated=True)

    assert seen["checkpoint"] is megatron
    assert seen["model"].model_name == "Qwen/Qwen3-4B"
    assert cli.flag_value("--custom-volume-path") == "run-1/iter_10_hf"


def test_launch_leaves_hf_checkpoints_unchanged(
    fake_modal_cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = fake_modal_cli()
    checkpoint = _checkpoint()
    seen: list[tuple[Checkpoint, Checkpoint]] = []
    original = endpoint_module.convert_megatron_checkpoint_to_hf

    def convert(checkpoint, model, **kwargs):
        result = original(checkpoint, model)
        seen.append((checkpoint, result))
        return result

    monkeypatch.setattr(endpoint_module, "convert_megatron_checkpoint_to_hf", convert)

    Endpoint.launch("Qwen/Qwen3-4B", checkpoint, unauthenticated=True)

    assert seen == [(checkpoint, checkpoint)]
    assert cli.flag_value("--custom-volume-path") == "run-1/iter_10_hf"


def test_launch_skips_conversion_for_hub_models(
    fake_modal_cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_modal_cli()

    def convert(*args, **kwargs):
        raise AssertionError("hub launch converted a checkpoint")

    monkeypatch.setattr(endpoint_module, "convert_megatron_checkpoint_to_hf", convert)

    Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=True)


def test_launch_covers_every_pinned_endpoint_create_flag() -> None:
    named_by_launch = {
        "name",
        "model",
        "env",
        "unauthenticated",
        "routing_region",
        "colocate_compute",
    }
    named_by_checkpoint = {"custom_volume_name", "custom_volume_path"}
    unsupported = {
        "custom_hf_repo",
        "custom_hf_revision",
        "custom_hf_token",
        "compute_regions",
    }

    click_dests = frozenset(
        param.name for param in modal_endpoint_create.params if param.name
    )
    assert click_dests == named_by_launch | named_by_checkpoint | unsupported

    launch_params = set(inspect.signature(Endpoint.launch).parameters)
    assert "kwargs" not in launch_params
    assert not unsupported & launch_params


def test_launch_rejects_unknown_endpoint_options(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    with pytest.raises(TypeError):
        Endpoint.launch(
            "Qwen/Qwen3-4B",
            unauthenticated=True,
            custom_hf_token="hf_test_token",
        )

    assert cli.commands == []


def test_launch_never_puts_an_hf_token_on_the_command_line(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=True)

    assert cli.last_command.count("--custom-hf-token") == 0


def test_launch_derives_stable_names_from_the_serving_spec(fake_modal_cli) -> None:
    fake_modal_cli()

    public = Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=True)
    repeated = Endpoint.launch(
        ModelConfig(model_name="Qwen/Qwen3-4B"), unauthenticated=True
    )
    authenticated = Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=False)
    other_model = Endpoint.launch("Qwen/Qwen3-8B", unauthenticated=True)
    other_region = Endpoint.launch(
        "Qwen/Qwen3-4B", unauthenticated=True, routing_region="us-east"
    )
    checkpointed = Endpoint.launch("Qwen/Qwen3-4B", _checkpoint(), unauthenticated=True)
    other_checkpoint = Endpoint.launch(
        "Qwen/Qwen3-4B",
        _checkpoint("/checkpoints/run-1/iter_20_hf"),
        unauthenticated=True,
    )

    assert public.endpoint_name.startswith("training-gym-")
    assert public.endpoint_name == repeated.endpoint_name
    assert (
        len(
            {
                public.endpoint_name,
                authenticated.endpoint_name,
                other_model.endpoint_name,
                other_region.endpoint_name,
                checkpointed.endpoint_name,
                other_checkpoint.endpoint_name,
            }
        )
        == 6
    )
    assert public.endpoint_name == "training-gym-814a133f9cff"


def test_launch_derived_name_changes_for_colocate(
    fake_modal_cli,
) -> None:
    fake_modal_cli()

    default = Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=True)
    colocated = Endpoint.launch(
        "Qwen/Qwen3-4B", unauthenticated=True, colocate_compute=True
    )

    assert default.endpoint_name != colocated.endpoint_name


def test_launch_uses_an_explicit_endpoint_name(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    endpoint = Endpoint.launch(
        "Qwen/Qwen3-4B", endpoint_name="my-ft", unauthenticated=True
    )

    assert endpoint.endpoint_name == "my-ft"
    assert cli.flag_value("--name") == "my-ft"
    assert cli.servers[-1][:2] == ("ep-my-ft", "Server")


def test_launch_does_not_stop_by_default(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=True)

    assert [command[4] for command in cli.commands] == ["create"]


def test_launch_reuses_existing_named_endpoint(fake_modal_cli) -> None:
    cli = fake_modal_cli()
    cli._create_returncode = 1
    cli._create_stdout = "Endpoint 'my-ft' already exists in environment 'ajhinh-dev'."

    endpoint = Endpoint.launch(
        "Qwen/Qwen3-4B",
        endpoint_name="my-ft",
        unauthenticated=True,
        recreate_if_existing=False,
    )

    assert [command[4] for command in cli.commands] == ["create"]
    assert endpoint.endpoint_name == "my-ft"
    assert endpoint.url == _FakeModalCli.DEFAULT_URL


def test_launch_raises_when_create_fails(fake_modal_cli) -> None:
    cli = fake_modal_cli()
    cli._create_returncode = 1
    cli._create_stdout = "invalid model"

    with pytest.raises(subprocess.CalledProcessError):
        Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=True)

    assert [command[4] for command in cli.commands] == ["create"]
    assert cli.servers == []


def test_launch_reuses_existing_named_endpoint_after_recreate_stop(
    fake_modal_cli,
) -> None:
    cli = fake_modal_cli()
    cli._create_returncode = 1
    cli._create_stdout = "Endpoint 'my-ft' already exists in environment 'ajhinh-dev'."

    endpoint = Endpoint.launch(
        "Qwen/Qwen3-4B",
        endpoint_name="my-ft",
        unauthenticated=True,
        recreate_if_existing=True,
    )

    assert [command[4] for command in cli.commands] == ["stop", "create"]
    assert endpoint.endpoint_name == "my-ft"
    assert endpoint.url == _FakeModalCli.DEFAULT_URL


def test_launch_stops_then_creates_when_recreate_if_existing(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    endpoint = Endpoint.launch(
        "Qwen/Qwen3-4B",
        endpoint_name="my-ft",
        unauthenticated=True,
        recreate_if_existing=True,
    )

    assert [command[4] for command in cli.commands] == ["stop", "create"]
    assert cli.commands[0][5:] == ["my-ft", "--yes"]
    assert cli.run_kwargs[0] == {
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": 120,
    }
    assert cli.flag_value("--name") == "my-ft"
    assert endpoint.endpoint_name == "my-ft"


def test_recreate_stop_forwards_environment(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    Endpoint.launch(
        "Qwen/Qwen3-4B",
        endpoint_name="my-ft",
        unauthenticated=True,
        environment="dev",
        recreate_if_existing=True,
    )

    assert cli.commands[0][1:] == [
        "-m",
        "modal",
        "endpoint",
        "stop",
        "my-ft",
        "--yes",
        "--env",
        "dev",
    ]
    assert cli.flag_value("--env") == "dev"


def test_recreate_creates_when_stop_name_is_absent(fake_modal_cli) -> None:
    cli = fake_modal_cli(stop_returncode=1)
    cli._stop_stderr = "Endpoint 'my-ft' not found in environment 'ajhinh-dev'."

    endpoint = Endpoint.launch(
        "Qwen/Qwen3-4B",
        endpoint_name="my-ft",
        unauthenticated=True,
        recreate_if_existing=True,
    )

    assert [command[4] for command in cli.commands] == ["stop", "create"]
    assert endpoint.endpoint_name == "my-ft"
    assert endpoint.url == _FakeModalCli.DEFAULT_URL


def test_recreate_creates_when_stop_already_stopped(fake_modal_cli) -> None:
    cli = fake_modal_cli(stop_returncode=1)
    cli._stop_stderr = (
        "Endpoint 'my-ft' in environment 'ajhinh-dev' is already stopped."
    )

    endpoint = Endpoint.launch(
        "Qwen/Qwen3-4B",
        endpoint_name="my-ft",
        unauthenticated=True,
        recreate_if_existing=True,
    )

    assert [command[4] for command in cli.commands] == ["stop", "create"]
    assert endpoint.endpoint_name == "my-ft"
    assert endpoint.url == _FakeModalCli.DEFAULT_URL


def test_recreate_raises_when_stop_app_is_absent(fake_modal_cli) -> None:
    cli = fake_modal_cli(stop_returncode=1)
    cli._stop_stderr = "App ap-test not found"

    with pytest.raises(subprocess.CalledProcessError):
        Endpoint.launch(
            "Qwen/Qwen3-4B",
            endpoint_name="my-ft",
            unauthenticated=True,
            recreate_if_existing=True,
        )

    assert [command[4] for command in cli.commands] == ["stop"]


def test_recreate_raises_when_stop_fails(fake_modal_cli) -> None:
    cli = fake_modal_cli(stop_returncode=1)

    with pytest.raises(subprocess.CalledProcessError):
        Endpoint.launch(
            "Qwen/Qwen3-4B",
            endpoint_name="my-ft",
            unauthenticated=True,
            recreate_if_existing=True,
        )

    assert [command[4] for command in cli.commands] == ["stop"]


def test_recreate_raises_when_stop_times_out(fake_modal_cli) -> None:
    cli = fake_modal_cli(
        stop_error=subprocess.TimeoutExpired(["modal", "endpoint", "stop"], 120)
    )

    with pytest.raises(subprocess.TimeoutExpired):
        Endpoint.launch(
            "Qwen/Qwen3-4B",
            endpoint_name="my-ft",
            unauthenticated=True,
            recreate_if_existing=True,
        )

    assert [command[4] for command in cli.commands] == ["stop"]


def test_recreate_does_not_change_derived_name(fake_modal_cli) -> None:
    fake_modal_cli()

    created = Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=True)
    recreated = Endpoint.launch(
        "Qwen/Qwen3-4B", unauthenticated=True, recreate_if_existing=True
    )

    assert created.endpoint_name == recreated.endpoint_name


def test_launch_polls_until_the_server_publishes_a_url(
    fake_modal_cli, clock: _FakeClock
) -> None:
    fake_modal_cli(
        [
            modal.exception.NotFoundError("no server yet"),
            None,
            "https://ws--ep-my-ft.modal.run/",
        ]
    )

    endpoint = Endpoint.launch(
        "Qwen/Qwen3-4B", endpoint_name="my-ft", unauthenticated=True
    )

    assert endpoint.url == "https://ws--ep-my-ft.modal.run"
    assert clock.slept == [1, 1]


def test_launch_times_out_when_no_url_is_published(
    fake_modal_cli, clock: _FakeClock
) -> None:
    fake_modal_cli([])

    with pytest.raises(TimeoutError, match="my-ft"):
        Endpoint.launch(
            "Qwen/Qwen3-4B",
            endpoint_name="my-ft",
            unauthenticated=True,
            wait_timeout_sec=5,
        )

    assert clock.now >= 5


def test_stop_invokes_modal_endpoint_stop(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    _endpoint(environment="dev").stop()

    assert [command[4] for command in cli.commands] == ["stop"]
    assert cli.commands[0][1:] == [
        "-m",
        "modal",
        "endpoint",
        "stop",
        "my-ft",
        "--yes",
        "--env",
        "dev",
    ]
    assert cli.run_kwargs[0] == {
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": 120,
    }


def test_stop_tolerates_missing_endpoint(fake_modal_cli) -> None:
    cli = fake_modal_cli(stop_returncode=1)
    cli._stop_stderr = "Endpoint 'my-ft' not found in environment 'ajhinh-dev'."

    _endpoint().stop()

    assert [command[4] for command in cli.commands] == ["stop"]


def test_headers_are_empty_for_public_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_KEY", "wk-test")
    monkeypatch.setenv("MODAL_SECRET", "ws-test")

    assert _endpoint()._headers() == {}


def test_headers_carry_proxy_credentials_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_KEY", "wk-test")
    monkeypatch.setenv("MODAL_SECRET", "ws-test")

    assert _endpoint(requires_proxy_auth=True)._headers() == {
        "Modal-Key": "wk-test",
        "Modal-Secret": "ws-test",
    }


def test_headers_require_configured_proxy_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(endpoint_module, "modal_proxy_auth_headers", lambda: {})

    with pytest.raises(DojoConfigError, match="MODAL_KEY and MODAL_SECRET"):
        _endpoint(requires_proxy_auth=True)._headers()


def test_wait_until_ready_polls_the_model_route(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    responses = [_FakeResponse(503), _FakeResponse(404), _FakeResponse()]
    requests: list[tuple[str, dict[str, Any]]] = []

    def _get(url: str, **kwargs: Any) -> _FakeResponse:
        requests.append((url, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(endpoint_module.httpx, "get", _get)

    _endpoint().wait_until_ready(timeout=60)

    assert requests[0][0] == "https://ws--ep.modal.run/v1/models"
    assert requests[0][1]["headers"] == {}
    assert clock.slept == [2, 2]


def test_wait_until_ready_sends_proxy_headers_when_required(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        endpoint_module,
        "modal_proxy_auth_headers",
        lambda: {"Modal-Key": "wk", "Modal-Secret": "ws"},
    )
    monkeypatch.setattr(
        endpoint_module.httpx,
        "get",
        lambda *_, **kwargs: captured.update(kwargs) or _FakeResponse(),
    )

    _endpoint(requires_proxy_auth=True).wait_until_ready(timeout=60)

    assert captured["headers"]["Modal-Key"] == "wk"


@pytest.mark.parametrize("status_code", [401, 403])
def test_wait_until_ready_reports_rejected_proxy_credentials(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock, status_code: int
) -> None:
    monkeypatch.setattr(
        endpoint_module,
        "modal_proxy_auth_headers",
        lambda: {"Modal-Key": "wk", "Modal-Secret": "ws"},
    )
    monkeypatch.setattr(
        endpoint_module.httpx, "get", lambda *_, **__: _FakeResponse(status_code)
    )

    with pytest.raises(RuntimeError, match="rejected proxy authentication"):
        _endpoint(requires_proxy_auth=True).wait_until_ready(timeout=60)


def test_wait_until_ready_surfaces_unexpected_statuses(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    monkeypatch.setattr(
        endpoint_module.httpx, "get", lambda *_, **__: _FakeResponse(418)
    )

    with pytest.raises(httpx.HTTPStatusError):
        _endpoint().wait_until_ready(timeout=60)


def test_wait_until_ready_times_out_and_keeps_the_last_error(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    def _get(*_: Any, **__: Any) -> _FakeResponse:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(endpoint_module.httpx, "get", _get)

    with pytest.raises(TimeoutError, match="my-ft") as excinfo:
        _endpoint().wait_until_ready(timeout=6)

    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)
    assert clock.slept == [2, 2, 2]


def test_chat_posts_the_conversation_and_returns_the_assistant_message(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    captured: dict[str, Any] = {}

    def _post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse(message={"role": "assistant", "content": "hi"})

    monkeypatch.setattr(endpoint_module.httpx, "post", _post)

    message = _endpoint().chat([{"role": "user", "content": "hello"}], temperature=0)

    assert message == {"role": "assistant", "content": "hi"}
    assert captured["url"] == "https://ws--ep.modal.run/v1/chat/completions"
    assert captured["json"] == {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }
    assert captured["headers"] == {}


def test_chat_serializes_tool_arguments_without_mutating_messages(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    captured: dict[str, Any] = {}

    def _post(url: str, **kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(message={"role": "assistant", "content": ""})

    monkeypatch.setattr(endpoint_module.httpx, "post", _post)

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"key": "value"},
                    },
                }
            ],
        }
    ]

    _endpoint().chat(messages)

    assert captured["json"]["messages"][0]["tool_calls"][0]["function"][
        "arguments"
    ] == ('{"key": "value"}')
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"key": "value"}


def test_chat_sends_proxy_headers_when_required(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        endpoint_module,
        "modal_proxy_auth_headers",
        lambda: {"Modal-Key": "wk", "Modal-Secret": "ws"},
    )
    monkeypatch.setattr(
        endpoint_module.httpx,
        "post",
        lambda *_, **kwargs: captured.update(kwargs) or _FakeResponse(),
    )

    _endpoint(requires_proxy_auth=True).chat([])

    assert captured["headers"]["Modal-Key"] == "wk"


def test_chat_requires_proxy_credentials_before_posting(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    def _post(*_: Any, **__: Any) -> _FakeResponse:
        raise AssertionError("chat must not post without proxy credentials")

    monkeypatch.setattr(endpoint_module, "modal_proxy_auth_headers", lambda: {})
    monkeypatch.setattr(endpoint_module.httpx, "post", _post)

    with pytest.raises(DojoConfigError, match="Proxy authentication requires"):
        _endpoint(requires_proxy_auth=True).chat([])


def test_chat_retries_transient_statuses(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    responses = [_FakeResponse(503), _FakeResponse(429), _FakeResponse()]
    monkeypatch.setattr(
        endpoint_module.httpx, "post", lambda *_, **__: responses.pop(0)
    )

    assert _endpoint().chat([])["content"] == "ok"
    assert clock.slept == [2, 4]


def test_chat_raises_after_the_last_transient_attempt(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    monkeypatch.setattr(
        endpoint_module.httpx, "post", lambda *_, **__: _FakeResponse(503)
    )

    with pytest.raises(httpx.HTTPStatusError):
        _endpoint().chat([], max_attempts=2)

    assert clock.slept == [2]


def test_chat_retries_request_errors_then_reraises(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    attempts: list[int] = []

    def _post(*_: Any, **__: Any) -> _FakeResponse:
        attempts.append(len(attempts) + 1)
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(endpoint_module.httpx, "post", _post)

    with pytest.raises(httpx.ConnectError):
        _endpoint().chat([], max_attempts=3)

    assert len(attempts) == 3
    assert clock.slept == [2, 4]


@pytest.mark.parametrize("status_code", [401, 403])
def test_chat_reports_rejected_proxy_credentials(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock, status_code: int
) -> None:
    monkeypatch.setattr(
        endpoint_module,
        "modal_proxy_auth_headers",
        lambda: {"Modal-Key": "wk", "Modal-Secret": "ws"},
    )
    monkeypatch.setattr(
        endpoint_module.httpx, "post", lambda *_, **__: _FakeResponse(status_code)
    )

    with pytest.raises(RuntimeError, match="Proxy credentials were rejected"):
        _endpoint(requires_proxy_auth=True).chat([])
