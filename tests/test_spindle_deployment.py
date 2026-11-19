"""Connection precedence, backend reuse, and Tinker routing without GPUs."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from modal_dojo.common.errors import DojoConfigError
from modal_dojo.frameworks.spindle import connection, deployment, launcher
from modal_dojo.frameworks.spindle.deployment import Requirements

MODEL = "Qwen/Qwen3.5-2B"
URL = "https://training.example.com"
REQ = Requirements(MODEL, 64, 32, 32768)


def row(name="existing", context=32768, **extra):
    return dict(
        name=name,
        base_model=MODEL,
        parameterization="lora",
        max_context_length=context,
        **extra,
    )


def transport(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(
        launcher.httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )


def test_direct_connection_does_not_discover_or_persist(monkeypatch):
    lookup = Mock(side_effect=AssertionError("Must not look up a default"))
    monkeypatch.setattr(connection, "_secret_exists", lookup)
    result = connection.resolve_connection(URL, "test-key", REQ, environment="test")
    assert result.base_url == URL
    assert result.api_key == "test-key"
    assert "test-key" not in repr(result)
    lookup.assert_not_called()


def test_explicit_url_never_borrows_an_unrelated_saved_key(monkeypatch):
    monkeypatch.setattr(connection, "_secret_exists", Mock(side_effect=AssertionError))
    with pytest.raises(DojoConfigError, match="api_key"):
        connection.resolve_connection(URL, None, REQ, environment="test")


@pytest.mark.parametrize("setup_fails", [False, True])
def test_dashboard_failure_does_not_provision_spindle(monkeypatch, setup_fails):
    from modal_dojo.frameworks.spindle.app import launch

    monkeypatch.setattr(
        "modal_dojo.common.config.require_migrated_config", lambda: None
    )
    monkeypatch.setattr(
        "modal_dojo.common.config.get_framework_status_url", lambda: None
    )
    monkeypatch.setattr(
        "modal_dojo.cli.setup.ensure_dashboard_deployed",
        Mock(
            side_effect=DojoConfigError("dashboard setup failed")
            if setup_fails
            else None
        ),
    )
    resolve = Mock(
        side_effect=AssertionError("Must not provision before dashboard setup")
    )
    monkeypatch.setattr(connection, "resolve_connection", resolve)
    with pytest.raises(
        DojoConfigError, match=r"dashboard setup failed|Could not connect"
    ):
        launch(SimpleNamespace(program=SimpleNamespace(model_name=MODEL)), yes=True)
    resolve.assert_not_called()


def test_lookup_uses_the_current_environment_and_the_servers_named_secret(monkeypatch):
    import modal

    server = Mock()
    server.get_web_url.return_value = URL
    platform = Mock()
    platform.remote.return_value = {"secrets": {"api": "custom-server-api"}}
    lookup = Mock(side_effect=[server, platform])
    exists = Mock(return_value=True)
    monkeypatch.setattr(modal.Function, "from_name", lookup)
    monkeypatch.setattr(connection, "_secret_exists", exists)

    result = connection.lookup_server("test")

    assert [call.args for call in lookup.call_args_list] == [
        (deployment.SERVER_NAME, "server"),
        (deployment.SERVER_NAME, "deployed_platform"),
    ]
    assert all(
        call.kwargs == {"environment_name": "test"} for call in lookup.call_args_list
    )
    exists.assert_called_once_with("custom-server-api", "test", ["TINKER_API_KEY"])
    assert result == connection.Connection(
        URL,
        secret_name="custom-server-api",
        environment="test",
        app_name=deployment.SERVER_NAME,
    )


def test_saved_connection_precedes_discovery_and_stays_remote(monkeypatch):
    exists = Mock(return_value=True)
    monkeypatch.setattr(connection, "_secret_exists", exists)
    monkeypatch.setattr(connection, "lookup_server", Mock(side_effect=AssertionError))
    result = connection.resolve_connection(None, None, REQ, environment="test")
    assert result.secret_name == connection.DEFAULT_SECRET
    assert result.api_key is None
    assert result.base_url is None
    assert result.environment == "test"
    exists.assert_called_once_with(
        connection.DEFAULT_SECRET, "test", ["TINKER_BASE_URL", "TINKER_API_KEY"]
    )


def test_discovery_reuses_existing_server_without_confirmation(monkeypatch):
    existing = connection.Connection(URL, secret_name="spindle-api", environment="test")
    monkeypatch.setattr(connection, "_secret_exists", lambda *args: False)
    monkeypatch.setattr(connection, "lookup_server", lambda env: existing)
    monkeypatch.setattr(connection, "select_config", Mock(side_effect=AssertionError))
    assert (
        connection.resolve_connection(None, None, REQ, environment="test") == existing
    )


def test_lookup_failure_does_not_trigger_deployment(monkeypatch):
    monkeypatch.setattr(connection, "_secret_exists", lambda *args: False)
    monkeypatch.setattr(
        connection, "lookup_server", Mock(side_effect=RuntimeError("permission denied"))
    )
    deploy = Mock()
    monkeypatch.setattr(connection, "deploy_server", deploy)
    with pytest.raises(RuntimeError, match="permission denied"):
        connection.resolve_connection(None, None, REQ, yes=True, environment="test")
    deploy.assert_not_called()


def test_noninteractive_missing_server_requires_permission(monkeypatch):
    monkeypatch.setattr(connection.sys.stdin, "isatty", lambda: False)
    preset = SimpleNamespace(
        model=MODEL,
        max_context_length=32768,
        trainer_nodes=1,
        trainer_gpus_per_node=4,
        trainer_gpu="H100",
        inference_gpus_per_node=1,
        inference_gpu="H100",
    )
    monkeypatch.setattr(connection, "_secret_exists", lambda *args: False)
    monkeypatch.setattr(connection, "lookup_server", lambda env: None)
    monkeypatch.setattr(connection, "select_config", lambda req: preset)
    deploy = Mock()
    monkeypatch.setattr(connection, "deploy_server", deploy)
    with pytest.raises(DojoConfigError, match="yes=True"):
        connection.resolve_connection(None, None, REQ, environment="test")
    deploy.assert_not_called()
    connection.resolve_connection(None, None, REQ, yes=True, environment="test")
    deploy.assert_called_once_with(preset, "test")


def test_server_created_while_user_confirms_is_reused(monkeypatch):
    existing = connection.Connection(URL, secret_name="spindle-api")
    monkeypatch.setattr(connection, "_secret_exists", lambda *args: False)
    monkeypatch.setattr(connection, "lookup_server", Mock(side_effect=[None, existing]))
    monkeypatch.setattr(connection, "select_config", lambda req: object())
    monkeypatch.setattr(connection, "_confirm_deploy", lambda *args, **kw: None)
    monkeypatch.setattr(connection, "deploy_server", Mock(side_effect=AssertionError))
    assert (
        connection.resolve_connection(None, None, REQ, yes=True, environment="test")
        == existing
    )


def test_external_tinker_uses_only_standard_api_and_no_modal_credentials(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "modal_dojo.common.config.modal_proxy_auth_headers",
        lambda: {"Modal-Key": "private"},
    )

    def handler(request):
        calls.append(request.url.path)
        assert "Modal-Key" not in request.headers
        assert request.headers["X-API-Key"] == "test-key"
        return httpx.Response(
            200,
            json={
                "supported_models": [{"model_name": MODEL, "max_context_length": 32768}]
            },
        )

    transport(monkeypatch, handler)
    result = launcher.check_server(URL, "test-key", REQ)
    assert result["tinker_provider"] == "external"
    assert calls == ["/api/v1/get_server_capabilities"]


@pytest.mark.parametrize("status", [401, 403, 500])
def test_http_failure_does_not_provision(monkeypatch, status):
    calls = []

    def handler(request):
        calls.append(request.method)
        return httpx.Response(status, json={"error": "failed"})

    transport(monkeypatch, handler)
    with pytest.raises(DojoConfigError):
        launcher.check_server(URL, "test-key", REQ)
    assert calls == ["GET"]


def test_invalid_response_is_reported_as_a_configuration_error(monkeypatch):
    transport(monkeypatch, lambda request: httpx.Response(200, json=[]))
    with pytest.raises(DojoConfigError, match="JSON object"):
        launcher.check_server(URL, "test-key", REQ)


def test_spindle_reuses_smallest_sufficient_deployment(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request.method)
        if request.url.path.endswith("get_server_capabilities"):
            return httpx.Response(200, json={"spindle_version": "0.1.8"})
        return httpx.Response(
            200,
            json={
                "deployments": [
                    row("short", 8192),
                    row("large", 65536),
                    row("fit", 32768),
                ]
            },
        )

    transport(monkeypatch, handler)
    result = launcher.check_server(URL, "test-key", REQ)
    assert result["spindle_deployment"] == "fit"
    assert requests == ["GET", "GET"]


def test_ready_backend_is_preferred_to_repairing_a_failed_one():
    def handler(request):
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={
                "deployments": [
                    row("failed", apps="failed"),
                    row("ready", 65536, apps="ready"),
                ]
            },
        )

    with httpx.Client(base_url=URL, transport=httpx.MockTransport(handler)) as client:
        assert deployment.ensure_backend(client, REQ)["name"] == "ready"


def test_gpu_override_adds_one_backend_and_reuses_it(monkeypatch):
    preset = SimpleNamespace(
        name="qwen35-2b-lora-32k",
        model=MODEL,
        trainer_gpu="H100",
        max_context_length=32768,
    )
    monkeypatch.setattr(deployment, "select_config", lambda req: preset)
    rows, created = [row("existing")], []

    def handler(request):
        if request.method == "POST":
            body = json.loads(request.content)
            created.append(body)
            rows.append(row(body["name"], apps="ready"))
            return httpx.Response(202, json={})
        return httpx.Response(200, json={"deployments": rows})

    req = Requirements(MODEL, 64, 32, 32768, {"trainer_gpu": "H100"})
    with httpx.Client(base_url=URL, transport=httpx.MockTransport(handler)) as client:
        first = deployment.ensure_backend(client, req)
        second = deployment.ensure_backend(client, req)
    assert first["name"] == second["name"]
    assert len(created) == 1
    assert created[0]["recipe"]["trainer_gpu"] == "H100"
    assert rows[0]["name"] == "existing"


def test_backend_identity_includes_hardware_and_is_scoped_by_frontend():
    preset = SimpleNamespace(
        name="model-32k", model=MODEL, trainer_gpu="H100", platform={}
    )
    first = deployment.deployment_name(preset)
    assert len(f"{deployment.SERVER_NAME}-inference-{first}") < 64
    assert deployment.deployment_name(copy.deepcopy(preset)) == first
    preset.platform = {"frontend": "another-server"}
    assert deployment.deployment_name(preset) == first
    preset.trainer_gpu = "H200"
    assert deployment.deployment_name(preset) != first


def test_real_spindle_preset_and_deployment_request():
    pytest.importorskip("spindle")
    from spindle.deployments import DeploymentConfig
    from spindle.control_plane.deployments import DeploymentRequest, DeploymentRoutes

    req = Requirements(
        MODEL,
        64,
        32,
        32768,
        {"trainer_gpu": "H200", "trainer_gpus_per_node": 2, "inference_gpu": "H200"},
    )
    config = deployment.select_config(req)
    assert config.model == MODEL
    assert config.max_context_length == 32768
    assert config.trainer_gpu == "H200"
    assert config.trainer_gpus_per_node == 2
    assert config.inference_gpu == "H200"
    config.platform["frontend"] = deployment.SERVER_NAME
    routes = DeploymentRoutes.from_definitions([DeploymentConfig.create(config)])
    name = deployment.deployment_name(config)
    compiled = asyncio.run(
        routes.build(DeploymentRequest(name=name, recipe=vars(config)))
    )
    assert compiled.definition_id == name
    assert compiled.recipe.trainer_gpu == "H200"
    assert compiled.recipe.trainer_gpus_per_node == 2
    assert compiled.recipe.inference_gpu == "H200"
    for app_name in (
        compiled.trainer_app_name,
        compiled.inference_app_name,
        f"{deployment.SERVER_NAME}-lora-{name}",
    ):
        assert len(app_name) < 64
    with pytest.raises(DojoConfigError, match="No bundled"):
        deployment.select_config(Requirements(MODEL, 64, 32, 1000000))


def test_local_spindle_version_must_match_the_managed_app_version(monkeypatch):
    spindle = pytest.importorskip("spindle")
    monkeypatch.setattr(spindle, "__version__", "0.1.9")
    with pytest.raises(DojoConfigError, match=r"requires Spindle 0\.1\.8"):
        deployment.select_config(REQ)


def test_deepseek_1017_has_a_bundled_64k_backend():
    pytest.importorskip("spindle")
    config = deployment.select_config(
        Requirements("deepseek-ai/DeepSeek-V4.1-Flash-1017", 2048, 32, 65536)
    )
    assert config.model == "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    assert config.max_context_length == 65536


def test_routing_keeps_program_model_and_other_services_unchanged(monkeypatch):
    tinker = pytest.importorskip("tinker")
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.options = kwargs

        def create_lora_training_client(self, base_model, *args, **kwargs):
            calls.append(base_model)

        async def create_lora_training_client_async(self, base_model, *args, **kwargs):
            calls.append(base_model)

        def create_sampling_client(
            self, model_path=None, base_model=None, *args, **kwargs
        ):
            calls.append(base_model)

    monkeypatch.setattr(tinker, "ServiceClient", FakeClient)
    with launcher.routed_clients(URL, MODEL, "selected-h100") as Client:
        client = Client()
        client.create_lora_training_client(MODEL)
        asyncio.run(client.create_lora_training_client_async(MODEL))
        client.create_sampling_client(base_model=MODEL)
        client.create_lora_training_client("different-model")
        Client(base_url="https://other.example.com").create_lora_training_client(MODEL)
    assert calls == [
        "selected-h100",
        "selected-h100",
        "selected-h100",
        "different-model",
        MODEL,
    ]
    assert tinker.ServiceClient is FakeClient


def test_raw_cookbook_config_and_hardware_validation(monkeypatch):
    pytest.importorskip("tinker_cookbook")
    from tinker_cookbook.rl.train import Config
    from modal_dojo import SpindleRecipe

    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    config = Config(
        model_name=MODEL,
        recipe_name="test",
        dataset_builder=lambda: None,
        learning_rate=1e-5,
        max_tokens=64,
        log_path="/tmp/test-spindle",
    )
    recipe = SpindleRecipe(config, context_length=32768, trainer_gpu="H100")
    assert recipe.program.config is config
    assert recipe.requirements.overrides == {"trainer_gpu": "H100"}
    assert recipe.base_url is None
    with pytest.raises(DojoConfigError, match="positive integer"):
        SpindleRecipe(config, trainer_gpus_per_node=0)
    with pytest.raises(DojoConfigError, match="leave room"):
        SpindleRecipe(config, context_length=64)


def test_backend_with_previous_server_suffix_is_reused(monkeypatch):
    config = SimpleNamespace(name="qwen35-2b-lora-32k", trainer_gpu="H100", model=MODEL)
    monkeypatch.setattr(deployment, "select_config", lambda req: config)
    existing = row(deployment.deployment_name(config) + "-4e8eaf", apps="ready")

    def handler(request):
        assert request.method == "GET"
        return httpx.Response(200, json={"deployments": [existing]})

    req = Requirements(MODEL, 64, 32, 32768, {"trainer_gpu": "H100"})
    with httpx.Client(base_url=URL, transport=httpx.MockTransport(handler)) as client:
        assert deployment.ensure_backend(client, req) == existing


def test_failed_backend_is_redeployed_in_place():
    existing = row(apps="failed")
    posts = []

    def handler(request):
        if request.method == "POST":
            posts.append(json.loads(request.content))
            existing["apps"] = "ready"
            return httpx.Response(202, json={})
        return httpx.Response(200, json={"deployments": [existing]})

    with httpx.Client(base_url=URL, transport=httpx.MockTransport(handler)) as client:
        result = deployment.ensure_backend(client, REQ)
    assert result["name"] == "existing"
    assert posts == [{"name": "existing", "base": "existing"}]


def test_real_spindle_http_api_registers_and_reuses_config():
    pytest.importorskip("spindle")
    from fastapi.testclient import TestClient
    from spindle.control_plane.http import create_control_plane_app
    from spindle.control_plane.deployments import DeploymentRoutes
    from spindle.deployments import DeploymentConfig

    config = deployment.select_config(REQ)
    config.session_idle_timeout_s = 600
    config.pool_idle_timeout_s = 900
    compiled = DeploymentConfig.create(config)

    class MemoryStore:
        def __init__(self):
            self.rows = {compiled.name: compiled.model_dump(mode="json")}

        async def get(self, key):
            return self.rows.get(key)

        async def put(self, key, value):
            self.rows[key] = value

        async def list_keys(self, prefix):
            return [key for key in self.rows if key.startswith(prefix)]

    store = MemoryStore()
    routes = DeploymentRoutes(lambda: store)
    published = []

    async def provision(item):
        published.append(item.name)

    async def status(item):
        return {"apps": "ready"}

    app = create_control_plane_app(
        SimpleNamespace(),
        routes,
        api_key="test-key",
        apps_status=status,
        deploy_apps=provision,
    )
    req = Requirements(MODEL, 64, 32, 32768, {"trainer_gpu": "H100"})
    with TestClient(app, headers={"X-API-Key": "test-key"}) as client:
        first = deployment.ensure_backend(client, req)
        second = deployment.ensure_backend(client, req)
    assert first["name"] == second["name"]
    assert published == [first["name"]]
    assert len(store.rows) == 2
    assert store.rows[first["name"]]["recipe"]["session_idle_timeout_s"] == 600
    assert store.rows[first["name"]]["recipe"]["pool_idle_timeout_s"] == 900


def test_managed_server_version_mismatch_does_not_provision(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"spindle_version": "0.1.5"})

    transport(monkeypatch, handler)
    with pytest.raises(DojoConfigError, match="not serving Spindle"):
        launcher.check_server(URL, "test-key", REQ, server_app=deployment.SERVER_NAME)
    assert calls == ["/api/v1/get_server_capabilities"]


def test_new_server_uses_stable_names_and_preserves_credentials(monkeypatch):
    pytest.importorskip("spindle")
    import modal
    from spindle import deployment_cli

    config = deployment.select_config(REQ)
    writes, published = [], []

    class FakeSecret:
        objects = SimpleNamespace(
            create=lambda name, values, **kwargs: writes.append((name, values, kwargs))
        )

    monkeypatch.setattr(modal, "Secret", FakeSecret)
    monkeypatch.setattr(
        "modal_dojo.common.config.modal_proxy_auth_headers",
        lambda: {"Modal-Key": "wk-test", "Modal-Secret": "ws-test"},
    )
    monkeypatch.setattr(connection, "_secret_exists", lambda *args: True)
    existing = connection.Connection(URL, secret_name="api", environment="test")
    monkeypatch.setattr(connection, "lookup_server", lambda env: existing)
    monkeypatch.setattr(
        deployment_cli, "deploy", lambda configs: published.extend(configs)
    )
    assert connection.deploy_server(copy.deepcopy(config), "test") == existing
    assert connection.deploy_server(copy.deepcopy(config), "test") == existing
    assert published[0].name == published[1].name
    assert published[0].recipe.platform["frontend"] == deployment.SERVER_NAME
    assert published[0].recipe.platform["modal"]["environment"] == "test"
    assert all(
        options["allow_existing"] and options["environment_name"] == "test"
        for _, _, options in writes
    )
    assert writes[1][0] == writes[3][0] == deployment.SERVER_NAME + "-api"
