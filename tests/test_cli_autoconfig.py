"""``modal-dojo autoconfig``: flag parsing, offline planning, dashboard calls."""

from __future__ import annotations

import json

import click
import pytest
from click.testing import CliRunner

from modal_dojo import cli as cli_module
from modal_dojo.cli import autoconfig as autoconfig_module
from modal_dojo.cli.autoconfig import parse_grid, parse_sets, parse_tokens


class FakeDashboardClient:
    payloads: dict[str, object] = {}
    posts: list[tuple[str, object]] = []
    gets: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_json(self, path, *, params=None, not_found_error=None, timeout=None):
        self.gets.append(path)
        if path not in self.payloads and not_found_error is not None:
            raise not_found_error
        return self.payloads[path]

    def post_json(self, path, body, *, timeout=None):
        self.posts.append((path, body))
        return self.payloads[path]


@pytest.fixture
def runner(monkeypatch):
    FakeDashboardClient.payloads = {}
    FakeDashboardClient.posts = []
    FakeDashboardClient.gets = []
    monkeypatch.setattr(autoconfig_module, "DashboardClient", FakeDashboardClient)
    monkeypatch.setattr(autoconfig_module, "POLL_INTERVAL_SECONDS", 0)
    return CliRunner()


def invoke(runner, *args):
    return runner.invoke(cli_module.entrypoint_cli, ["autoconfig", *args])


def test_flag_parsers() -> None:
    assert parse_tokens("32k") == 32768
    assert parse_tokens("32768") == 32768
    assert parse_tokens("1M") == 1_048_576
    assert parse_tokens(None) is None
    with pytest.raises(click.BadParameter):
        parse_tokens("lots")
    assert parse_sets(("recipe.gpu_type=B300", "recipe.lr=1e-6")) == {
        "recipe.gpu_type": "B300",
        "recipe.lr": 1e-6,
    }
    assert parse_grid(("recipe.lr=1e-6, 5e-6", "recipe.use_kl_loss=true,false")) == {
        "recipe.lr": [1e-6, 5e-6],
        "recipe.use_kl_loss": [True, False],
    }
    with pytest.raises(click.BadParameter):
        parse_grid(("recipe.lr",))


def test_models_lists_registry_with_history(runner) -> None:
    FakeDashboardClient.payloads["/api/autoconfig/catalogue"] = {
        "rows": [
            {
                "name": "Qwen3.8-27B",
                "run": {
                    "id": "tr-1",
                    "status": "completed",
                    "cost_per_step_usd": 1.5,
                    "time_per_step_s": 120.0,
                    "initial_reward": 0.2,
                },
            }
        ]
    }
    result = invoke(runner, "models", "--json")
    assert result.exit_code == 0, result.output
    models = {m["name"]: m for m in json.loads(result.output)}
    assert models["Qwen3.8-27B"]["history"]["cost_per_step_usd"] == 1.5
    assert models["Qwen3.6-27B"]["history"] is None
    assert models["Qwen3.6-27B"]["recipe"] == "Qwen3_6_27B_Recipe"
    assert "Qwen3.5-4B-SFT" not in models


def test_knobs_shows_common_fields_by_default(runner) -> None:
    result = invoke(runner, "knobs", "qwen3.8-27b", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    paths = [k["path"] for k in payload["knobs"]]
    assert "recipe.lr" in paths and "recipe.gpu_type" in paths
    assert all(k["highlight"] for k in payload["knobs"])
    assert len(
        json.loads(invoke(runner, "knobs", "Qwen3.8-27B", "--all", "--json").output)[
            "knobs"
        ]
    ) > len(paths)
    missing = invoke(runner, "knobs", "Nope-1B", "--json")
    assert missing.exit_code != 0
    assert "unknown model 'Nope-1B'" in missing.stderr


def test_plan_is_offline_and_reports_memory(runner) -> None:
    result = invoke(
        runner,
        "plan",
        "-m",
        "Qwen3.8-27B",
        "--context",
        "32k",
        "--grid",
        "recipe.lr=1e-6,5e-6",
        "--json",
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["request"]["context_length"] == 32768
    assert [v["overrides"] for v in plan["variants"]] == [
        {"recipe.lr": 1e-6},
        {"recipe.lr": 5e-6},
    ]
    assert plan["fits"] and plan["variants"][0]["memory"]["fits"]
    assert FakeDashboardClient.posts == [] and FakeDashboardClient.gets == []

    bad = invoke(
        runner,
        "plan",
        "-m",
        "Qwen3.8-27B",
        "--grid",
        "recipe.learning_rate=1",
        "--json",
    )
    assert bad.exit_code != 0
    assert "recipe.learning_rate" in bad.stderr

    dataset = invoke(
        runner, "plan", "-m", "Qwen3.8-27B", "--dataset", "me/my-set", "--json"
    )
    assert dataset.exit_code != 0 and "input column" in dataset.stderr


def test_sweep_refuses_oom_then_submits_and_waits(runner) -> None:
    oom = invoke(
        runner,
        "sweep",
        "-m",
        "Qwen3.8-27B",
        "--context",
        "64k",
        "--set",
        "recipe.gpu_type=H200",
        "--json",
    )
    assert oom.exit_code != 0
    assert "exceed GPU memory" in oom.stderr
    assert FakeDashboardClient.posts == []

    FakeDashboardClient.payloads["/api/autoconfig/sweeps"] = {
        "operation_id": "fc-1",
        "group_id": "autoconfig-lr-abc",
        "status": "pending",
    }
    FakeDashboardClient.payloads["/api/autoconfig/sweeps/fc-1"] = {
        "operation_id": "fc-1",
        "group_id": "autoconfig-lr-abc",
        "status": "succeeded",
        "result": {
            "group_id": "autoconfig-lr-abc",
            "launched": [
                {
                    "entry": "Qwen3.8-27B",
                    "overrides": {"recipe.lr": 1e-6},
                    "training_run_id": "tr-1",
                    "modal_app_id": "ap-1",
                }
            ],
            "failures": [],
        },
    }
    result = invoke(
        runner,
        "sweep",
        "-m",
        "Qwen3.8-27B",
        "--context",
        "32k",
        "--grid",
        "recipe.lr=1e-6",
        "--name",
        "lr",
        "--wait",
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["operation"]["status"] == "succeeded"
    assert payload["plan"]["variants"][0]["context_length"] == 32768
    path, body = FakeDashboardClient.posts[0]
    assert path == "/api/autoconfig/sweeps"
    assert body["entries"] == ["Qwen3.8-27B"] and body["context_length"] == 32768
    assert body["grid"] == {"recipe.lr": [1e-6]} and body["name"] == "lr"
    assert body["dataset"]["hf_repo"] == "SWE-bench/SWE-smith"
    assert FakeDashboardClient.gets == ["/api/autoconfig/sweeps/fc-1"]


def test_status_polls_and_maps_not_found(runner) -> None:
    FakeDashboardClient.payloads["/api/autoconfig/sweeps/fc-9"] = {
        "operation_id": "fc-9",
        "group_id": "autoconfig-x",
        "status": "failed",
        "error": {"type": "RuntimeError", "message": "boom"},
    }
    result = invoke(runner, "status", "fc-9")
    assert result.exit_code == 0, result.output
    assert "RuntimeError: boom" in result.output
    missing = invoke(runner, "status", "fc-nope", "--json")
    assert missing.exit_code != 0
    assert "was not found" in missing.stderr
