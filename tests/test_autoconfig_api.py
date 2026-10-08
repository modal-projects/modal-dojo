"""The Autoconfig API: catalogue payload, sweep submission and polling."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from modal_dojo import _autoconfig
from modal_dojo._autoconfig import (
    SweepBackendError,
    SweepNotFoundError,
    SweepOperation,
    SweepRequest,
    launch_sweep,
    mount_autoconfig_api,
    plan_sweep,
    sweep_group_id,
)
from modal_dojo.common.catalogue import AUTOCONFIG_ENTRY_TAG, CATALOGUE
from modal_dojo.common.run_summary import RunSummary


class FakeOperations:
    def __init__(self) -> None:
        self.submitted: list[tuple[SweepRequest, str]] = []
        self.calls: dict[str, SweepOperation] = {}

    def submit(self, request: SweepRequest, group_id: str) -> SweepOperation:
        self.submitted.append((request, group_id))
        op = SweepOperation(
            operation_id=f"fc-{len(self.submitted)}",
            group_id=group_id,
            status="pending",
        )
        self.calls[op.operation_id] = op
        return op

    def get(self, operation_id: str) -> SweepOperation:
        if operation_id == "fc-down":
            raise SweepBackendError("modal unavailable")
        try:
            return self.calls[operation_id]
        except KeyError:
            raise SweepNotFoundError("sweep operation not found") from None


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(_autoconfig, "build_catalogue", lambda *a, **k: {"rows": []})
    ops = FakeOperations()
    summaries: list[RunSummary] = []

    async def load_run_summaries() -> list[RunSummary]:
        return summaries

    web = FastAPI()
    mount_autoconfig_api(
        web,
        load_run_summaries=load_run_summaries,
        operations=ops,
        rates=lambda: {"gpu_hour_cost_h200": "4.0"},
    )
    return TestClient(web), ops


def test_catalogue_and_entries_are_served(api) -> None:
    client, _ = api
    assert client.get("/api/autoconfig/catalogue").json() == {"rows": []}
    names = [e["name"] for e in client.get("/api/autoconfig/entries").json()]
    assert names == [e.name for e in CATALOGUE]


def test_submitting_a_sweep_is_accepted_and_pollable(api) -> None:
    client, ops = api
    res = client.post(
        "/api/autoconfig/sweeps",
        json={
            "entries": ["Qwen3.6-27B"],
            "name": "LR sweep",
            "grid": {"recipe.lr": [1e-6]},
        },
    )
    assert res.status_code == 202
    op = res.json()
    assert op["status"] == "pending"
    assert op["group_id"].startswith("autoconfig-lr-sweep-")
    request, group_id = ops.submitted[0]
    assert request.steps == 3 and group_id == op["group_id"]

    polled = client.get(f"/api/autoconfig/sweeps/{op['operation_id']}")
    assert polled.status_code == 200 and polled.json() == op


def test_sweep_requests_are_validated(api) -> None:
    client, ops = api
    assert (
        client.post("/api/autoconfig/sweeps", json={"entries": []}).status_code == 422
    )
    unknown = client.post("/api/autoconfig/sweeps", json={"entries": ["Nope-1B"]})
    assert unknown.status_code == 422 and "unknown catalogue entry" in unknown.text
    extra = client.post(
        "/api/autoconfig/sweeps", json={"entries": ["Kimi-K3"], "bogus": 1}
    )
    assert extra.status_code == 422
    assert ops.submitted == []


def test_polling_maps_backend_errors(api) -> None:
    client, _ = api
    assert client.get("/api/autoconfig/sweeps/fc-missing").status_code == 404
    assert client.get("/api/autoconfig/sweeps/fc-down").status_code == 503


def test_group_ids_are_prefixed_and_unique() -> None:
    a, b = sweep_group_id("My Sweep!"), sweep_group_id("My Sweep!")
    assert a.startswith("autoconfig-my-sweep-") and a != b
    assert sweep_group_id(None).startswith("autoconfig-")


def test_plan_sweep_tags_every_variant_with_its_entry() -> None:
    request = SweepRequest(
        entries=["Qwen3.6-27B", "Kimi-K3"], steps=2, grid={"recipe.lr": [1e-6, 2e-6]}
    )
    planned = plan_sweep(request, "autoconfig-abc123")

    assert [name for name, _ in planned] == ["Qwen3.6-27B"] * 2 + ["Kimi-K3"] * 2
    for name, cfg in planned:
        assert cfg.group_id == "autoconfig-abc123"
        assert cfg.group_overrides[AUTOCONFIG_ENTRY_TAG] == name
        assert cfg.group_axes == ["recipe.lr"]
        assert cfg.recipe.num_rollout == 2
        assert cfg.dataset.hf_repo == "SWE-bench/SWE-smith"
        assert cfg.dataset.input_column == "problem_statement"
    assert [cfg.recipe.lr for _, cfg in planned] == [1e-6, 2e-6, 1e-6, 2e-6]


def test_launch_sweep_records_launches_and_failures(monkeypatch) -> None:
    from modal_dojo.common.train import TrainConfig

    launched: list[Any] = []

    def fake_launch(self, *, show_output=True):
        launched.append(self)
        if self.recipe.lr == 2e-6:
            raise RuntimeError("quota")
        return type(
            "Run",
            (),
            {"training_run_id": f"run-{len(launched)}", "modal_app_id": "ap-1"},
        )()

    monkeypatch.setattr(TrainConfig, "launch", fake_launch)
    result = launch_sweep(
        {"entries": ["Qwen3.6-27B"], "grid": {"recipe.lr": [1e-6, 2e-6]}},
        "autoconfig-xyz",
    )

    assert result["group_id"] == "autoconfig-xyz"
    assert result["launched"] == [
        {
            "entry": "Qwen3.6-27B",
            "overrides": {"recipe.lr": 1e-6},
            "training_run_id": "run-1",
            "modal_app_id": "ap-1",
        }
    ]
    assert result["failures"] == [
        {
            "entry": "Qwen3.6-27B",
            "overrides": {"recipe.lr": 2e-6},
            "error": "RuntimeError: quota",
        }
    ]


def test_catalogue_prices_rows_on_first_request(monkeypatch) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(
        _autoconfig, "build_catalogue", lambda s, rates, at: seen.append((rates, at))
    )

    async def none() -> list[RunSummary]:
        return []

    web = FastAPI()
    mount_autoconfig_api(
        web,
        load_run_summaries=none,
        operations=FakeOperations(),
        rates=lambda: {"x": 1},
    )
    TestClient(web).get("/api/autoconfig/catalogue")
    assert seen[0][0] == {"x": 1} and seen[0][1] is not None
