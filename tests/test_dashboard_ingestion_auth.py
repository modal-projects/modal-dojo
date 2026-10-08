"""Ingestion-route authorization."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from modal_dojo import _dashboard
from modal_dojo.common.framework import Framework
from modal_dojo.common.run import TrainingRun
from modal_dojo.utils import metadata
from modal_dojo.utils.metadata import MetadataStore

RUN_ID = "run-auth-1"
TOKEN = "test-status-token-run-auth-1"


# Each ingestion POST reads training_run_id off a *different* pydantic model,
# so they are parametrized rather than sampled: an auth check wired to the
# wrong handler's variable is the mistake this catches.
ROUTES = [
    ("/api/framework-status", {"phase": "training"}),
    ("/api/training-rollouts", {"rollout_id": 1}),
    ("/api/advantage-distributions", {"rollout_id": 1}),
    ("/api/metric-points", {"points": [{"step": 1, "metrics": {"a": 1.0}}]}),
]
ROUTE_IDS = [route.rsplit("/", 1)[-1] for route, _ in ROUTES]


def _client(monkeypatch, tmp_path) -> TestClient:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("ok")
    (static / "favicon.svg").write_text("<svg/>")
    monkeypatch.setattr(_dashboard, "STATIC_DIR", str(static))
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    return TestClient(_dashboard.fastapi_app.local())


def _save_run(token: str | None = TOKEN) -> None:
    TrainingRun(
        training_run_id=RUN_ID,
        modal_app_id="ap-auth",
        framework=Framework.SLIME,
        config={"model": {"model_name": "Qwen/Qwen3-4B"}},
        created_at=100,
        started_at=100,
        updated_at=150,
    ).save()
    if token is not None:
        metadata.vol_put(
            MetadataStore.FRAMEWORK_STATUS_TOKENS, RUN_ID, {"token": token}
        )


@pytest.mark.parametrize(("route", "extra"), ROUTES, ids=ROUTE_IDS)
def test_every_rejection_is_indistinguishable(
    fake_volume, monkeypatch, tmp_path, route, extra
):
    """Rejections must not reveal whether a run id exists.

    The handlers used to look the run up (404) before checking the token
    (403), so an anonymous caller could enumerate real training-run ids by
    the status code alone. Asserting the responses are *equal* — rather than
    each being 403 — is what pins the property: it fails just as loudly if a
    future change makes unknown runs 404 again, or gives any one failure
    mode a distinguishing body.
    """
    _save_run()
    rejections = {
        "unknown run, no credentials": ("ghost-run", {}),
        "unknown run, valid-looking token": ("ghost-run", f"Bearer {TOKEN}"),
        "real run, no credentials": (RUN_ID, {}),
        "real run, wrong token": (RUN_ID, "Bearer wrong-token"),
        "real run, Basic instead of Bearer": (RUN_ID, "Basic dHJhaW5pbmctZ3ltOnB3"),
        "real run, bare Bearer": (RUN_ID, "Bearer"),
    }

    seen = {}
    with _client(monkeypatch, tmp_path) as client:
        for label, (run_id, auth) in rejections.items():
            response = client.post(
                route,
                json={"training_run_id": run_id, **extra},
                headers={"Authorization": auth} if isinstance(auth, str) else {},
            )
            seen[label] = (response.status_code, response.json())

    refused = (403, {"detail": "Invalid status token"})
    assert seen == dict.fromkeys(rejections, refused)


@pytest.mark.parametrize(("route", "extra"), ROUTES, ids=ROUTE_IDS)
def test_correct_token_still_reaches_the_handler(
    fake_volume, monkeypatch, tmp_path, route, extra
):
    """The ingestion path every training container depends on."""
    _save_run()
    with _client(monkeypatch, tmp_path) as client:
        response = client.post(
            route,
            json={"training_run_id": RUN_ID, **extra},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize("stored", [None, ""], ids=["no-record", "empty-token"])
def test_run_with_no_token_on_record_refuses_the_dummy(
    fake_volume, monkeypatch, tmp_path, stored
):
    """The dummy the no-token path compares against must never authenticate.

    That path exists so a missing token record costs the same comparison as
    a wrong token; the constant is public, so a refactor that let it match
    would hand every tokenless run a skeleton key.
    """
    _save_run(token=stored)
    with _client(monkeypatch, tmp_path) as client:
        response = client.post(
            "/api/framework-status",
            json={"training_run_id": RUN_ID, "phase": "training"},
            headers={"Authorization": f"Bearer {_dashboard._MISSING_TOKEN_DUMMY}"},
        )

    assert response.status_code == 403


def test_payload_validation_still_runs_after_auth(fake_volume, monkeypatch, tmp_path):
    """Authenticating first must not swallow the handler's own 400s."""
    _save_run()
    with _client(monkeypatch, tmp_path) as client:
        response = client.post(
            "/api/framework-status",
            json={"training_run_id": RUN_ID, "phase": "not-a-phase"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 400
