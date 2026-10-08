"""Autoconfig: the dashboard's recipe catalogue and the sweeps that fill it.

The page reads ``GET /api/autoconfig/catalogue`` and kicks off sweeps with
``POST /api/autoconfig/sweeps``, which spawns ``launch_autoconfig_sweep`` in
the dashboard's Modal app and answers with the call id. The page then polls
``GET /api/autoconfig/sweeps/{operation_id}`` until the launch finishes, the
same submit-and-poll shape as autoinference's recipe-seed API.
"""

from __future__ import annotations

import datetime as dt
import re
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from modal_dojo.common.catalogue import (
    AUTOCONFIG_ENTRY_TAG,
    AUTOCONFIG_GROUP_PREFIX,
    CATALOGUE,
    STEPS,
    build_catalogue,
    find_entry,
)
from modal_dojo.common.run_summary import RunSummary

SWEEP_FUNCTION_NAME = "launch_autoconfig_sweep"
RATES_TTL_S = 3600.0
# ``modal.FunctionCall.from_id`` accepts any id; only ours are sweeps.
_CALL_ID = re.compile(r"^fc-[A-Za-z0-9]+$")


class DatasetSpec(BaseModel):
    """A ``HuggingFaceDataset`` the sweep trains on."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    hf_repo: str = Field(min_length=1)
    hf_split: str = Field(default="train", min_length=1)
    hf_config: str | None = None
    input_column: str = Field(min_length=1)
    output_column: str | None = None
    input_format: Literal["text", "messages", "raw"] = "text"


# The catalogue benchmarks recipes on SWE-bench-style tasks: SWE-smith's
# training split pairs each problem statement with the patch that fixed it.
DEFAULT_DATASET = DatasetSpec(
    hf_repo="SWE-bench/SWE-smith",
    hf_split="train",
    input_column="problem_statement",
    output_column="patch",
)


class SweepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Catalogue entry names to benchmark.
    entries: list[str] = Field(min_length=1)
    steps: int = Field(default=STEPS, ge=1, le=100)
    # ``TrainingGroup`` grid: dotted ``TrainConfig`` paths to the values to try.
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    name: str | None = Field(default=None, max_length=60)
    dataset: DatasetSpec = DEFAULT_DATASET


class SweepError(BaseModel):
    type: str
    message: str


class SweepOperation(BaseModel):
    operation_id: str
    group_id: str
    status: Literal["pending", "succeeded", "failed"]
    result: dict[str, Any] | None = None
    error: SweepError | None = None


class SweepOperations(Protocol):
    def submit(self, request: SweepRequest, group_id: str) -> SweepOperation: ...

    def get(self, operation_id: str) -> SweepOperation: ...


class SweepBackendError(RuntimeError):
    """Modal could not serve the sweep operation."""


class SweepNotFoundError(LookupError):
    """No such sweep operation."""


def sweep_group_id(name: str | None) -> str:
    """A fresh ``autoconfig-…`` group id; the page finds sweep runs by it."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    parts = [AUTOCONFIG_GROUP_PREFIX, *([slug] if slug else []), secrets.token_hex(3)]
    return "-".join(parts)


def validate_request(request: SweepRequest) -> None:
    """Reject unknown entries before anything is spawned."""
    for name in request.entries:
        find_entry(name)


def plan_sweep(request: SweepRequest, group_id: str) -> list[tuple[str, Any]]:
    """Every ``(entry name, TrainConfig)`` the sweep launches.

    Each entry's recipe class, at its defaults plus ``steps`` rollouts, is the
    base of one ``TrainingGroup`` over the request's grid. Every variant is
    tagged with the entry it benchmarks so the catalogue can find it.
    """
    import modal_dojo
    from modal_dojo.common.dataset import HuggingFaceDataset
    from modal_dojo.common.models.validation import _ValidationConfig
    from modal_dojo.common.train import TrainConfig
    from modal_dojo.common.training_group import TrainingGroup

    dataset = HuggingFaceDataset(**request.dataset.model_dump())
    configs: list[tuple[str, Any]] = []
    for name in request.entries:
        entry = find_entry(name)
        model = _ValidationConfig.find(entry.name).model_config()
        recipe = getattr(modal_dojo, entry.recipe)(num_rollout=request.steps)
        base = TrainConfig(
            model=model, dataset=dataset, eval_dataset=dataset, recipe=recipe
        )
        group = TrainingGroup(base, request.grid, name=group_id)
        for overrides, cfg in group.iter_variants():
            cfg.group_id = group_id
            cfg.group_overrides = {AUTOCONFIG_ENTRY_TAG: entry.name, **overrides}
            configs.append((entry.name, cfg))
    return configs


def launch_sweep(payload: Mapping[str, Any], group_id: str) -> dict[str, Any]:
    """Body of ``launch_autoconfig_sweep``: launch every variant, detached."""
    request = SweepRequest.model_validate(payload)
    launched: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for entry_name, cfg in plan_sweep(request, group_id):
        overrides = {
            k: v
            for k, v in (cfg.group_overrides or {}).items()
            if k != AUTOCONFIG_ENTRY_TAG
        }
        try:
            run = cfg.launch(show_output=False)
        except Exception as exc:  # noqa: BLE001 — recorded per variant
            failures.append(
                {
                    "entry": entry_name,
                    "overrides": overrides,
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            )
            continue
        launched.append(
            {
                "entry": entry_name,
                "overrides": overrides,
                "training_run_id": run.training_run_id,
                "modal_app_id": run.modal_app_id,
            }
        )
    return {"group_id": group_id, "launched": launched, "failures": failures}


class ModalSweepOperations:
    """Sweeps as spawned calls of the dashboard app's launcher function."""

    def __init__(self, function: Any) -> None:
        self._function = function

    def submit(self, request: SweepRequest, group_id: str) -> SweepOperation:
        import modal

        try:
            call = self._function.spawn(request.model_dump(mode="json"), group_id)
        except modal.exception.Error as exc:
            raise SweepBackendError("could not submit sweep") from exc
        return SweepOperation(
            operation_id=call.object_id, group_id=group_id, status="pending"
        )

    def get(self, operation_id: str) -> SweepOperation:
        import modal

        if not _CALL_ID.match(operation_id):
            raise SweepNotFoundError("sweep operation not found")
        try:
            call = modal.FunctionCall.from_id(operation_id)
            result = call.get(timeout=0)
        except modal.exception.NotFoundError as exc:
            raise SweepNotFoundError("sweep operation not found") from exc
        except TimeoutError:
            return SweepOperation(
                operation_id=operation_id, group_id="", status="pending"
            )
        except modal.exception.Error as exc:
            raise SweepBackendError("could not read sweep") from exc
        except Exception as exc:  # noqa: BLE001 — the launcher raised
            return SweepOperation(
                operation_id=operation_id,
                group_id="",
                status="failed",
                error=SweepError(type=type(exc).__name__, message=str(exc)[:500]),
            )
        if not isinstance(result, dict):
            return SweepOperation(
                operation_id=operation_id,
                group_id="",
                status="failed",
                error=SweepError(
                    type="invalid_result", message="sweep returned a non-object"
                ),
            )
        return SweepOperation(
            operation_id=operation_id,
            group_id=str(result.get("group_id") or ""),
            status="succeeded",
            result=result,
        )


def billing_rates() -> Mapping[str, Any]:
    """Current GPU list prices of the workspace."""
    import modal

    return modal.Workspace.from_context().billing.rates()


def mount_autoconfig_api(
    web: Any,
    *,
    load_run_summaries: Callable[[], Awaitable[list[RunSummary]]],
    operations: SweepOperations,
    rates: Callable[[], Mapping[str, Any]] = billing_rates,
) -> Any:
    from fastapi import HTTPException, status
    from fastapi.concurrency import run_in_threadpool

    rates_cache: dict[str, Any] = {"at": None, "rates": {}, "priced_at": None}

    def priced_rates() -> tuple[Mapping[str, Any], str | None]:
        at = rates_cache["at"]
        if at is None or time.monotonic() - at > RATES_TTL_S:
            try:
                rates_cache["rates"] = dict(rates())
                rates_cache["priced_at"] = dt.date.today().isoformat()
            except Exception as exc:  # noqa: BLE001 — unpriced rows beat no rows
                print(f"[autoconfig] billing rates unavailable: {exc}", flush=True)
            rates_cache["at"] = time.monotonic()
        return rates_cache["rates"], rates_cache["priced_at"]

    @web.get("/api/autoconfig/catalogue")
    async def catalogue():
        summaries = await load_run_summaries()
        current, priced_at = await run_in_threadpool(priced_rates)
        return await run_in_threadpool(build_catalogue, summaries, current, priced_at)

    @web.get("/api/autoconfig/entries")
    async def entries():
        return [{"name": e.name, "recipe": e.recipe} for e in CATALOGUE]

    @web.post(
        "/api/autoconfig/sweeps",
        response_model=SweepOperation,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit_sweep(request: SweepRequest) -> SweepOperation:
        try:
            validate_request(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            return await run_in_threadpool(
                operations.submit, request, sweep_group_id(request.name)
            )
        except SweepBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @web.get("/api/autoconfig/sweeps/{operation_id}", response_model=SweepOperation)
    async def get_sweep(operation_id: str) -> SweepOperation:
        try:
            return await run_in_threadpool(operations.get, operation_id)
        except SweepNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SweepBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return web
