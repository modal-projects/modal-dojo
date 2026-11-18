"""Stop training after a number of rollouts, dollars, or hours."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.errors import DojoConfigError

BUDGET_ENV = "MODAL_DOJO_BUDGET"

UNBOUNDED_ROLLOUTS = 1_000_000


# TODO: add to the API reference once the API shape is final.
@exclude_from_api_reference
@dataclass(frozen=True)
class Budget:
    """When to stop training: the first limit to run out ends the run.

    The run then saves a final checkpoint. Limits are checked between rollouts.

    Attributes:
        max_rollouts: Rollouts to run.
        max_cost_usd: Estimated GPU spend at list price.
        max_hours: Wall-clock hours since the training function started.
    """

    max_rollouts: int | None = None
    max_cost_usd: float | None = None
    max_hours: float | None = None

    def __post_init__(self) -> None:
        limits = {
            "max_rollouts": self.max_rollouts,
            "max_cost_usd": self.max_cost_usd,
            "max_hours": self.max_hours,
        }
        if all(value is None for value in limits.values()):
            raise DojoConfigError(
                "Budget needs at least one of max_rollouts, max_cost_usd, max_hours"
            )
        for name, value in limits.items():
            if value is not None and not (math.isfinite(value) and value > 0):
                raise DojoConfigError(f"Budget.{name} must be positive, got {value!r}")
        if self.max_rollouts is not None and self.max_rollouts != int(
            self.max_rollouts
        ):
            raise DojoConfigError(
                f"Budget.max_rollouts must be a whole number, got {self.max_rollouts!r}"
            )

    @property
    def num_rollout(self) -> int:
        return (
            int(self.max_rollouts)
            if self.max_rollouts is not None
            else UNBOUNDED_ROLLOUTS
        )

    def driver_env(
        self, *, started_at: float, gpus: int, gpu_hour_usd: float | None
    ) -> dict[str, str]:
        usd_per_second = None
        if self.max_cost_usd is not None:
            if gpu_hour_usd is None or gpus < 1:
                raise DojoConfigError(
                    "Budget.max_cost_usd needs a list price for the run's GPU "
                    "type, and the workspace's billing rates have none."
                )
            usd_per_second = gpus * gpu_hour_usd / 3600
        state = {
            "started_at": started_at,
            "max_seconds": None if self.max_hours is None else self.max_hours * 3600,
            "max_cost_usd": self.max_cost_usd,
            "usd_per_second": usd_per_second,
        }
        return {BUDGET_ENV: json.dumps(state)}


def exhausted(state: dict[str, Any], now: float) -> str | None:
    elapsed = now - state["started_at"]
    if state.get("max_seconds") is not None and elapsed >= state["max_seconds"]:
        return f"time: {elapsed / 3600:.2f} of {state['max_seconds'] / 3600:g} hours"
    if state.get("max_cost_usd") is not None:
        spent = elapsed * state["usd_per_second"]
        if spent >= state["max_cost_usd"]:
            return f"cost: ${spent:.2f} of ${state['max_cost_usd']:g}"
    return None


def budget_final(args: Any, rollout_id: int) -> bool:
    """Whether to save and stop after ``rollout_id``; called by the patched loop."""
    raw = os.environ.get(BUDGET_ENV)
    if not raw:
        return False
    if rollout_id >= args.num_rollout - 1:
        print(f"[modal-dojo] Budget: rollout {rollout_id} is the last; saving.")
        return True
    reason = exhausted(json.loads(raw), time.time())
    if reason is None:
        return False
    print(
        f"[modal-dojo] Budget exhausted ({reason}) after rollout {rollout_id}; "
        "saving a final checkpoint and stopping."
    )
    return True


def driver_budget_env(recipe: Any, *, started_at: float) -> dict[str, str]:
    budget = getattr(recipe, "budget", None)
    if budget is None:
        return {}
    gpu_hour_usd = None
    if budget.max_cost_usd is not None:
        from modal_dojo.common.derived_metrics import billing_rates, gpu_hour_cost

        gpu_hour_usd = gpu_hour_cost(billing_rates(), recipe.gpu_type)
    return budget.driver_env(
        started_at=started_at,
        gpus=recipe.gpu_allocation.total_gpus,
        gpu_hour_usd=gpu_hour_usd,
    )


def validate_recipe_budget(recipe: Any) -> None:
    budget = getattr(recipe, "budget", None)
    if budget is None or budget.max_rollouts is None:
        return
    if recipe.num_rollout != budget.max_rollouts:
        raise DojoConfigError(
            f"num_rollout={recipe.num_rollout} disagrees with "
            f"budget.max_rollouts={budget.max_rollouts}; set only the budget"
        )
