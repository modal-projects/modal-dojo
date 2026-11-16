"""Training budgets: stop after so many rollouts, dollars, or hours.

A ``Budget`` rides on the recipe. The launcher turns it into the
``MODAL_DOJO_BUDGET`` environment variable for the training driver, and a
patch on the framework's driver loop (``budget_stop_patch``) asks
``budget_final`` after every rollout whether to stop. When it says yes, the
loop takes its usual final-rollout path, which saves a checkpoint
synchronously, and exits.
"""

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

#: ``num_rollout`` for a budget with no rollout limit: training stops on cost
#: or time instead.
UNBOUNDED_ROLLOUTS = 1_000_000


# TODO: add a reference page once the Environment / ManagedTrainConfig / Budget
# API shape is final; until then these stay out of the generated SDK reference.
@exclude_from_api_reference
@dataclass(frozen=True)
class Budget:
    """How much training to do. Training stops when any limit runs out.

    Whichever limit runs out first, the run finishes the rollout in progress,
    saves one last checkpoint, and stops. Limits are checked between
    rollouts, so a run overshoots by up to one rollout plus the final save.

    ```python
    Budget(max_rollouts=200)                    # a fixed number of steps
    Budget(max_cost_usd=50)                     # as many as $50 buys
    Budget(max_hours=6, max_cost_usd=100)       # whichever comes first
    ```

    Attributes:
        max_rollouts: Rollouts (training steps) to run.
        max_cost_usd: Estimated spend: the run's training GPUs at the
            workspace's list price per GPU-hour, from when its training
            function starts. Sandboxes, CPU, and storage are not counted.
        max_hours: Wall-clock hours from when the training function starts.
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
        """The recipe's ``num_rollout`` under this budget."""
        return (
            int(self.max_rollouts)
            if self.max_rollouts is not None
            else UNBOUNDED_ROLLOUTS
        )

    def driver_env(
        self, *, started_at: float, gpus: int, gpu_hour_usd: float | None
    ) -> dict[str, str]:
        """The driver's ``MODAL_DOJO_BUDGET`` for a run started at ``started_at``."""
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
    """Which time or cost limit ``state`` has reached at ``now``, if any."""
    elapsed = now - state["started_at"]
    if state.get("max_seconds") is not None and elapsed >= state["max_seconds"]:
        return f"time: {elapsed / 3600:.2f} of {state['max_seconds'] / 3600:g} hours"
    if state.get("max_cost_usd") is not None:
        spent = elapsed * state["usd_per_second"]
        if spent >= state["max_cost_usd"]:
            return f"cost: ${spent:.2f} of ${state['max_cost_usd']:g}"
    return None


def budget_final(args: Any, rollout_id: int) -> bool:
    """Whether ``rollout_id`` is the run's last; called by the patched driver.

    True on the budget's last rollout, or once its time or cost runs out.
    Always False for a run without a budget, which keeps its own behavior.
    """
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
    """The training driver's budget env vars for ``recipe``; empty without one.

    Called in the training function, which prices the run's GPUs at the
    workspace's list rate when the budget has a cost limit.
    """
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
    """A recipe's ``num_rollout`` must match its budget's rollout limit."""
    budget = getattr(recipe, "budget", None)
    if budget is None or budget.max_rollouts is None:
        return
    if recipe.num_rollout != budget.max_rollouts:
        raise DojoConfigError(
            f"num_rollout={recipe.num_rollout} disagrees with "
            f"budget.max_rollouts={budget.max_rollouts}; set only the budget"
        )
