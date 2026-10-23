"""``FlyteTrainingRun`` — a waitable handle around a ``flyte.Run``.

Gives the Flyte backend the same ergonomics as the Modal ``TrainingRun``:
a stable ``training_run_id``, a console URL, and a blocking ``wait()``.
"""

from __future__ import annotations

import time
from typing import Any


class FlyteTrainingRun:
    """Handle to a launched Flyte training pipeline.

    Attributes:
        training_run_id: The dojo run id (reused for object-store linkage).
        run: The underlying :class:`flyte.Run` handle.
        run_id: The Flyte execution name.
    """

    def __init__(self, training_run_id: str, run: Any) -> None:
        self.training_run_id = training_run_id
        self.run = run
        self._run_id: str | None = getattr(run, "id", None) or getattr(
            run, "name", None
        )

    @property
    def run_id(self) -> str | None:
        if self._run_id:
            return self._run_id
        self._run_id = getattr(self.run, "id", None) or getattr(self.run, "name", None)
        return self._run_id

    @property
    def url(self) -> str:
        """Flyte / Union console URL for the execution, if available."""
        url = getattr(self.run, "url", None)
        return url or f"https://console.union.ai/runs/{self.run_id or 'unknown'}"

    @property
    def status(self) -> Any:
        return getattr(self.run, "status", None)

    def wait(self, *, timeout: float | None = None) -> "FlyteTrainingRun":
        """Block until the Flyte run finishes (or ``timeout`` seconds elapse)."""
        started = time.time()
        while True:
            state = getattr(self.run, "status", None)
            if state is None:
                return self
            name = str(getattr(state, "phase", state))
            if any(tok in name for tok in ("SUCCEEDED", "FAILED", "ERROR")):
                return self
            if timeout is not None and (time.time() - started) > timeout:
                return self
            time.sleep(2)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"FlyteTrainingRun(training_run_id={self.training_run_id!r}, "
            f"run_id={self.run_id!r}, url={self.url!r})"
        )
