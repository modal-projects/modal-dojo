"""Harbor episodes in Harbor's own ``ModalEnvironment``.

Harbor builds the task's image and runs its sandbox (``EnvironmentConfig(
type="modal")``), and its ``Verifier`` grades the episode. ``EnvironmentSandbox``
gives agent harnesses a synchronous view of that environment.

Imports Harbor at module level; import this module lazily, after
``require_package("harbor", HARBOR_VERSION)``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import shlex
import tempfile
import uuid
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from typing import Any, TypeVar

from harbor.environments.base import BaseEnvironment
from harbor.environments.factory import EnvironmentFactory
from harbor.models.task.task import Task
from harbor.models.trial.config import EnvironmentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.verifier.verifier import (
    RewardFileEmptyError,
    RewardFileNotFoundError,
    Verifier,
    VerifierOutputParseError,
)

from modal_dojo.common.sandbox import SandboxResult, extract_tar_gz

logger = logging.getLogger(__name__)

_APP_NAME = "modal-dojo-harbor"
_T = TypeVar("_T")


class EnvironmentSandbox:
    """A synchronous view of a started Harbor environment, for harnesses.

    Harnesses run in a worker thread, so each call is scheduled on the event
    loop that owns the environment. Never call it from that loop's thread.
    """

    def __init__(self, environment: BaseEnvironment, loop: asyncio.AbstractEventLoop):
        self.environment = environment
        self._loop = loop

    def call(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run one of the environment's coroutines and wait for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def run(self, *cmd: str, timeout: int | None = None) -> SandboxResult:
        result = self.call(self.environment.exec(shlex.join(cmd), timeout_sec=timeout))
        return SandboxResult(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            returncode=result.return_code,
        )

    def write(self, path: str, data: str | bytes) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "file"
            local.write_bytes(data.encode() if isinstance(data, str) else data)
            self.call(self.environment.upload_file(local, path))

    def read(self, path: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "file"
            self.call(self.environment.download_file(path, local))
            return local.read_bytes()

    def upload_dir(self, local_dir: Path, remote_dir: str) -> None:
        self.call(self.environment.upload_dir(local_dir, remote_dir))

    def download_dir(self, remote_dir: str, local_dir: Path) -> None:
        self.call(self.environment.download_dir(remote_dir, local_dir))


def _pick_reward(rewards: dict[str, float | int], reward_key: str) -> float:
    if reward_key in rewards:
        return float(rewards[reward_key])
    if len(rewards) == 1:
        return float(next(iter(rewards.values())))
    raise VerifierOutputParseError(
        f"reward.json has no {reward_key!r} key and several values: {sorted(rewards)}"
    )


class HarborEpisode:
    """One task's started environment, plus Harbor's ``Task`` to grade it by."""

    def __init__(
        self,
        sandbox: EnvironmentSandbox,
        task: Task,
        trial_paths: TrialPaths,
        task_name: str,
    ) -> None:
        self.sandbox = sandbox
        self.task = task
        self.trial_paths = trial_paths
        self.task_name = task_name

    async def verify(self, reward_key: str) -> float:
        """Run the task's tests with Harbor's ``Verifier``; return the reward."""
        environment = self.sandbox.environment
        # Clear what the model could have planted: its own tests or reward.
        paths = EnvironmentPaths()
        await environment.exec(
            f"rm -rf {paths.tests_dir} {paths.verifier_dir}"
            f" && mkdir -p {paths.verifier_dir}"
        )
        verifier = Verifier(
            task=self.task, trial_paths=self.trial_paths, environment=environment
        )
        try:
            result = await verifier.verify()
            return _pick_reward(result.rewards or {}, reward_key)
        except (
            RewardFileNotFoundError,
            RewardFileEmptyError,
            VerifierOutputParseError,
        ) as exc:
            logger.warning("Harbor task %s: %s; scoring 0.0", self.task_name, exc)
            return 0.0


@contextlib.asynccontextmanager
async def episode_environment(
    task: dict[str, Any], *, scratch_dir: str
) -> AsyncIterator[HarborEpisode]:
    """Start the task's ``ModalEnvironment``; stop it when the episode ends.

    ``task`` is a row's label from ``HarborEnvironment``; its ``task_files``
    hold the task directory Harbor builds the image from.
    """
    with tempfile.TemporaryDirectory(prefix="harbor-episode-") as tmp:
        task_dir = Path(tmp) / "task"
        extract_tar_gz(base64.b64decode(task["task_files"]), task_dir)
        harbor_task = Task(task_dir)
        trial_paths = TrialPaths(trial_dir=Path(tmp) / "trial")
        trial_paths.mkdir()
        lifetime = task["agent_timeout_sec"] + task["verifier_timeout_sec"] + 3600
        environment = EnvironmentFactory.create_environment_from_config(
            config=EnvironmentConfig(
                type="modal",
                kwargs={"app_name": _APP_NAME, "sandbox_timeout_secs": int(lifetime)},
            ),
            environment_dir=harbor_task.paths.environment_dir,
            environment_name=harbor_task.short_name,
            session_id=f"{task['task_name']}__{uuid.uuid4().hex[:8]}",
            trial_paths=trial_paths,
            task_env_config=harbor_task.config.environment,
        )
        await environment.start(force_build=False)
        try:
            await environment.exec(f"mkdir -p {shlex.quote(scratch_dir)}")
            sandbox = EnvironmentSandbox(environment, asyncio.get_running_loop())
            yield HarborEpisode(sandbox, harbor_task, trial_paths, task["task_name"])
        finally:
            try:
                await environment.stop(delete=True)
            except Exception:
                logger.exception(
                    "Harbor task %s: failed to stop its environment", task["task_name"]
                )
