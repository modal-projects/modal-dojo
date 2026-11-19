"""Harbor's own code running in a modal-dojo ``Sandbox``.

``SandboxEnvironment`` gives Harbor's ``Verifier`` and Terminus 2's
``TmuxSession`` the environment interface they expect.

Imports Harbor at module level; import this module lazily, after
``require_package("harbor", HARBOR_VERSION)``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import shlex
import tempfile
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.task.task import Task
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.verifier.verifier import (
    RewardFileEmptyError,
    RewardFileNotFoundError,
    Verifier,
    VerifierOutputParseError,
)

from modal_dojo.common.sandbox import Sandbox, extract_tar_gz

logger = logging.getLogger(__name__)

_ROOT_USERS = (None, "root", 0)


def _write_local(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class SandboxEnvironment(BaseEnvironment):
    """Harbor's environment interface over an already started ``Sandbox``."""

    def __init__(
        self,
        sb: Sandbox,
        *,
        name: str,
        env_config: EnvironmentConfig,
        timeout_sec: int,
        trial_paths: TrialPaths | None = None,
    ) -> None:
        """``trial_paths`` is where the ``Verifier`` downloads its logs."""
        self._sb = sb
        self._timeout_sec = timeout_sec
        super().__init__(
            environment_dir=Path(name) / "environment",  # unused: already built
            environment_name=name,
            session_id=name,
            trial_paths=trial_paths or TrialPaths(trial_dir=Path(name)),
            task_env_config=env_config,
        )

    @staticmethod
    def type() -> str:
        return "modal-dojo-sandbox"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities()

    def _validate_definition(self) -> None:
        """Nothing to check: the sandbox already runs the task's image."""

    async def start(self, force_build: bool) -> None:
        """The caller owns the sandbox's lifecycle."""

    async def stop(self, delete: bool) -> None:
        """The caller owns the sandbox's lifecycle."""

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        data = await asyncio.to_thread(Path(source_path).read_bytes)
        await asyncio.to_thread(self._sb.write, target_path, data)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await asyncio.to_thread(self._sb.upload_dir, Path(source_dir), target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        data = await asyncio.to_thread(self._sb.read, source_path)
        await asyncio.to_thread(_write_local, Path(target_path), data)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await asyncio.to_thread(self._sb.download_dir, source_dir, Path(target_dir))

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if user not in _ROOT_USERS:
            raise NotImplementedError(f"Running as user {user!r} is not supported")
        lines = [
            f"export {k}={shlex.quote(v)}"
            for k, v in (self._merge_env(env) or {}).items()
        ]
        if cwd:
            lines.append(f"cd {shlex.quote(cwd)}")
        lines.append(command)
        result = await asyncio.to_thread(
            self._sb.run,
            "bash",
            "-c",
            "\n".join(lines),
            timeout=timeout_sec or self._timeout_sec,
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )


def _pick_reward(rewards: dict[str, float | int], reward_key: str) -> float:
    if reward_key in rewards:
        return float(rewards[reward_key])
    if len(rewards) == 1:
        return float(next(iter(rewards.values())))
    raise VerifierOutputParseError(
        f"reward.json has no {reward_key!r} key and several values: {sorted(rewards)}"
    )


def verify(sb: Sandbox, task_files: str, *, task_name: str, reward_key: str) -> float:
    """Run the task's tests in ``sb`` with Harbor's ``Verifier``; return the reward.

    ``task_files`` is the base64 tarball of the task directory that
    ``HarborEnvironment`` puts in each row's label.
    """
    with tempfile.TemporaryDirectory(prefix="harbor-verify-") as tmp:
        task_dir = Path(tmp) / "task"
        extract_tar_gz(base64.b64decode(task_files), task_dir)
        (task_dir / "environment").mkdir(exist_ok=True)
        task = Task(task_dir)
        trial_paths = TrialPaths(trial_dir=Path(tmp) / "trial")
        trial_paths.mkdir()

        # Clear what the model could have planted: its own tests or reward.
        paths = EnvironmentPaths()
        sb.run(
            "bash",
            "-c",
            f"rm -rf {paths.tests_dir} {paths.verifier_dir}"
            f" && mkdir -p {paths.verifier_dir}",
        )
        environment = SandboxEnvironment(
            sb,
            name=task_name,
            env_config=task.config.environment,
            timeout_sec=int(task.config.verifier.timeout_sec),
            trial_paths=trial_paths,
        )
        verifier = Verifier(task=task, trial_paths=trial_paths, environment=environment)
        try:
            result = asyncio.run(verifier.verify())
            return _pick_reward(result.rewards or {}, reward_key)
        except (
            RewardFileNotFoundError,
            RewardFileEmptyError,
            VerifierOutputParseError,
        ) as exc:
            logger.warning("Harbor task %s: %s; scoring 0.0", task_name, exc)
            return 0.0
