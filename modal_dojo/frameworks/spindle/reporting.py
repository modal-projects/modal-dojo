"""Bridge tinker-cookbook callbacks to Dojo's existing authenticated ingestion APIs."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import logging
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import httpx

from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.framework import Framework
from modal_dojo.common.run import FrameworkStatusUpdate, TrainingRun, TrainingRunStatus
from modal_dojo.common.status import SpindleStatus
from modal_dojo.common.training_rollout import (
    TrainingRolloutResult,
    TrainingRolloutSample,
)
from modal_dojo.utils.metadata import MetadataStore, metadata_environment, vol_put

logger = logging.getLogger(__name__)
_training_rollout = ContextVar("spindle_training_rollout", default=False)


class DojoReporter:
    def __init__(self, config, deploy_config):
        self.config = config
        self.deploy_config = deploy_config
        self.groups = []
        self.next_step = 0
        # Preserve per-endpoint ordering without blocking other streams.
        self.queues = {
            endpoint: asyncio.Queue(maxsize=128)
            for endpoint in ("framework-status", "training-rollouts", "metric-points")
        }
        self.failures = 0
        self.token = secrets.token_urlsafe(32)
        now = int(time.time())
        self.run = TrainingRun(
            training_run_id="spindle-" + uuid.uuid4().hex[:12],
            framework=Framework.SPINDLE,
            framework_status=SpindleStatus.INITIALIZING,
            config={
                "model": {"model_name": config.model_name},
                "recipe": {
                    key: getattr(config, key)
                    for key in (
                        "learning_rate",
                        "lora_rank",
                        "loss_fn",
                        "max_tokens",
                        "max_steps",
                        "recipe_name",
                        "renderer_name",
                    )
                },
                "dataset": {"type": type(config.dataset_builder).__name__},
            },
            source_model={"model_name": config.model_name},
            created_at=now,
            started_at=now,
            metadata={
                "log_path": config.log_path,
                "driver": "local",
                "modal_environment": deploy_config.platform["modal"]["environment"],
            },
        )

    async def __aenter__(self):
        from modal.config import config as modal_config
        from modal_dojo.cli.setup import ensure_dashboard_deployed
        from modal_dojo.common.config import modal_proxy_auth_headers
        from modal_dojo.common.dashboard import deployed_dashboard_url

        environment = self.deploy_config.platform["modal"]["environment"]
        if environment and environment != modal_config.get("environment"):
            # Dashboard setup uses process-wide Modal config; isolate the override.
            await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable,
                    "-c",
                    "from modal_dojo.cli.setup import ensure_dashboard_deployed; "
                    "ensure_dashboard_deployed()",
                ],
                env={**os.environ, "MODAL_ENVIRONMENT": environment},
                check=True,
            )
        else:
            await asyncio.to_thread(ensure_dashboard_deployed)
        url = await asyncio.to_thread(
            deployed_dashboard_url, environment_name=environment
        )
        if not url:
            raise DojoConfigError(
                "Could not connect to Dojo. Run `modal-dojo setup` in the training "
                "environment before launching Spindle."
            )
        self.run.metadata["dashboard_url"] = url.rstrip("/")
        await self.run.save(is_async=True)
        with metadata_environment(environment):
            await vol_put(
                MetadataStore.FRAMEWORK_STATUS_TOKENS,
                self.run.training_run_id,
                {"token": self.token},
                is_async=True,
            )
        self.client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            headers={
                **modal_proxy_auth_headers(),
                "Authorization": f"Bearer {self.token}",
            },
            timeout=30,
        )
        try:
            response = await self.client.post(
                "/api/framework-status",
                json={
                    "training_run_id": self.run.training_run_id,
                    "phase": SpindleStatus.INITIALIZING.value,
                    "is_active": False,
                },
            )
            response.raise_for_status()
        except BaseException as exc:
            await self.client.aclose()
            self.run.status = TrainingRunStatus.FAILED
            self.run.error_message = f"Dojo reporting preflight failed: {exc}"
            self.run.ended_at = int(time.time())
            await self.run.save(is_async=True)
            raise
        self.workers = [
            asyncio.create_task(self._worker(endpoint, queue))
            for endpoint, queue in self.queues.items()
        ]
        print(
            f"Dojo run: {url.rstrip('/')}/training/{self.run.training_run_id}",
            flush=True,
        )
        return self

    def _enqueue(self, endpoint, payload):
        try:
            self.queues[endpoint].put_nowait(
                {"training_run_id": self.run.training_run_id, **payload}
            )
        except asyncio.QueueFull:
            self.failures += 1
            logger.warning(
                "Dojo reporting queue full; raw data remains in the tinker-cookbook log path"
            )

    def _phase(self, phase: SpindleStatus, **kwargs):
        payload = {"phase": phase.value, **kwargs}
        self.run.apply_framework_status(
            FrameworkStatusUpdate(training_run_id=self.run.training_run_id, **payload)
        )
        self._enqueue("framework-status", payload)

    async def _worker(self, endpoint, queue):
        while True:
            payload = await queue.get()
            try:
                if payload is None:
                    return
                # Ingestion is idempotent by run/step.
                for attempt in range(2):
                    try:
                        response = await self.client.post(
                            f"/api/{endpoint}", json=payload
                        )
                        response.raise_for_status()
                        break
                    except (httpx.TransportError, httpx.HTTPStatusError) as error:
                        if attempt or (
                            isinstance(error, httpx.HTTPStatusError)
                            and error.response.status_code < 500
                        ):
                            raise
                        await asyncio.sleep(0.5)
            except Exception as exc:
                self.failures += 1
                logger.warning("Dojo %s reporting failed: %r", endpoint, exc)
            finally:
                queue.task_done()

    async def capture_group(self, record):
        if _training_rollout.get():
            self.groups.append(record)

    def _flush_groups(self, step, metrics=None):
        if not self.groups:
            return
        samples = []
        for group_index, record in enumerate(self.groups):
            group = record["trajectory_group"]
            for index, (trajectory, decoded, final_reward) in enumerate(
                zip(
                    group["trajectories_G"],
                    record["decoded"],
                    group["final_rewards_G"],
                    strict=True,
                )
            ):
                # Retain all transitions and token scores for multi-turn episodes.
                samples.append(
                    TrainingRolloutSample(
                        rollout_index=len(samples),
                        sample_index=len(samples),
                        group_index=group_index,
                        prompt=decoded[0]["prompt"] if decoded else "",
                        response="\n\n".join(turn["response"] for turn in decoded),
                        score=float(
                            final_reward
                            + sum(t["reward"] for t in trajectory["transitions"])
                        ),
                        metadata={
                            "spindle_trajectory": trajectory,
                            "decoded_transitions": decoded,
                            "final_reward": final_reward,
                            "tags": record["tags"],
                            "metrics": group.get(
                                "metrics_G", [{}] * len(record["decoded"])
                            )[index],
                        },
                    )
                )
        result = TrainingRolloutResult(
            training_run_id=self.run.training_run_id,
            rollout_id=step,
            created_at=int(time.time()),
            samples=samples,
            metrics=metrics or {},
        )
        self.run.record_latest_rollout(result)
        self._enqueue("training-rollouts", result.model_dump(mode="json"))
        self.groups.clear()

    def log_metrics(self, metrics, step=None):
        # Only training metrics establish a batch boundary.
        if "progress/batch" not in metrics:
            return
        step = int(metrics["progress/batch"])
        scalars = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(value)
        }
        self.run.metrics = scalars
        self.next_step = step + 1
        self._flush_groups(step, scalars)
        self._enqueue(
            "metric-points",
            {
                "points": [
                    {
                        "step": step,
                        "metrics": scalars,
                        "time": time.time(),
                        "step_key": "progress/batch",
                    }
                ],
                "final": True,
            },
        )
        self._phase(
            self.run.framework_status,
            is_active=True,
            progress_current=self.next_step,
            progress_total=self.config.max_steps,
            progress_unit="step",
            rollout_id=step,
        )

    @contextmanager
    def tinker_cookbook_hooks(self, rl_train):
        original_log = rl_train.ml_log
        operations = {
            phase: getattr(rl_train, phase.value)
            for phase in (
                SpindleStatus.DO_GROUP_ROLLOUT_AND_FILTER_CONSTANT_REWARD,
                SpindleStatus.TRAIN_STEP,
                SpindleStatus.SAVE_CHECKPOINT_AND_GET_SAMPLING_CLIENT,
            )
        }
        reporter = self

        class Logger(original_log.Logger):
            def log_hparams(self, config):
                pass

            def log_metrics(self, metrics, step=None):
                reporter.log_metrics(metrics, step)

        def setup_logging(*args, **kwargs):
            result = original_log.setup_logging(*args, **kwargs)
            result.loggers.append(Logger())
            return result

        def wrap_operation(operation, phase):
            @wraps(operation)
            async def wrapped(*args, **kwargs):
                if reporter.run.framework_status != phase:
                    reporter._phase(phase, is_active=True)
                token = _training_rollout.set(
                    phase is SpindleStatus.DO_GROUP_ROLLOUT_AND_FILTER_CONSTANT_REWARD
                )
                try:
                    return await operation(*args, **kwargs)
                finally:
                    _training_rollout.reset(token)

            return wrapped

        rl_train.ml_log = SimpleNamespace(
            **{**vars(original_log), "setup_logging": setup_logging}
        )
        for phase, operation in operations.items():
            setattr(rl_train, phase.value, wrap_operation(operation, phase))
        try:
            yield
        finally:
            rl_train.ml_log = original_log
            for phase, operation in operations.items():
                setattr(rl_train, phase.value, operation)

    async def __aexit__(self, exc_type, exc, tb):
        self._flush_groups(self.next_step)
        try:

            async def drain():
                for queue in self.queues.values():
                    await queue.put(None)
                await asyncio.gather(*self.workers)

            await asyncio.wait_for(drain(), timeout=90)
        except TimeoutError:
            self.failures += sum(q.qsize() for q in self.queues.values()) + 1
            for worker in self.workers:
                worker.cancel()
            await asyncio.gather(*self.workers, return_exceptions=True)
            logger.warning(
                "Dojo reporting drain timed out; raw data remains in the tinker-cookbook log path"
            )
        finally:
            await self.client.aclose()
        self.run.status = (
            TrainingRunStatus.COMPLETED
            if exc is None
            else TrainingRunStatus.CANCELLED
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
            else TrainingRunStatus.FAILED
        )
        self.run.error_message = str(exc) if exc else None
        self.run.ended_at = int(time.time())
        self.run.duration_seconds = self.run.ended_at - self.run.started_at
        if exc is None:
            self.run.completed_at = self.run.ended_at
        self.run.metadata["reporting_failures"] = self.failures
        checkpoints = Path(self.config.log_path) / "checkpoints.jsonl"
        if checkpoints.exists():
            try:
                self.run.metadata["spindle_checkpoints"] = [
                    json.loads(line)
                    for line in checkpoints.read_text().splitlines()
                    if line.strip()
                ]
            except (OSError, ValueError) as error:
                self.run.metadata["checkpoint_read_error"] = str(error)
                logger.warning(
                    "Could not read tinker-cookbook checkpoint records: %s", error
                )
        try:
            await self.run.save(is_async=True)
        except Exception:
            if exc is None:
                raise
            logger.exception("Could not persist terminal Spindle run status")
