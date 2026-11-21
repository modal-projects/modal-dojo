from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

AGENT_CONFIG = {
    "custom_generate_function_path": "miles.rollout.generate_hub.agentic_tool_call.generate",
    "custom_agent_function_path": f"{__name__}.run",
    # agent function mode could be either inline or subproc
    # subproc will create a process for each rollout which will hog up host RAM
    # inline is not ideal either because sync work on one process will block all other processes
    # for now, inline is the better option, though we should move towards a hybrid future
    "custom_agent_function_mode": "inline",
    "custom_rm_path": f"{__name__}.reward",
    # TODO: revisit session server v2 when thinking about compaction (ss v2)
    "use_session_server": True,
    # miles has a 120s timeout to collect finished sessions from session server
    # with one worker, sessions that finish together queue behind it and hit the timeout
    # so, we use multiple workers to distribute the session collection work
    "session_server_workers": 8,
    "session_server_ip": "0.0.0.0",
    "session_message_matcher": "loose_tool_call",
}

# context manager that keeps each tunnel alive until the rollout process exits
_tunnels: dict[int, tuple[Any, str]] = {}


def _public_url(base_url: str) -> str:
    import modal

    url = urlsplit(base_url)
    if url.port not in _tunnels:
        tunnel = modal.forward(url.port)
        _tunnels[url.port] = tunnel, tunnel.__enter__().url
    return _tunnels[url.port][1] + url.path


async def run(base_url: str, metadata: dict[str, Any], **kwargs: Any) -> dict:
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
    )
    from harbor.trial.trial import Trial

    with tempfile.TemporaryDirectory() as trials_dir:
        trial = await Trial.create(
            TrialConfig(
                task=TaskConfig(path=Path(metadata["harbor_task_path"])),
                agent=AgentConfig(
                    name=metadata["harbor_agent"],
                    model_name="openai/policy",
                    base_url=_public_url(base_url) + "/v1",
                    model_api_format="openai_chat_completions",
                    env={"OPENAI_API_KEY": "unused"},
                ),
                environment=EnvironmentConfig(type="modal"),
                trials_dir=Path(trials_dir),
            )
        )
        # TODO: figure out best way to provide a time bound for this
        result = await trial.run()

    rewards = result.verifier_result.rewards if result.verifier_result else {}
    reward = rewards.get("reward", None)
    exc = result.exception_info
    error = f"{exc.exception_type}: {exc.exception_message}" if exc else None
    if error:
        logger.warning("Harbor trial %s: %s", result.trial_name, error)
    return {"reward": reward, "error": error}


async def reward(args: Any, samples: Any, **kwargs: Any) -> float | list:
    if isinstance(samples, list):
        return [sample.metadata.get("reward", 0.0) for sample in samples]
    return samples.metadata.get("reward", 0.0)
