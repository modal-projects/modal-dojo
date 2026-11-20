from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

TRIALS_DIR_ENV = "MODAL_DOJO_HARBOR_TRIALS_DIR"

AGENT_CONFIG = {
    "custom_generate_function_path": "miles.rollout.generate_hub.agentic_tool_call.generate",
    "custom_agent_function_path": f"{__name__}.run",
    "custom_agent_function_mode": "inline",  # TODO: should this be configurable?
    "custom_rm_path": f"{__name__}.reward",
    "use_session_server": True,
    "session_server_workers": 1,
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
            trials_dir=Path(os.environ[TRIALS_DIR_ENV]),
        )
    )
    result = await trial.run()
    rewards = result.verifier_result.rewards if result.verifier_result else None
    # TODO: think about how to discern between agent failures vs sandbox failures
    return {"reward": (rewards or {}).get("reward", 0.0)}


async def reward(args: Any, sample: Any, **kwargs: Any) -> float:
    return sample.metadata.get("reward", 0.0)
