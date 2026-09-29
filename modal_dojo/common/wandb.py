"""Weights & Biases run metadata.

Pure data — each framework config writes its own converter from this to its
specific CLI flags (e.g. SlimeRecipe emits `--wandb-project`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar
from urllib.parse import quote

from modal_dojo.common.metrics import MetricConfig


@dataclass
class WandbConfig(MetricConfig):
    """Weights & Biases run metadata and credentials.

    Args:
        project:
            W&B project name.
        entity:
            W&B entity or team slug. The API key's default entity applies when
            empty.
        group:
            W&B group tag for related runs.
        exp_name:
            W&B run display name.
        key:
            W&B API key. ``WANDB_API_KEY`` takes precedence.
        disable_random_suffix:
            Preserve the configured run name.
        modal_wandb_secret_name:
            Modal secret containing the W&B API key.
    """

    project: str = ""
    entity: str = ""
    group: str = ""
    exp_name: str = ""
    key: str = ""
    disable_random_suffix: bool = True
    modal_wandb_secret_name: str = "wandb-secret"

    provider: ClassVar[str] = "wandb"

    def runtime_env(self, *, run_id: str, entity: str = "") -> dict[str, str]:
        env = super().runtime_env(run_id=run_id, entity=entity)
        if key := os.environ.get("WANDB_API_KEY", "") or self.key:
            env["WANDB_API_KEY"] = key
        if run_id:
            env.update(WANDB_RUN_ID=run_id, WANDB_RESUME="allow")
        if entity:
            env["WANDB_ENTITY"] = entity
        return env

    def url(self, *, entity: str = "", run_id: str = "") -> str | None:
        entity = (entity or self.entity).strip()
        project = self.project.strip()
        if not entity or not project:
            return None
        base = f"https://wandb.ai/{quote(entity, safe='')}/{quote(project, safe='')}"
        return f"{base}/runs/{quote(run_id, safe='')}" if run_id else base


def apply_wandb_image(image: Any) -> Any:
    return image.uv_pip_install("wandb==0.28.1")


def preflight_wandb(wandb_cfg: WandbConfig) -> str:
    """Return the resolved W&B entity for constructing deep links."""
    key = os.environ.get("WANDB_API_KEY", "") or wandb_cfg.key
    if not key:
        raise RuntimeError(
            "W&B logging is enabled but no WANDB_API_KEY is available - add it "
            f"to Modal secret '{wandb_cfg.modal_wandb_secret_name}', set "
            "metrics.key, or drop metrics= to disable logging."
        )

    import wandb

    project = wandb_cfg.project or "uncategorized"
    entity = wandb_cfg.entity or os.environ.get("WANDB_ENTITY", "")
    try:
        wandb.login(key=key, verify=True, relogin=True)
        probe = wandb.init(
            project=project,
            entity=entity or None,
            name="_preflight",
            settings=wandb.Settings(silent=True, init_timeout=60),
        )
        entity = probe.entity
        probe_path = f"{probe.entity}/{probe.project}/{probe.id}"
        wandb.finish()
        try:
            wandb.Api(api_key=key).run(probe_path).delete()
        except Exception:
            pass
    except Exception as exc:
        raise RuntimeError(
            f"W&B pre-flight failed for project '{project}': {exc}\n"
            f"The key in Modal secret '{wandb_cfg.modal_wandb_secret_name}' "
            "cannot log there. Fix the secret or drop metrics=."
        ) from exc
    finally:
        # Ray workers must start their own services for shared-mode logging.
        # finish() closes the probe run but leaves WANDB_SERVICE inherited.
        wandb.teardown()
    return entity
