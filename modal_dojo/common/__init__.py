"""Common cross-framework utilities and compatibility re-exports.

Exports shared constants + helpers used by every framework package. Each
framework's launcher merges ``COMMON_TRAINING_GYM_TAGS`` with its own
``_modal_framework`` tag and the user's ``config.app_tags`` when constructing
its ``modal.App``.
"""

from __future__ import annotations

import os
import re

from modal_dojo.utils.gpu import GPUType
from modal_dojo.common.modal_refs import (
    ModalCaptureError,
    register_modal_cloudpickle_reducers,
)
from modal_dojo.utils.metadata import (
    MetadataStore,
    vol_get,
    vol_list,
    vol_put,
)

COMMON_TRAINING_GYM_TAGS: dict[str, str] = {
    "training": "True",
    "source": "training-gym",
    "_modal_job_type": "training",
}


def modal_tag_value(value: object) -> str:
    raw_name = str(value).rsplit("/", 1)[-1].lower()
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw_name).strip("-_.")


def hf_token() -> str | None:
    """Return ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN`` from the launching process."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return token or None


def hf_secrets() -> list:
    """Return a list of Modal Secrets providing ``HF_TOKEN`` to containers.

    Priority:
      1. Local ``HF_TOKEN`` env var → ``Secret.from_dict`` (also silences
         the local-side ``huggingface_hub`` "unauthenticated" warning).
      2. Named Modal secret ``huggingface-secret`` → used if it exists in
         the workspace.
      3. Empty list → container runs without an HF token.
    """
    from modal import Secret

    token = hf_token()
    if token:
        return [Secret.from_dict({"HF_TOKEN": token})]

    try:
        secret = Secret.from_name("huggingface-secret")
        secret.hydrate()
        return [secret]
    except Exception:
        return []


def proxy_auth_secrets() -> list:
    """Return a Modal Secret with ``MODAL_KEY`` / ``MODAL_SECRET`` for train workers.

    Custom deployments can sit behind Modal proxy auth.
    Driver-shell env does not reach Ray rollout actors, so frameworks attach this
    secret to the train function the same way they attach wandb / HF secrets.
    Loads from env or ``~/.training-gym.toml`` via :func:`load_proxy_auth`.
    Returns ``[]`` when the pair is unset (callers that hit proxy-auth endpoints
    will then get 401).
    """
    from modal import Secret

    from modal_dojo.common.config import load_proxy_auth

    if not load_proxy_auth():
        return []
    key = os.environ.get("MODAL_KEY", "").strip()
    secret = os.environ.get("MODAL_SECRET", "").strip()
    if not (key and secret):
        return []
    return [Secret.from_dict({"MODAL_KEY": key, "MODAL_SECRET": secret})]


__all__ = [
    "COMMON_TRAINING_GYM_TAGS",
    "GPUType",
    "MetadataStore",
    "ModalCaptureError",
    "modal_tag_value",
    "register_modal_cloudpickle_reducers",
    "hf_secrets",
    "hf_token",
    "proxy_auth_secrets",
    "vol_get",
    "vol_list",
    "vol_put",
]
