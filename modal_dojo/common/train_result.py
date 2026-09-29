"""Write ``MetadataStore.TRAIN_RESULTS`` for the dashboard."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any

from modal_dojo.common.framework import Framework
from modal_dojo.utils.metadata import (
    MetadataStore,
    vol_put_with_summary,
)

TRAIN_RESULTS_STORE_NAME = MetadataStore.TRAIN_RESULTS.value


def train_result_payload(
    *,
    app_name: str,
    framework: Framework,
    training_run_id: str,
    checkpoint_dir: str = "",
    checkpoints_volume_name: str = "",
    checkpoints_mount_path: str = "",
    model_config: Any = None,
    metrics: dict[str, Any] | None = None,
    group_id: str = "",
) -> dict[str, Any]:
    if isinstance(model_config, dict):
        model_name = str(model_config.get("model_name") or "")
        model_path = model_config.get("model_path")
    elif model_config is not None:
        model_name = model_config.model_name or ""
        model_path = model_config.model_path
    else:
        model_name = ""
        model_path = None
    return {
        "app_name": app_name,
        "framework": framework.value if isinstance(framework, Framework) else framework,
        "training_run_id": training_run_id,
        "checkpoint_dir": checkpoint_dir,
        "checkpoints_volume_name": checkpoints_volume_name,
        "checkpoints_mount_path": checkpoints_mount_path,
        "model_config": {"model_name": model_name, "model_path": model_path},
        "metrics": metrics or {},
        "group_id": group_id,
    }


def save_train_result_blob(
    payload: dict[str, Any], *, is_async: bool = False
) -> None | Awaitable[None]:
    return vol_put_with_summary(
        MetadataStore.TRAIN_RESULTS,
        str(payload["training_run_id"]),
        payload,
        summary_store=MetadataStore.TRAIN_RESULTS_SUMMARY,
        item_id_key="training_run_id",
        sort_key=lambda item: str(item.get("training_run_id", "")),
        reverse=True,
        is_async=is_async,
    )
