"""Typed public representation and compatibility helpers for run lists."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Literal, cast
from urllib.parse import quote

from pydantic import BaseModel, Field, ValidationError, model_serializer
from pydantic.fields import FieldInfo

from modal_training_gym.common.modal_urls import modal_app_dashboard_url


JsonDict = dict[str, object]
StepTimes = dict[str, dict[str, int | None]]
SubstepTimes = dict[str, dict[str, dict[str, float | int | bool | None]]]


class FrameworkProgress(BaseModel):
    current: int | None = None
    total: int | None = None
    unit: str = "step"
    phase: str = ""
    is_active: bool | None = None
    rollout_id: int | None = None
    step_id: int | None = None
    updated_at: int = 0


_STAGE_LABELS = {
    "initializing": "Initializing",
    "download_model": "Downloading model",
    "convert_model": "Converting model",
    "prepare_dataset": "Preparing dataset",
    "initialize_rollouts": "Initializing rollouts",
    "generate_rollouts": "Generating rollouts",
    "evaluate_rollouts": "Evaluating rollouts",
    "compute_log_probs": "Computing log probs",
    "optimizer_step": "Optimizer step",
    "weight_sync": "Weight sync",
    "offload_rollout": "Offload rollout",
    "offload_train": "Offload train",
    "checkpoint_save": "Saving checkpoint",
    "training": "Training",
}
_SFT_STAGE_LABELS = {
    **_STAGE_LABELS,
    "generate_rollouts": "Preparing batch",
    "weight_sync": "Training",
}
_QUEUEABLE_STAGES = {"download_model", "convert_model"}


def _display_status(status: str, *, has_train_result: bool) -> str:
    normalized = status.strip().lower()
    if has_train_result or normalized == "completed":
        return "completed"
    if normalized in {"cancelled", "stopped", "failed"}:
        return normalized
    return "pending"


def _display_stage(
    status: str,
    progress: FrameworkProgress | None,
    training_type: Literal["sft", "rl"],
) -> str:
    normalized = status.strip().lower()
    if not normalized:
        return ""
    labels = _SFT_STAGE_LABELS if training_type == "sft" else _STAGE_LABELS
    label = labels.get(normalized, normalized.replace("_", " ").title())
    if (
        normalized in _QUEUEABLE_STAGES
        and progress is not None
        and progress.is_active is False
    ):
        return f"Queuing for GPU — {label}"
    return label


def _run_list_field(
    title: str,
    *,
    default: object = ...,
    filterable: bool = False,
    timestamp: bool = False,
) -> FieldInfo:
    return Field(
        default,
        title=title,
        json_schema_extra={
            "list": True,
            "filterable": filterable,
            "timestamp": timestamp,
        },
    )


class LatestRollout(BaseModel):
    rollout_id: int = 0
    mean: float = 0.0
    total: int = 0
    created_at: int = 0


class MetricLink(BaseModel):
    label: str
    url: str
    run_id: str = ""
    attempt: int | None = None


WandbLink = MetricLink


def _wandb_label(label: object) -> str:
    text = _text(label).strip()
    if not text:
        return "W&B"
    return text.replace("Metric", "W&B", 1)


def _legacy_wandb_link_dump(links: object) -> list[JsonDict]:
    legacy_links: list[JsonDict] = []
    if not isinstance(links, list):
        return legacy_links
    for raw_link in links:
        link = _mapping(raw_link)
        if not link.get("url"):
            continue
        legacy_link = dict(link)
        legacy_link["label"] = _wandb_label(link.get("label"))
        legacy_links.append(legacy_link)
    return legacy_links


def _add_legacy_wandb_fields(data: JsonDict, *, include_group: bool) -> JsonDict:
    data["wandb_project"] = data.get("metric_project", "")
    if include_group:
        data["wandb_group"] = data.get("metric_group", "")
    data["wandb_entity"] = data.get("metric_entity", "")
    data["wandb_training_run_id"] = data.get("metric_run_id", "")
    data["wandb_url"] = data.get("metric_url")
    data["wandb_links"] = _legacy_wandb_link_dump(data.get("metric_links"))
    return data


class ConfigSummary(BaseModel):
    model_name: str = ""
    dataset_name: str = ""
    dataset_prompt_data: str = ""
    gpu_type: str = ""
    actor_num_nodes: int = 0
    actor_num_gpus_per_node: int = 0
    lr: float = 0.0
    global_batch_size: int = 0
    metric_provider: str = ""
    metric_project: str = ""
    metric_group: str = ""
    metric_entity: str = ""
    metric_run_id: str = ""
    metric_url: str | None = None
    metric_links: list[MetricLink] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> JsonDict:
        return _add_legacy_wandb_fields(handler(self), include_group=True)


class TrainResultSummary(BaseModel):
    training_run_id: str = ""
    app_name: str = ""
    checkpoint_dir: str = ""
    model_name: str = ""
    model_path: str = ""
    metric_provider: str = ""
    metric_project: str = ""
    metric_entity: str = ""
    metric_run_id: str = ""
    metric_url: str | None = None
    metric_links: list[MetricLink] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> JsonDict:
        return _add_legacy_wandb_fields(handler(self), include_group=False)


class ResumeState(BaseModel):
    attempt_count: int = 0
    resumed_from_checkpoint: bool = False
    resume_checkpoint_name: str = ""
    resume_checkpoint_path: str = ""
    resume_from_iteration: int | None = None
    last_attempt_status: str = ""
    last_attempt_started_at: int = 0
    attempt_starts: list[int] = Field(default_factory=list)


class GroupTag(BaseModel):
    key: str
    label: str = ""
    value: Any = None


class GroupTags(BaseModel):
    group_id: str = ""
    axes: list[str] = Field(default_factory=list)
    overrides: JsonDict = Field(default_factory=dict)
    tags: list[GroupTag] = Field(default_factory=list)


class RunSummary(BaseModel):
    training_run_id: str
    run_id: str = _run_list_field("Run")
    status: str = "running"
    display_status: str = _run_list_field(
        "Status",
        default="",
        filterable=True,
    )
    display_stage: str = _run_list_field(
        "Stage",
        default="",
    )
    framework: str = ""
    framework_status: str = ""
    framework_progress: FrameworkProgress | None = None
    latest_rollout: LatestRollout | None = None
    model: str = _run_list_field("Model", default="", filterable=True)
    dataset: str = _run_list_field("Dataset", default="", filterable=True)
    recipe: str = _run_list_field("Recipe", default="", filterable=True)
    training_type: Literal["sft", "rl"] = _run_list_field(
        "Training type",
        default="rl",
        filterable=True,
    )
    group_id: str = _run_list_field(
        "Group",
        default="",
        filterable=True,
    )
    group_tags: GroupTags | None = None
    modal_app_id: str = ""
    modal_app_url: str | None = None
    dataset_id: str = ""
    deployment_id: str = ""
    created_at: int = _run_list_field("Created", default=0, timestamp=True)
    started_at: int = 0
    updated_at: int = _run_list_field(
        "Last updated",
        default=0,
        timestamp=True,
    )
    ended_at: int | None = None
    completed_at: int | None = None
    duration_seconds: int | None = None
    has_train_result: bool = False
    config_summary: JsonDict | ConfigSummary = Field(default_factory=dict)
    train_result: TrainResultSummary | None = None
    metric_links: list[MetricLink] = Field(default_factory=list)
    resume_state: ResumeState | None = None

    config: JsonDict = Field(default_factory=dict)
    metadata: JsonDict | None = None
    error_message: str = ""
    step_times: StepTimes | None = None
    substep_times: SubstepTimes | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> JsonDict:
        data = handler(self)
        data["wandb_links"] = _legacy_wandb_link_dump(data.get("metric_links"))
        return data


def _unwrap(value: object) -> object:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _text(value: object) -> str:
    value = _unwrap(value)
    return "" if value is None else str(value)


def _mapping(value: object) -> JsonDict:
    value = _unwrap(value)
    return dict(value) if isinstance(value, dict) else {}


def _number(value: object, default: float = 0.0) -> float:
    value = _unwrap(value)
    if value in (None, "") or not isinstance(value, (str, int, float)):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _optional_int(value: object) -> int | None:
    value = _unwrap(value)
    if value in (None, ""):
        return None
    if not isinstance(value, (str, int, float)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed):
        return None
    return int(parsed) if parsed.is_integer() else None


def _integer(value: object, default: int = 0) -> int:
    parsed = _optional_int(value)
    return default if parsed is None else parsed


def _timestamp(value: object) -> int:
    value = _unwrap(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value) if math.isfinite(value) else 0
    if not isinstance(value, str) or not value.strip():
        return 0
    try:
        parsed_number = float(value)
    except ValueError:
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return 0
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.UTC)
        return int(parsed.timestamp())
    if not math.isfinite(parsed_number):
        return 0
    return int(parsed_number)


def _modal_app_url(modal_app_id: str) -> str | None:
    app_id = modal_app_id.strip()
    if not app_id:
        return None
    if app_id.startswith(("http://", "https://")):
        return app_id
    return modal_app_dashboard_url(app_id)


def _wandb_url(entity: object, project: object, run_id: object) -> str | None:
    clean_entity = _text(entity).strip()
    clean_project = _text(project).strip()
    clean_run_id = _text(run_id).strip()
    if not clean_entity or not clean_project:
        return None
    base = f"https://wandb.ai/{quote(clean_entity, safe='')}/{quote(clean_project, safe='')}"
    return f"{base}/runs/{quote(clean_run_id, safe='')}" if clean_run_id else base


def _metric_summary(
    *,
    entity: object,
    project: object,
    group: object = "",
    run_id: object = "",
    metric: object = None,
) -> dict[str, Any]:
    metric_data = _mapping(metric)
    provider = _text(metric_data.get("provider")).strip()
    url = _text(metric_data.get("url")).strip()
    if not url and provider in ("", "wandb"):
        url = _wandb_url(entity, project, run_id) or ""
    metric_link = (
        [MetricLink(label="Metric", url=url, run_id=_text(run_id).strip())]
        if url
        else []
    )
    return {
        "metric_provider": provider,
        "metric_project": _text(project),
        "metric_group": _text(group),
        "metric_entity": _text(entity),
        "metric_run_id": _text(run_id),
        "metric_url": url or None,
        "metric_links": metric_link,
    }


def _config_summary(config: object, training_run_id: str) -> ConfigSummary | JsonDict:
    if not isinstance(config, dict):
        return {}
    model = _mapping(config.get("model"))
    recipe = _mapping(config.get("recipe")) or _mapping(config.get("preset"))
    legacy_wandb = _mapping(config.get("wandb"))
    metric_config = _mapping(config.get("metrics")) or legacy_wandb
    dataset = _mapping(config.get("dataset"))
    dataset_name = (
        _text(dataset.get("hf_repo"))
        or _text(dataset.get("prompt_data"))
        or _text(dataset.get("name"))
    )
    metric_run_id = _text(metric_config.get("run_id")) or (
        training_run_id[:8] if legacy_wandb else ""
    )
    return ConfigSummary(
        model_name=_text(model.get("model_name")),
        dataset_name=dataset_name,
        dataset_prompt_data=_text(dataset.get("prompt_data")),
        gpu_type=_text(recipe.get("gpu_type")),
        actor_num_nodes=_integer(recipe.get("actor_num_nodes")),
        actor_num_gpus_per_node=_integer(recipe.get("actor_num_gpus_per_node")),
        lr=_number(config.get("lr")),
        global_batch_size=_integer(config.get("global_batch_size")),
        **_metric_summary(
            entity=metric_config.get("entity"),
            project=metric_config.get("project"),
            group=metric_config.get("group"),
            run_id=(
                metric_run_id
                if metric_config.get("project")
                and metric_config.get("entity")
                or metric_config.get("url")
                else ""
            ),
            metric=metric_config,
        ),
    )


def _train_result_summary(result: JsonDict) -> TrainResultSummary:
    model = _mapping(result.get("model_config"))
    metric = _mapping(result.get("metrics"))
    return TrainResultSummary(
        training_run_id=_text(result.get("training_run_id")),
        app_name=_text(result.get("app_name")),
        checkpoint_dir=_text(result.get("checkpoint_dir")),
        model_name=_text(model.get("model_name")),
        model_path=_text(model.get("model_path")),
        **_metric_summary(
            entity=metric.get("entity") or result.get("wandb_entity"),
            project=metric.get("project") or result.get("wandb_project"),
            group=metric.get("group"),
            run_id=metric.get("run_id") or result.get("wandb_training_run_id"),
            metric=metric,
        ),
    )


def _framework_progress(metadata: JsonDict) -> FrameworkProgress | None:
    progress = _mapping(metadata.get("framework_progress"))
    if not progress:
        return None
    is_active = progress.get("is_active")
    return FrameworkProgress(
        current=_optional_int(progress.get("current")),
        total=(
            total
            if (total := _optional_int(progress.get("total"))) is not None and total > 0
            else None
        ),
        unit=_text(progress.get("unit")) or "step",
        phase=_text(progress.get("phase")),
        is_active=is_active if isinstance(is_active, bool) else None,
        rollout_id=_optional_int(progress.get("rollout_id")),
        step_id=_optional_int(progress.get("step_id")),
        updated_at=_timestamp(progress.get("updated_at")),
    )


def _latest_rollout(metadata: JsonDict) -> LatestRollout | None:
    rollout = _mapping(metadata.get("latest_rollout"))
    if not rollout:
        return None
    return LatestRollout(
        rollout_id=_integer(rollout.get("rollout_id")),
        mean=_number(rollout.get("mean")),
        total=_integer(rollout.get("total")),
        created_at=_timestamp(rollout.get("created_at")),
    )


def _resume_state(metadata: JsonDict) -> ResumeState | None:
    attempt_count = _integer(metadata.get("attempt_count"))
    checkpoint_path = _text(metadata.get("resume_checkpoint_path"))
    resumed = metadata.get("resumed_from_checkpoint") is True or bool(checkpoint_path)
    raw_attempt_starts = metadata.get("attempt_starts")
    attempt_starts = (
        sorted(
            {
                parsed
                for item in raw_attempt_starts
                if (parsed := _optional_int(item)) is not None
            }
        )[-50:]
        if isinstance(raw_attempt_starts, list)
        else []
    )
    if attempt_count <= 1 and not resumed:
        return None
    return ResumeState(
        attempt_count=attempt_count,
        resumed_from_checkpoint=resumed,
        resume_checkpoint_name=_text(metadata.get("resume_checkpoint_name")),
        resume_checkpoint_path=checkpoint_path,
        resume_from_iteration=_optional_int(metadata.get("resume_from_iteration")),
        last_attempt_status=_text(metadata.get("last_attempt_status")),
        last_attempt_started_at=_timestamp(metadata.get("last_attempt_started_at")),
        attempt_starts=attempt_starts,
    )


def _group_tags(metadata: JsonDict, group_id: str) -> GroupTags | None:
    raw = _mapping(metadata.get("group_tags"))
    if not raw and not group_id:
        return None
    overrides = _mapping(raw.get("overrides"))
    axes_value = raw.get("axes")
    axes = (
        [_text(item) for item in axes_value]
        if isinstance(axes_value, list)
        else list(overrides)
    )
    raw_tags = raw.get("tags")
    tags: list[GroupTag] = []
    if isinstance(raw_tags, list):
        for item in raw_tags:
            tag = _mapping(item)
            key = _text(tag.get("key"))
            if key:
                tags.append(
                    GroupTag(
                        key=key,
                        label=_text(tag.get("label"))
                        or key.rsplit(".", 1)[-1].replace("_", " "),
                        value=tag.get("value"),
                    )
                )
    if not tags:
        tags = [
            GroupTag(
                key=key,
                label=key.rsplit(".", 1)[-1].replace("_", " "),
                value=value,
            )
            for key, value in overrides.items()
        ]
    return GroupTags(
        group_id=_text(raw.get("group_id")) or group_id,
        axes=axes,
        overrides=overrides,
        tags=tags,
    )


def _metric_attempt_links(metadata: JsonDict) -> list[MetricLink]:
    attempts = metadata.get("metric_attempts") or metadata.get("wandb_attempts")
    if not isinstance(attempts, list):
        return []
    links: list[MetricLink] = []
    for raw_attempt in attempts:
        attempt = _mapping(raw_attempt)
        attempt_number = _integer(attempt.get("attempt"))
        run_id = _text(attempt.get("run_id"))
        provider = _text(attempt.get("provider")).strip()
        url = _text(attempt.get("url")).strip()
        if not url and provider in ("", "wandb"):
            url = (
                _wandb_url(
                    attempt.get("entity"), attempt.get("project"), attempt.get("run_id")
                )
                or ""
            )
        if url:
            links.append(
                MetricLink(
                    label=f"Metric a{attempt_number}"
                    if attempt_number > 1
                    else "Metric",
                    url=url,
                    run_id=run_id,
                    attempt=attempt_number or None,
                )
            )
    return links


def _dedupe_links(*groups: list[MetricLink]) -> list[MetricLink]:
    seen: set[str] = set()
    links: list[MetricLink] = []
    for group in groups:
        for link in group:
            if link.url and link.url not in seen:
                seen.add(link.url)
                links.append(link)
    return links


def build_run_summary(
    run: JsonDict, result: JsonDict | None = None, *, fallback_index: int = 0
) -> RunSummary:
    """Build one public summary from persisted, potentially historical records."""
    raw_id = _text(run.get("run_id") or run.get("training_run_id"))
    modal_app_id = _text(run.get("modal_app_id"))
    created_at = _timestamp(run.get("created_at") or run.get("started_at"))
    training_run_id = (
        raw_id
        or f"unknown-run-{modal_app_id or 'no-app'}-{created_at}-{fallback_index}"
    )
    metadata = _mapping(run.get("metadata"))
    raw_config = run.get("config")
    config = _mapping(raw_config)
    config_summary = _config_summary(raw_config, training_run_id)
    result_summary = _train_result_summary(result) if result else None
    group_id = (
        _text(run.get("group_id"))
        or _text(metadata.get("group_id"))
        or _text(_mapping(metadata.get("group_tags")).get("group_id"))
    )
    started_at = _timestamp(run.get("started_at")) or created_at
    ended_at = _timestamp(run.get("ended_at")) or None
    status = _text(run.get("status")) or "running"
    completed_at = _timestamp(run.get("completed_at")) or (
        ended_at if status == "completed" else None
    )
    updated_at = (
        _timestamp(run.get("updated_at"))
        or completed_at
        or ended_at
        or started_at
        or created_at
    )
    duration = _optional_int(run.get("duration_seconds"))
    if duration is not None and duration < 0:
        duration = None
    if duration is None and started_at and ended_at:
        duration = max(0, ended_at - started_at)
    modal_app_url = _modal_app_url(modal_app_id)
    config_links = (
        config_summary.metric_links if isinstance(config_summary, ConfigSummary) else []
    )
    links = _dedupe_links(
        _metric_attempt_links(metadata),
        result_summary.metric_links if result_summary else [],
        config_links,
    )
    config_model = (
        config_summary.model_name if isinstance(config_summary, ConfigSummary) else ""
    )
    config_dataset = (
        config_summary.dataset_name if isinstance(config_summary, ConfigSummary) else ""
    )
    model = (result_summary.model_name if result_summary else "") or config_model
    framework = _text(run.get("framework")) or "(untagged)"
    framework_status = _text(run.get("framework_status"))
    framework_progress = _framework_progress(metadata)
    training_type: Literal["sft", "rl"] = (
        "sft"
        if _text(_mapping(config.get("recipe")).get("loss_type")) == "sft_loss"
        else "rl"
    )
    return RunSummary(
        training_run_id=training_run_id,
        run_id=training_run_id,
        status=status,
        display_status=_display_status(
            status,
            has_train_result=result_summary is not None,
        ),
        display_stage=_display_stage(
            framework_status, framework_progress, training_type
        ),
        framework=framework,
        framework_status=framework_status,
        framework_progress=framework_progress,
        latest_rollout=_latest_rollout(metadata),
        model=model,
        dataset=config_dataset,
        recipe=framework,
        training_type=training_type,
        group_id=group_id,
        group_tags=_group_tags(metadata, group_id),
        modal_app_id=modal_app_id,
        modal_app_url=modal_app_url,
        dataset_id=_text(run.get("dataset_id")),
        deployment_id=_text(run.get("deployment_id")),
        created_at=created_at,
        started_at=started_at,
        updated_at=updated_at,
        ended_at=ended_at,
        completed_at=completed_at,
        duration_seconds=duration,
        has_train_result=result_summary is not None,
        config_summary=config_summary,
        train_result=result_summary,
        metric_links=links,
        resume_state=_resume_state(metadata),
        config=config,
        metadata=metadata or None,
        error_message=_text(run.get("error_message")),
        step_times=cast(StepTimes, run.get("step_times"))
        if isinstance(run.get("step_times"), dict)
        else None,
        substep_times=cast(SubstepTimes, run.get("substep_times"))
        if isinstance(run.get("substep_times"), dict)
        else None,
    )


def build_run_summaries(
    runs: list[JsonDict], train_results: list[JsonDict] | None = None
) -> list[RunSummary]:
    """Join, normalize, de-duplicate, and sort persisted run-list records."""
    results_by_id: dict[str, JsonDict] = {}
    for result in train_results or []:
        if not isinstance(result, dict):
            continue
        result_id = _text(result.get("training_run_id"))
        if result_id:
            results_by_id[result_id] = result

    deduped: dict[str, RunSummary] = {}
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            continue
        run_id = _text(run.get("run_id") or run.get("training_run_id"))
        try:
            summary = build_run_summary(
                run, results_by_id.get(run_id), fallback_index=index
            )
        except ValidationError:
            continue
        existing = deduped.get(summary.training_run_id)
        if existing is None or (summary.created_at, summary.started_at) >= (
            existing.created_at,
            existing.started_at,
        ):
            deduped[summary.training_run_id] = summary
    return sorted(
        deduped.values(),
        key=lambda item: (item.created_at, item.training_run_id),
        reverse=True,
    )
