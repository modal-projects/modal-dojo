"""List-field metadata, facets, and filtering for training-run summaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from modal_training_gym.common.run_summary import RunSummary

UNTAGGED_RECIPE = "(untagged)"
NO_GROUP = "(no group)"
PENDING_STATUS = "pending"

FACET_NAMES = ("status", "recipe", "group", "training_type")


def run_list_field_metadata() -> dict[str, dict[str, object]]:
    """Return list metadata from selected ``RunSummary`` fields."""
    metadata: dict[str, dict[str, object]] = {}
    for field_name, field in RunSummary.model_fields.items():
        extra = (
            field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
        )
        if not extra.get("list"):
            continue
        metadata[field_name] = {
            "label": field.title or field_name.replace("_", " ").title(),
            "filterable": bool(extra.get("filterable")),
            "timestamp": bool(extra.get("timestamp")),
        }
    return metadata


def run_facet_values(summary: RunSummary) -> dict[str, str]:
    """Return the buckets the run list's filter chips group a run into."""
    return {
        "status": summary.display_status or PENDING_STATUS,
        "recipe": summary.recipe or summary.framework or UNTAGGED_RECIPE,
        "group": summary.group_id or NO_GROUP,
        "training_type": summary.training_type,
    }


def count_run_facets(summaries: Iterable[RunSummary]) -> dict[str, object]:
    """Count runs per facet bucket and per status, over every known run.

    The list endpoint serves one page at a time, so the chips and the totals on
    the page can no longer be counted from the rows the client happens to hold.
    """
    facets: dict[str, dict[str, int]] = {name: {} for name in FACET_NAMES}
    total = 0
    for summary in summaries:
        total += 1
        for name, value in run_facet_values(summary).items():
            facets[name][value] = facets[name].get(value, 0) + 1
    return {"total": total, **facets}


def _search_values(summary: RunSummary) -> Iterable[str]:
    yield summary.run_id
    yield summary.training_run_id
    yield summary.modal_app_id
    yield summary.model
    yield summary.dataset
    yield summary.recipe
    yield summary.framework
    yield summary.framework_status
    yield summary.deployment_id
    yield summary.group_id
    if summary.group_tags is not None:
        yield summary.group_tags.group_id
        yield from summary.group_tags.axes
        for tag in summary.group_tags.tags:
            yield tag.key
            yield tag.label
            yield str(tag.value)
        for key, value in summary.group_tags.overrides.items():
            yield key
            yield str(value)
    if summary.train_result is not None:
        yield summary.train_result.training_run_id
        yield summary.train_result.checkpoint_dir
        yield summary.train_result.model_name
        yield summary.train_result.model_path


def _matches_query(summary: RunSummary, query: str) -> bool:
    if not query:
        return True
    return any(query in value.casefold() for value in _search_values(summary) if value)


def filter_run_summaries(
    summaries: Iterable[RunSummary],
    *,
    filters: Mapping[str, str] | None = None,
    facets: Mapping[str, set[str]] | None = None,
    query: str = "",
    since: int | None = None,
    limit: int | None = None,
    offset: int = 0,
    sort_by: Literal["updated", "created"] = "updated",
) -> list[RunSummary]:
    """Filter summaries by their configured list fields and update recency.

    ``facets`` keeps only runs whose bucket is in the given set (see
    ``run_facet_values``); a facet without an entry is unfiltered. ``query`` is
    matched case-insensitively against the run's identifiers, model, dataset and
    group tags, and ``offset``/``limit`` page the sorted result.
    """
    metadata = run_list_field_metadata()
    active_filters = {
        name: value.strip().casefold()
        for name, value in (filters or {}).items()
        if metadata.get(name, {}).get("filterable") and value.strip()
    }
    active_facets = {
        name: values for name, values in (facets or {}).items() if name in FACET_NAMES
    }
    normalized_query = query.strip().casefold()

    selected: list[RunSummary] = []
    for summary in summaries:
        if since is not None and max(summary.created_at, summary.updated_at) < since:
            continue
        if any(
            str(getattr(summary, name)).casefold() != expected
            for name, expected in active_filters.items()
        ):
            continue
        if active_facets:
            values = run_facet_values(summary)
            if any(
                values[name] not in expected for name, expected in active_facets.items()
            ):
                continue
        if not _matches_query(summary, normalized_query):
            continue
        selected.append(summary)

    if sort_by == "created":
        selected.sort(
            key=lambda summary: (summary.created_at, summary.run_id), reverse=True
        )
    else:
        selected.sort(
            key=lambda summary: (summary.updated_at, summary.run_id), reverse=True
        )
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected
