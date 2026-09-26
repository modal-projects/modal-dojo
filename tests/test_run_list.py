from __future__ import annotations

import pytest

from modal_training_gym.common.run_list import (
    count_run_facets,
    filter_run_summaries,
    run_list_field_metadata,
)
from modal_training_gym.common.run_summary import RunSummary


def _summary(**overrides) -> RunSummary:
    values = {
        "training_run_id": "run-1",
        "run_id": "run-1",
        "status": "running",
        "display_status": "pending",
        "display_stage": "Training",
        "framework_status": "training",
        "model": "org/model",
        "dataset": "org/data",
        "recipe": "slime",
        "group_id": "nightly",
        "created_at": 100,
        "updated_at": 200,
    }
    values.update(overrides)
    return RunSummary(**values)


def test_schema_metadata_drives_columns_and_filters():
    fields = run_list_field_metadata()

    assert list(fields) == [
        "run_id",
        "display_status",
        "display_stage",
        "model",
        "dataset",
        "recipe",
        "training_type",
        "group_id",
        "created_at",
        "updated_at",
    ]
    assert {name for name, metadata in fields.items() if metadata["filterable"]} == {
        "display_status",
        "model",
        "dataset",
        "recipe",
        "training_type",
        "group_id",
    }


def test_filtering_uses_projection_values_and_update_recency():
    older = _summary(run_id="older", training_run_id="older")
    newer = _summary(
        run_id="newer",
        training_run_id="newer",
        status="failed",
        display_status="failed",
        created_at=250,
        updated_at=300,
    )

    assert filter_run_summaries(
        [older, newer],
        filters={"display_status": "FAILED"},
    ) == [newer]
    assert filter_run_summaries([older, newer], since=225, limit=1) == [newer]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("display_status", "failed"),
        ("model", "org/model"),
        ("dataset", "org/data"),
        ("recipe", "slime"),
        ("group_id", "nightly"),
    ],
)
def test_each_filterable_field_matches_case_insensitively(field, value):
    matching = _summary(
        run_id="matching",
        training_run_id="matching",
        **{field: value},
    )
    nonmatching = _summary(
        run_id="nonmatching",
        training_run_id="nonmatching",
        **{field: "different"},
    )

    assert filter_run_summaries(
        [nonmatching, matching],
        filters={field: f" {value.upper()} "},
    ) == [matching]


def test_multiple_filters_use_intersection_semantics():
    matching = _summary(
        run_id="matching",
        training_run_id="matching",
        display_status="failed",
        model="org/model",
        dataset="org/data",
        recipe="slime",
        group_id="nightly",
    )
    wrong_model = _summary(
        run_id="wrong-model",
        training_run_id="wrong-model",
        display_status="failed",
        model="other/model",
    )
    wrong_group = _summary(
        run_id="wrong-group",
        training_run_id="wrong-group",
        display_status="failed",
        group_id="daytime",
    )

    assert filter_run_summaries(
        [wrong_model, matching, wrong_group],
        filters={
            "display_status": "failed",
            "model": "org/model",
            "dataset": "org/data",
            "recipe": "slime",
            "group_id": "nightly",
        },
    ) == [matching]


def test_empty_and_nonfilterable_filters_are_ignored():
    first = _summary(run_id="a", training_run_id="a", updated_at=100)
    second = _summary(run_id="b", training_run_id="b", updated_at=200)

    assert filter_run_summaries(
        [first, second],
        filters={"model": " ", "run_id": "not-a-filter"},
    ) == [second, first]


def test_since_uses_created_or_updated_time_inclusively_before_limiting():
    recently_created = _summary(
        run_id="recently-created",
        training_run_id="recently-created",
        created_at=300,
        updated_at=100,
    )
    recently_updated = _summary(
        run_id="recently-updated",
        training_run_id="recently-updated",
        created_at=100,
        updated_at=300,
    )
    old = _summary(
        run_id="old",
        training_run_id="old",
        created_at=299,
        updated_at=299,
    )

    assert filter_run_summaries(
        [old, recently_created, recently_updated],
        since=300,
    ) == [recently_updated, recently_created]
    assert filter_run_summaries(
        [old, recently_created, recently_updated],
        since=300,
        limit=1,
    ) == [recently_updated]


def test_facets_select_buckets_and_default_to_every_bucket():
    slime = _summary(run_id="slime", training_run_id="slime", recipe="slime")
    miles = _summary(
        run_id="miles",
        training_run_id="miles",
        recipe="miles",
        display_status="failed",
        group_id="",
    )

    assert filter_run_summaries([slime, miles], facets={"recipe": {"miles"}}) == [miles]
    assert filter_run_summaries(
        [slime, miles],
        facets={"status": {"pending", "failed"}, "group": {"nightly", "(no group)"}},
    ) == [slime, miles]
    assert filter_run_summaries([slime, miles], facets={"group": {"(no group)"}}) == [
        miles
    ]


def test_query_matches_the_group_metadata_the_tags_column_renders():
    tagged = _summary(
        run_id="tagged",
        training_run_id="tagged",
        group_tags={
            "group_id": "sweep-1",
            "axes": ["learning_rate"],
            "tags": [{"key": "lr", "label": "Learning rate", "value": 3e-4}],
        },
    )
    plain = _summary(run_id="plain", training_run_id="plain")

    for query in ("learning_rate", "Learning rate", "0.0003", "sweep-1"):
        assert filter_run_summaries([tagged, plain], query=query) == [tagged]


def test_query_matches_identifiers_and_paging_slices_the_sorted_result():
    first = _summary(run_id="first", training_run_id="first", created_at=300)
    second = _summary(
        run_id="second",
        training_run_id="second",
        model="org/other",
        created_at=200,
    )
    third = _summary(run_id="third", training_run_id="third", created_at=100)

    assert filter_run_summaries([first, second, third], query=" ORG/OTHER ") == [second]
    assert filter_run_summaries(
        [first, second, third],
        sort_by="created",
        offset=1,
        limit=1,
    ) == [second]


def test_facet_counts_cover_every_run():
    runs = [
        _summary(run_id="a", training_run_id="a"),
        _summary(run_id="b", training_run_id="b", display_status="failed"),
        _summary(
            run_id="c",
            training_run_id="c",
            recipe="",
            framework="miles",
            group_id="",
        ),
    ]

    assert count_run_facets(runs) == {
        "total": 3,
        "status": {"pending": 2, "failed": 1},
        "recipe": {"slime": 2, "miles": 1},
        "group": {"nightly": 2, "(no group)": 1},
        "training_type": {"rl": 3},
    }


def test_training_type_facet_filters_and_counts():
    rl = _summary(run_id="rl", training_run_id="rl")
    sft = _summary(run_id="sft", training_run_id="sft", training_type="sft")

    assert filter_run_summaries(
        [rl, sft],
        facets={"training_type": {"sft"}},
    ) == [sft]
    assert count_run_facets([rl, sft])["training_type"] == {"rl": 1, "sft": 1}
