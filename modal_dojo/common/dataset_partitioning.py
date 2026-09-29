from collections import Counter
from collections.abc import Callable
import random
from typing import Any

RowKey = str | Callable[[dict[str, Any]], str]


def _value(row: dict[str, Any], key: RowKey | None) -> str:
    if key is None:
        return ""
    if callable(key):
        return str(key(row))
    value: Any = row
    for part in key.split("."):
        value = value[part]
    return str(value)


def sample_rows(
    rows: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    stratify_key: RowKey | None = None,
) -> list[dict[str, Any]]:
    if count < 0 or count > len(rows):
        raise ValueError(f"sample count {count} is outside [0, {len(rows)}]")

    rows_by_category: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        rows_by_category.setdefault(_value(row, stratify_key), []).append(index)

    rng = random.Random(seed)
    for indices in rows_by_category.values():
        rng.shuffle(indices)
    tie_break = {name: rng.random() for name in sorted(rows_by_category)}
    category_share = {
        name: len(indices) / len(rows) for name, indices in rows_by_category.items()
    }

    selected_counts: Counter[str] = Counter()

    def category_deficit(name: str, position: int) -> tuple[float, float, str]:
        expected = position * category_share[name]
        return expected - selected_counts[name], tie_break[name], name

    selected: set[int] = set()
    for position in range(1, count + 1):
        categories_with_rows_left = [
            name
            for name, indices in rows_by_category.items()
            if selected_counts[name] < len(indices)
        ]
        name = max(
            categories_with_rows_left,
            key=lambda value: category_deficit(value, position),
        )
        selected.add(rows_by_category[name][selected_counts[name]])
        selected_counts[name] += 1
    return [row for index, row in enumerate(rows) if index in selected]


def split_rows(
    rows: list[dict[str, Any]],
    *,
    eval_fraction: float,
    seed: int,
    group_key: RowKey | None = None,
    stratify_key: RowKey | None = None,
    min_train_groups: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1")
    if min_train_groups < 1:
        raise ValueError("min_train_groups must be positive")
    if not rows:
        raise ValueError("cannot split an empty dataset")

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(
            _value(row, group_key) if group_key is not None else str(index), []
        ).append(index)
    group_category_counts = {
        group: Counter(_value(rows[index], stratify_key) for index in indices)
        for group, indices in groups.items()
    }
    category_counts = Counter(_value(row, stratify_key) for row in rows)
    groups_by_category = {
        name: {group for group, counts in group_category_counts.items() if counts[name]}
        for name in category_counts
    }
    eval_target_num_rows = round(len(rows) * eval_fraction)
    eval_target_category_counts = {
        name: count * eval_fraction for name, count in category_counts.items()
    }

    rng = random.Random(seed)
    tie_break = {group: rng.random() for group in sorted(groups)}
    eval_selected: set[str] = set()
    eval_num_rows = 0
    eval_category_counts: Counter[str] = Counter()

    def can_move_to_eval(group: str) -> bool:
        if group in eval_selected:
            return False
        return all(
            len(groups_by_category[name] - eval_selected - {group}) >= min_train_groups
            for name in group_category_counts[group]
        )

    def move_to_eval(group: str) -> None:
        nonlocal eval_num_rows
        eval_selected.add(group)
        eval_num_rows += len(groups[group])
        eval_category_counts.update(group_category_counts[group])

    for name in sorted(
        category_counts, key=lambda value: (category_counts[value], value)
    ):
        if (
            len(groups_by_category[name]) < min_train_groups
            or eval_category_counts[name]
        ):
            continue
        eval_candidates = [
            group for group in groups_by_category[name] if can_move_to_eval(group)
        ]
        if eval_candidates:
            move_to_eval(
                min(
                    eval_candidates,
                    key=lambda group: (
                        len(groups[group]),
                        tie_break[group],
                        group,
                    ),
                )
            )

    def eval_selection_error(group: str) -> tuple[float, float, str]:
        size_error = abs(
            eval_num_rows + len(groups[group]) - eval_target_num_rows
        ) / max(eval_target_num_rows, 1)
        category_error = sum(
            abs(
                eval_category_counts[name] + group_category_counts[group][name] - target
            )
            / max(target, 1.0)
            for name, target in eval_target_category_counts.items()
        ) / max(len(eval_target_category_counts), 1)
        return size_error + category_error, tie_break[group], group

    while eval_num_rows < eval_target_num_rows:
        remaining = eval_target_num_rows - eval_num_rows
        eval_candidates = [
            group
            for group in groups
            if can_move_to_eval(group) and len(groups[group]) <= remaining
        ]
        if not eval_candidates:
            eval_candidates = [group for group in groups if can_move_to_eval(group)]
        if not eval_candidates:
            break
        move_to_eval(min(eval_candidates, key=eval_selection_error))

    eval_indices = sorted(index for group in eval_selected for index in groups[group])
    eval_index_set = set(eval_indices)
    train_rows = [row for index, row in enumerate(rows) if index not in eval_index_set]
    eval_rows = [rows[index] for index in eval_indices]
    return train_rows, eval_rows
