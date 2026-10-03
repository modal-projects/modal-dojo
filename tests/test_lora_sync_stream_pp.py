"""Pinned-source and allocation-plan regressions for pipeline adapter sync."""

import math
from enum import Enum
from pathlib import Path

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_lora_sync_stream_pp as patch,
)

SOURCE = Path(__file__).parent / "testdata/miles/k3_hf_weight_iterator.py.input"


class Dtype(Enum):
    BF16 = 2
    FP32 = 4

    @property
    def itemsize(self):
        return self.value


def chunks(meta, limit):
    namespace = {"math": math, "MegatronHfWeightIteratorBase": type("Base", (), {})}
    exec(patch.OVERRIDE, namespace)
    return list(namespace["_lora_pp_chunks"](meta, limit))


def test_exact_pinned_source_and_idempotence(tmp_path):
    target = tmp_path / "iterator.py"
    target.write_text(SOURCE.read_text())
    patch.apply(target)
    expected = SOURCE.with_suffix(".output").read_text()
    assert target.read_text() == expected
    compile(expected, str(target), "exec")
    patch.apply(target)
    assert target.read_text() == expected


@pytest.mark.parametrize("damage", ["missing", "duplicate", "partial", "old"])
def test_source_drift_rejected(tmp_path, damage):
    source = SOURCE.read_text()
    if damage == "missing":
        source = source.replace(patch.ANCHOR, "def renamed(")
    elif damage == "duplicate":
        source += source
    elif damage == "partial":
        source += "\n# " + patch.MARKER
    else:
        source += "\n# PATCHED_LORA_SYNC_STREAM_PP\n"
    target = tmp_path / "iterator.py"
    target.write_text(source)
    with pytest.raises(RuntimeError):
        patch.apply(target)
    assert target.read_text() == source


def test_six_gib_stage_is_split_before_allocation():
    mib = 2**20
    meta = [(f"expert{i}", (896, 3072, 32), Dtype.BF16) for i in range(36)]
    meta.append(("other", (28 * mib,), Dtype.BF16))
    plan = chunks(meta, 256 * mib)
    assert sum(size * dtype.itemsize for dtype, _, size in plan) == 6104 * mib
    assert max(size * dtype.itemsize for dtype, _, size in plan) <= 256 * mib
    assert [name for _, entries, _ in plan for name, _ in entries] == [
        name for name, _, _ in meta
    ]


@pytest.mark.parametrize("limit", [8, 16, 32, 64])
def test_mixed_dtypes_and_oversized_atomic_tensors(limit):
    meta = [
        ("a", (3, 2), Dtype.BF16),
        ("b", (2, 2), Dtype.FP32),
        ("c", (100,), Dtype.BF16),
        ("empty", (0,), Dtype.FP32),
        ("scalar", (), Dtype.FP32),
        ("d", (2,), Dtype.BF16),
    ]
    plan = chunks(meta, limit)
    seen = {}
    for dtype, entries, size in plan:
        assert size == sum(math.prod(shape) for _, shape in entries)
        if size * dtype.itemsize > limit:
            assert len(entries) == 1  # Never split or combine oversized tensors.
        for name, shape in entries:
            assert name not in seen
            seen[name] = (shape, dtype)
    assert seen == {name: (shape, dtype) for name, shape, dtype in meta}
    assert chunks(meta, limit) == plan  # Same collective order on every PP rank.


def test_empty_stage_and_invalid_budget():
    assert chunks([], 256) == []
    for limit in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            chunks([], limit)
