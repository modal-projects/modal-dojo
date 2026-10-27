"""Fail closed when a throughput sample doesn't actually fill the window."""

from types import SimpleNamespace

import pytest

from modal_dojo.frameworks.miles.benchmarks.beam import validate_sample


def sample(prompt=16, response=8, status="TRUNCATED", logprobs=8):
    return SimpleNamespace(
        tokens=[0] * (prompt + response),
        response_length=response,
        rollout_log_probs=[0.0] * logprobs,
        status=status,
        index=3,
    )


def test_complete_forced_sample_is_measured():
    assert validate_sample(sample(), 16, 8)["total_tokens"] == 24


@pytest.mark.parametrize(
    "kwargs",
    [
        {"response": 7},
        {"prompt": 15},
        {"status": "ABORTED"},
        {"logprobs": 7},
    ],
)
def test_incomplete_samples_cannot_be_benchmarked(kwargs):
    with pytest.raises(ValueError):
        validate_sample(sample(**kwargs), 16, 8)


def test_batch_audit_rejects_missing_or_repeated_samples():
    from modal_dojo.frameworks.miles.benchmarks.beam import audit_full_batch

    args = SimpleNamespace(rollout_batch_size=2, n_samples_per_prompt=1)
    with pytest.raises(ValueError, match="Incomplete"):
        audit_full_batch(0, args, [sample()], {}, 1.0)
    with pytest.raises(ValueError, match="distinct"):
        audit_full_batch(0, args, [sample(), sample()], {}, 1.0)
