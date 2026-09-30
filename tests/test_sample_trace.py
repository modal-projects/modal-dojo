"""Per-sample execution-trace capture for slime rollouts.

Covers the pure, slime-free seams of the dashboard trajectory-timeline feature:

  * :func:`_normalize_trace` — coercing slime's (version-varying) trace shapes
    into ``{name, start, end, attributes, parent}`` spans: rebasing times to 0,
    dropping payload-bearing attributes, and handling instant events; and
  * the capture gate in :func:`report_rollout_samples` — off by default, and
    when on, tracing only the first ``trace_sample_limit`` samples per rollout.
"""

from types import SimpleNamespace

from modal_dojo.common.sample import Sample, TraceSpan
from modal_dojo.frameworks.slime import phase_reporting as pr


# ── _normalize_trace ─────────────────────────────────────────────────────────


def test_normalize_rebases_times_to_zero():
    """Absolute (epoch-ish) span times are rebased so the first span starts at 0."""
    trace = pr._normalize_trace(
        [
            {"name": "generate", "start": 1000.0, "end": 1002.5},
            {"name": "reward", "start": 1002.5, "end": 1003.0},
        ]
    )
    assert trace == [
        {
            "name": "generate",
            "start": 0.0,
            "end": 2.5,
            "attributes": {},
            "parent": None,
        },
        {"name": "reward", "start": 2.5, "end": 3.0, "attributes": {}, "parent": None},
    ]


def test_normalize_drops_payload_attributes_keeps_scalars():
    """Only small scalars survive — long strings / lists / dicts are stripped so
    a trace can't smuggle the response or tool payloads that already live on the
    Sample."""
    trace = pr._normalize_trace(
        [
            {
                "name": "generate",
                "start": 0.0,
                "end": 1.0,
                "attributes": {
                    "tokens": 128,
                    "cached": True,
                    "temp": 0.7,
                    "text": "x" * (pr._TRACE_ATTR_STR_MAX + 1),
                    "logits": [1, 2, 3],
                    "nested": {"a": 1},
                    "short": "ok",
                },
            }
        ]
    )
    assert trace[0]["attributes"] == {
        "tokens": 128,
        "cached": True,
        "temp": 0.7,
        "short": "ok",
    }


def test_normalize_instant_event_has_no_end():
    """A span with no end (only a start/ts) is an instant event: end -> None."""
    trace = pr._normalize_trace(
        [
            {"name": "anchor", "start": 5.0, "end": 5.0},  # zero-width -> instant
            {"name": "mark", "ts": 6.0, "parent": "anchor"},  # only a timestamp
        ]
    )
    assert trace[0]["end"] is None  # end == start collapses to an instant
    assert trace[1] == {
        "name": "mark",
        "start": 1.0,
        "end": None,
        "attributes": {},
        "parent": "anchor",
    }


def test_normalize_accepts_dict_wrapped_and_name_keyed_shapes():
    """slime may hand us a list, a {'spans': [...]} dict, or a name-keyed dict."""
    spans = [{"name": "g", "start": 0.0, "end": 1.0}]
    assert pr._normalize_trace({"spans": spans}) == pr._normalize_trace(spans)
    keyed = pr._normalize_trace({"g": {"start": 2.0, "end": 3.0}})
    assert keyed == [
        {"name": "g", "start": 0.0, "end": 1.0, "attributes": {}, "parent": None}
    ]


def test_normalize_caps_span_count():
    raw = [
        {"name": f"s{i}", "start": float(i), "end": float(i) + 0.1} for i in range(500)
    ]
    trace = pr._normalize_trace(raw)
    assert len(trace) == pr._TRACE_MAX_SPANS


def test_normalize_empty_inputs_return_none():
    assert pr._normalize_trace(None) is None
    assert pr._normalize_trace([]) is None
    assert pr._normalize_trace({}) is None
    assert pr._normalize_trace([{"no_timing": 1}]) is None


# ── _extract_trace ───────────────────────────────────────────────────────────


def test_extract_trace_from_attr_metadata_and_dict():
    span = [{"name": "g", "start": 0.0}]
    assert pr._extract_trace(SimpleNamespace(trace=span)) is span
    assert pr._extract_trace(SimpleNamespace(metadata={"trace": span})) is span
    assert pr._extract_trace({"trace": span}) is span
    assert pr._extract_trace({"metadata": {"trace": span}}) is span
    assert pr._extract_trace(SimpleNamespace(prompt="x")) is None


# ── inference metadata ───────────────────────────────────────────────────────


def test_sample_to_dict_extracts_prefix_cache_info():
    sample = SimpleNamespace(
        prompt="prompt",
        response="response",
        reward=1.0,
        response_length=80,
        prefix_cache_info=SimpleNamespace(
            total_prompt_tokens=1_000,
            cached_tokens=250,
        ),
    )

    recorded = pr._sample_to_dict(sample)

    assert recorded["metadata"]["inference"] == {
        "tokens_in": 1_000,
        "tokens_out": 80,
        "cached_tokens": 250,
        "new_tokens": 750,
        "cache_hit_rate": 0.25,
    }


def test_sample_to_dict_100_percent_cache_hit():
    """All prompt tokens cached → cache_hit_rate is 1.0, new_tokens is 0."""
    sample = SimpleNamespace(
        prompt="prompt",
        response="response",
        reward=1.0,
        response_length=120,
        prefix_cache_info=SimpleNamespace(
            total_prompt_tokens=500,
            cached_tokens=500,
        ),
    )

    recorded = pr._sample_to_dict(sample)

    assert recorded["metadata"]["inference"] == {
        "tokens_in": 500,
        "tokens_out": 120,
        "cached_tokens": 500,
        "new_tokens": 0,
        "cache_hit_rate": 1.0,
    }


def test_sample_to_dict_no_inference_without_prefix_cache_info():
    """Samples without prefix_cache_info (e.g. custom rollouts) get no inference key."""
    sample = SimpleNamespace(
        prompt="prompt",
        response="response",
        reward=0.0,
        response_length=4,
    )

    recorded = pr._sample_to_dict(sample)

    assert "inference" not in recorded["metadata"]


# ── capture gate + sampling ──────────────────────────────────────────────────


def _fake_sample(i: int):
    return SimpleNamespace(
        prompt=f"p{i}",
        response=f"r{i}",
        reward=float(i),
        trace=[{"name": "generate", "start": 0.0, "end": 1.0}],
    )


def test_report_rollout_disabled_by_default(monkeypatch):
    monkeypatch.delenv(pr.CAPTURE_TRACE_ENV, raising=False)
    captured = {}
    monkeypatch.setattr(pr, "_enqueue_rollout", lambda p: captured.update(p))
    pr.report_rollout_samples(0, SimpleNamespace(), [_fake_sample(0)], {}, None)
    assert all("trace" not in s for s in captured["samples"])


def test_report_rollout_traces_only_first_n(monkeypatch):
    monkeypatch.setenv(pr.CAPTURE_TRACE_ENV, "1")
    monkeypatch.setenv(pr.TRACE_SAMPLE_LIMIT_ENV, "2")
    captured = {}
    monkeypatch.setattr(pr, "_enqueue_rollout", lambda p: captured.update(p))
    samples = [_fake_sample(i) for i in range(5)]
    pr.report_rollout_samples(3, SimpleNamespace(), samples, {}, None)

    have_trace = ["trace" in s for s in captured["samples"]]
    assert have_trace == [True, True, False, False, False]
    assert captured["samples"][0]["trace"][0]["name"] == "generate"


def test_trace_sample_limit_default_on_bad_env(monkeypatch):
    monkeypatch.setenv(pr.TRACE_SAMPLE_LIMIT_ENV, "not-an-int")
    assert pr._trace_sample_limit() == pr._TRACE_SAMPLE_LIMIT_DEFAULT


# ── Sample model round-trip ──────────────────────────────────────────────────


def test_sample_trace_round_trips_through_model():
    """The trace must survive Sample validation + JSON dump (how it persists to
    the volume); without the model field pydantic would silently drop it."""
    raw = {
        "score": 1.0,
        "prompt": "p",
        "response": "r",
        "trace": [
            {"name": "generate", "start": 0.0, "end": 1.0, "attributes": {"tokens": 4}}
        ],
    }
    sample = Sample.model_validate(raw)
    assert isinstance(sample.trace[0], TraceSpan)
    dumped = sample.model_dump(mode="json")
    assert dumped["trace"][0]["name"] == "generate"
    assert dumped["trace"][0]["attributes"] == {"tokens": 4}


def test_sample_trace_defaults_none():
    assert Sample().trace is None
