from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from modal_dojo.common.models.base import ParsedResponse


class TraceSpan(BaseModel):
    """One span or instant event in a sample's execution trace.

    A duration span has both ``start`` and ``end`` (seconds, rebased so the
    sample's first span starts at 0); an instant event has ``end is None``.
    ``attributes`` carries timings/counts only — never response or tool
    payloads, which already live on the Sample — so traces stay small (see the
    recorder's normalizer). ``parent`` is the enclosing span's name, if any.
    """

    name: str = ""
    start: float = 0.0
    end: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    parent: str | None = None


class Sample(BaseModel):
    """A prompt, response, parsed structure, score, and metadata from one model call.

    Attributes:
        score: Reward or eval score for this call.
        prompt: Text sent to the model for this call.
        response: Raw model output.
        parsed_response: Structured parse of ``response``, if available.
        metadata: Custom metadata attached to this call by a reward or rollout function.
        trace: Execution spans for a sampled subset of rollouts when recording
            is on. ``None`` otherwise.
    """

    score: float = 0.0
    prompt: str = ""
    response: str = ""
    parsed_response: ParsedResponse | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace: list[TraceSpan] | None = None
