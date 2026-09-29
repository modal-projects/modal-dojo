"""Per-rollout training data — prompts, responses, rewards.

Mirrors the EvalResult shape but lives under its own MetadataStore so the
dashboard can list rollouts per training run without scanning the eval store.
Records are written by slime's `log_rollout_data` hook through the async
phase-reporter; reads happen via the dashboard's
``/api/runs/{id}/rollouts`` endpoint.
"""

from __future__ import annotations

import ast
import copy
import json
import re
import time
from collections.abc import Awaitable
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from modal_dojo.common.advantage_distribution import quantile
from modal_dojo.common.coerce import optional_int, safe_int
from modal_dojo.common.sample import Sample
from modal_dojo.utils.metadata import (
    MetadataStore,
    vol_get_summary_items,
    vol_put_with_summary,
)


class TrainingRolloutSample(Sample):
    """``rollout_index`` is slime's per-episode ``Sample.rollout_id``, not the
    step id on ``TrainingRolloutResult``."""

    rollout_index: int | None = None
    sample_index: int | None = None
    group_index: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _promote_sample(cls, value: Any) -> Any:
        if isinstance(value, Sample) and not isinstance(value, cls):
            return value.model_dump()
        return value


# Metadata keys that are internal bookkeeping rather than reward-function
# tags — numeric, but not something a user wants charted as a custom metric.
_NON_TAG_METADATA_KEYS = frozenset(
    {
        "inference",
        "_metadata_type",
        "audio",
        "image",
        "image_ref",
        "response_length",
        "prompt_length",
        "rollout_id",
        "rollout_idx",
    }
)


_GEMMA_MARKER_RE = re.compile(r"<\|(?:turn|channel)>|<(?:turn|channel)\|>")


def _clean_prompt(text: str) -> str:
    """Make a chat-templated prompt readable for display.

    Dataset prompts often arrive as a chat template wrapping a Python repr
    of the messages list (e.g. ``<|im_start|>user\\n[{'content': '...',
    'role': 'user'}]<|im_end|>...``) plus a leaked reference/assistant turn.
    Pull the message content out and drop the template scaffolding; fall
    back to stripping special tokens when there's no messages repr.
    """
    start, end = text.find("[{"), text.rfind("}]")
    if start != -1 and end > start:
        try:
            data = ast.literal_eval(text[start : end + 2])
            if isinstance(data, list):
                parts = [
                    str(m["content"])
                    for m in data
                    if isinstance(m, dict) and m.get("content")
                ]
                if parts:
                    return "\n\n".join(parts).strip()
        except (ValueError, SyntaxError):
            pass
    # Gemma-style templates use bare <bos>/<eos> and half-delimited turn markers
    # (<|turn>, <turn|>) alongside the usual <|...|> tokens, so strip all three.
    is_gemma = _GEMMA_MARKER_RE.search(text) is not None
    cleaned = _GEMMA_MARKER_RE.sub("", text)
    cleaned = re.sub(r"<(?:bos|eos)>", "", cleaned)
    cleaned = re.sub(r"<\|[^|<>]*\|>", "", cleaned)
    cleaned = re.sub(r"</?think>", "", cleaned)
    # Drop standalone role-header lines left behind by the template.
    roles = (
        "system|user|assistant|model|thought" if is_gemma else "system|user|assistant"
    )
    cleaned = re.sub(rf"(?m)^({roles})\s*$\n?", "", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


_CHAT_TURN_RE = re.compile(r"<\|im_start\|>(\w+)\n")
_TOOL_RESPONSE_RE = re.compile(
    r"^\s*<tool_response>\s*(.*?)\s*</tool_response>\s*$", re.DOTALL
)


def _transcript_messages(text: str) -> list[dict[str, str]] | None:
    if "<|im_start|>" not in text:
        return None
    pieces = _CHAT_TURN_RE.split(text)
    turns = [("assistant", pieces[0]), *zip(pieces[1::2], pieces[2::2])]
    messages: list[dict[str, str]] = []
    for role, body in turns:
        body = body.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        if not body:
            continue
        if role == "user" and (match := _TOOL_RESPONSE_RE.match(body)):
            role, body = "tool", match.group(1)
        messages.append({"role": role, "content": body})
    return messages if len(messages) > 1 else None


def _apply_parsed(rows: object) -> None:
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = row.get("response")
        transcript = _transcript_messages(raw) if isinstance(raw, str) else None
        if transcript:
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                row["metadata"] = metadata
            metadata.setdefault("trajectory_messages", transcript)
        parsed = row.get("parsed_response")
        if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
            if isinstance(raw, str):
                row["raw_response"] = raw
            row["response"] = (
                raw
                if transcript and isinstance(raw, str)
                else parsed.get("content") or ""
            )
            if parsed.get("thinking") and not transcript:
                row["thinking"] = parsed["thinking"]
            if parsed.get("tool_calls"):
                row["tool_calls"] = parsed["tool_calls"]
        # Clean the (chat-templated) prompt for display, keeping the raw.
        raw_prompt = row.get("prompt")
        if isinstance(raw_prompt, str) and raw_prompt:
            cleaned_prompt = _clean_prompt(raw_prompt)
            if cleaned_prompt and cleaned_prompt != raw_prompt:
                row["raw_prompt"] = raw_prompt
                row["prompt"] = cleaned_prompt


def _numeric_tags(samples: list[TrainingRolloutSample]) -> dict[str, list[float]]:
    values_by_tag: dict[str, list[float]] = {}
    for sample in samples:
        for key, value in sample.metadata.items():
            if key in _NON_TAG_METADATA_KEYS:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values_by_tag.setdefault(key, []).append(float(value))
    return values_by_tag


_SUMMARY_PERCENTILES: dict[str, float] = {"p50": 0.5, "p90": 0.9, "p99": 0.99}


def _value_stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        **{name: quantile(ordered, q) for name, q in _SUMMARY_PERCENTILES.items()},
    }


class TrainingRolloutSummary(BaseModel):
    """Lightweight rollout data returned by the run-rollouts list endpoint."""

    training_run_id: str
    rollout_id: int
    created_at: int
    total: int
    episode_count: int | None = None
    mean: float
    reward_stats: dict[str, Any] | None = None
    export_size_bytes: int | None = None
    rollout_time: float | None = None
    error_summary: dict[str, Any] | None = None
    tag_stats: dict[str, dict[str, Any]] | None = None


class TrainingRolloutResult(BaseModel):
    """All samples from one rollout of one training run."""

    training_run_id: str
    rollout_id: int
    created_at: int = 0
    n_samples_per_prompt: int | None = None
    samples: list[TrainingRolloutSample] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    rollout_time: float | None = None

    def _rollout_groups(self) -> list[list[TrainingRolloutSample]]:
        groups: dict[tuple[str, int], list[TrainingRolloutSample]] = {}
        for position, sample in enumerate(self.samples):
            index = sample.rollout_index
            if index is None:
                index = optional_int(sample.metadata.get("rollout_id"))
            key = ("rollout", index) if index is not None else ("sample", position)
            groups.setdefault(key, []).append(sample)
        return list(groups.values())

    def _episode_rewards(self) -> list[float]:
        return [
            sum(s.score for s in group) / len(group) for group in self._rollout_groups()
        ]

    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def episode_count(self) -> int:
        return len(self._rollout_groups())

    @property
    def mean(self) -> float:
        rewards = self._episode_rewards()
        if not rewards:
            return 0.0
        return sum(rewards) / len(rewards)

    @property
    def reward_stats(self) -> dict[str, Any] | None:
        rewards = self._episode_rewards()
        return _value_stats(rewards) if rewards else None

    @property
    def storage_key(self) -> str:
        # One canonical file per (run, rollout). Zero-padded rollout id so
        # listings return them in step order without needing a sort.
        return f"{self.training_run_id}__{self.rollout_id:08d}"

    @property
    def error_summary(self) -> dict[str, Any] | None:
        """Extract error diagnostics from Harbor/agent rollout metrics.

        Returns None when there's nothing notable; otherwise a compact dict
        with the counts a human needs to diagnose an all-zero-reward rollout
        without digging into raw logs.
        """
        m = self.metrics
        if not m:
            return None

        total_samples = (
            safe_int(m.get("agent/valid_sample_count"))
            or safe_int(m.get("agent/raw_zero_reward_sample_count"))
            or self.total
            or 0
        )
        if not total_samples:
            return None

        out: dict[str, Any] = {}

        # Count samples that errored due to infra (sandbox creation, image build, etc.)
        for key, label in (
            ("agent/exit_status/remoteerror_sample_count", "remote_error"),
            ("agent/response_missing_sample_count", "response_missing"),
            ("agent/invalid_infra_sample_count", "infra_invalid"),
            ("agent/limits_exceeded_sample_count", "limits_exceeded"),
        ):
            v = safe_int(m.get(key))
            if v:
                out[label] = v

        if not out:
            return None

        out["total_samples"] = total_samples
        infra_errors = out.get("remote_error", 0) + out.get("infra_invalid", 0)
        if infra_errors >= total_samples:
            out["verdict"] = "all_infra_failure"
        elif infra_errors > 0:
            out["verdict"] = "partial_infra_failure"
        return out

    @property
    def tag_stats(self) -> dict[str, dict[str, Any]]:
        """Per-tag count/mean/min/max, weighted per rollout like ``mean``."""
        values_by_tag: dict[str, list[float]] = {}
        for group in self._rollout_groups():
            for tag, values in _numeric_tags(group).items():
                values_by_tag.setdefault(tag, []).append(sum(values) / len(values))

        return {tag: _value_stats(values) for tag, values in values_by_tag.items()}

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "training_run_id": self.training_run_id,
            "rollout_id": self.rollout_id,
            "created_at": self.created_at,
            "total": self.total,
            "episode_count": self.episode_count,
            "mean": self.mean,
        }
        reward_stats = self.reward_stats
        if reward_stats:
            summary["reward_stats"] = reward_stats
        if self.rollout_time is not None:
            summary["rollout_time"] = self.rollout_time
        err = self.error_summary
        if err:
            summary["error_summary"] = err
        tags = self.tag_stats
        if tags:
            summary["tag_stats"] = tags
        return summary

    def _touch_created_at(self) -> None:
        if not self.created_at:
            self.created_at = int(time.time())

    @staticmethod
    def _summary_sort_key(item: dict[str, Any]) -> tuple[str, int]:
        return (
            str(item.get("training_run_id", "")),
            int(item.get("rollout_id", 0) or 0),
        )

    def _summary_item(self, *, export_size_bytes: int) -> dict[str, Any]:
        # summary_key keeps (run_id, rollout_id) uniqueness across runs.
        return {
            **self.to_summary(),
            "export_size_bytes": export_size_bytes,
            "summary_key": self.storage_key,
        }

    def save(self, *, is_async: bool = False) -> None | Awaitable[None]:
        self._touch_created_at()
        payload = self.model_dump(mode="json")
        export_payload = copy.deepcopy(payload)
        _apply_parsed(export_payload.get("samples"))
        export_size_bytes = len(
            (json.dumps(export_payload, ensure_ascii=False, indent=2) + "\n").encode()
        )
        return vol_put_with_summary(
            MetadataStore.TRAINING_ROLLOUTS,
            self.storage_key,
            payload,
            summary_store=MetadataStore.TRAINING_ROLLOUTS_SUMMARY,
            summary_item=self._summary_item(export_size_bytes=export_size_bytes),
            item_id_key="summary_key",
            sort_key=self._summary_sort_key,
            reverse=False,
            is_async=is_async,
        )

    @classmethod
    def list_summaries_for_run(
        cls, training_run_id: str
    ) -> list[TrainingRolloutSummary]:
        """Lightweight per-rollout summaries for one run, sorted by rollout_id."""
        items = vol_get_summary_items(MetadataStore.TRAINING_ROLLOUTS_SUMMARY) or []
        summaries: list[TrainingRolloutSummary] = []
        for item in items:
            if (
                not isinstance(item, dict)
                or item.get("training_run_id") != training_run_id
            ):
                continue
            try:
                summaries.append(TrainingRolloutSummary.model_validate(item))
            except ValidationError:
                continue
        return sorted(summaries, key=lambda summary: summary.rollout_id)
