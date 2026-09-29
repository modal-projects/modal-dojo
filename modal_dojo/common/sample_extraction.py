"""Trace / image / trajectory extraction from framework Samples for dashboard
rollout reporting.

Shared by the slime and miles frameworks (miles is a slime fork; its Sample
carries the same prompt/response/reward/metadata/multimodal fields). Everything
here is duck-typed so neither framework need be importable; heavy optional deps
(PIL) are imported lazily inside the functions that need them.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import json
import os
from typing import Any

from modal_dojo.common.coerce import optional_int
from modal_dojo.common.sample import Sample

# Import path of the run model's response parser (a (str) -> ParsedResponse
# callable). The launcher exports it; the recorder resolves and applies it.
RESPONSE_PARSER_PATH_ENV = "MODAL_DOJO_RESPONSE_PARSER_PATH"
# Per-sample execution-trace capture (off by default — traces inflate rollout
# payloads). When on, only the first TRACE_SAMPLE_LIMIT_ENV samples of each
# rollout carry a trace, and only timing/scalar attributes are kept.
CAPTURE_TRACE_ENV = "MODAL_DOJO_CAPTURE_TRACE"
TRACE_SAMPLE_LIMIT_ENV = "MODAL_DOJO_TRACE_SAMPLE_LIMIT"
_TRACE_SAMPLE_LIMIT_DEFAULT = 16
# Backstops so a pathological trace can't blow up the payload we already keep
# small by sampling + dropping payload-bearing attributes.
_TRACE_MAX_SPANS = 256
_TRACE_ATTR_STR_MAX = 200
# Input-image capture (image-modality runs). The limit bounds *distinct* images, not
# samples: a prompt group's ``n_samples_per_prompt`` samples share one screenshot. Each
# distinct image is thumbnailed and size-capped before it goes on the payload.
IMAGE_SAMPLE_LIMIT_ENV = "MODAL_DOJO_IMAGE_SAMPLE_LIMIT"
_IMAGE_LIMIT_DEFAULT = 16
_IMAGE_MAX_DIM = 512
_IMAGE_MAX_BYTES = 256 * 1024
_IMAGE_REF_CHARS = 16
# Per-sample multi-turn trajectory capture. The full ``trajectory_messages`` blob
# (agent transcripts, many turns) is large, so only the first
# TRAJECTORY_SAMPLE_LIMIT_ENV samples of each rollout carry it and each message's
# content is length-capped — enough for the dashboard's ConversationView to render
# the conversation without bloating the rollout payload. Without it, multi-turn
# rollouts (e.g. toolathlon) collapse to a single flat block on the dashboard.
TRAJECTORY_SAMPLE_LIMIT_ENV = "MODAL_DOJO_TRAJECTORY_SAMPLE_LIMIT"
_TRAJECTORY_SAMPLE_LIMIT_DEFAULT = 16
_TRAJECTORY_MSG_CHARS_MAX = 8000
_TRAJECTORY_MAX_MESSAGES = 128


# Keys handled separately below (compacted/size-limited their own way) —
# never copy them through the generic passthrough too, or they'd end up
# duplicated under the same name with two different shapes.
_RESERVED_METADATA_KEYS = frozenset(
    {"trajectory_messages", "eval_report", "image", "image_ref"}
)
# Per-tag cap on the generic metadata passthrough so a stray large value a
# reward function stashes on the sample can't bloat the rollout payload.
_MAX_TAG_VALUE_BYTES = 2048


def _is_small_json_value(value: Any, max_bytes: int = _MAX_TAG_VALUE_BYTES) -> bool:
    """Best-effort check that ``value`` is JSON-serializable and small.

    Reward/rollout functions can stash arbitrary tags on ``sample.metadata``;
    this keeps the passthrough safe (skip non-serializable values) and
    bounded (skip oversized ones) without requiring callers to sanitize.
    """
    try:
        return len(json.dumps(value)) <= max_bytes
    except (TypeError, ValueError):
        return False


def _resolve_hook(path: str | None) -> Any:
    if not path:
        return None
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        return None
    module = importlib.import_module(module_name)
    return getattr(module, attr, None)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return ""
    if isinstance(value, str):
        return value
    return str(value)


# Metadata key for the dashboard score when ``sample.reward`` is not a float
# yet (common with OPD: reward holds the teacher ``/generate`` payload until
# post-process). Custom generate / RM hooks should set:
#
#   sample.metadata["shaped_reward"] = float(task_score)
#
# Extraction maps that onto gym ``Sample.score``; do not grow gym Sample with
# slime/OPD fields like ``reward: dict``.
_SHAPED_REWARD_KEY = "shaped_reward"


def _sample_score(sample: Sample, reward: float | None = None) -> float:
    """Resolve reward score from a Modal Dojo Sample.

    Resolution order:

    1. Numeric ``reward`` (normal GRPO / post-process).
    2. ``sample.metadata["shaped_reward"]`` — set this in custom generate/RM
       when the framework reward must stay non-scalar until later (OPD).
    3. ``0.0``.
    """
    if reward is not None:
        return float(reward)

    value = sample.metadata.get(_SHAPED_REWARD_KEY)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    return 0.0


_RESPONSE_PARSER: Any = None
_RESPONSE_PARSER_LOADED = False


def _response_parser() -> Any:
    """Resolve the run model's response parser — a ``(str) -> ParsedResponse``
    callable — from the import path the launcher exports in
    ``RESPONSE_PARSER_PATH_ENV``. Cached per process; ``None`` if unset."""
    global _RESPONSE_PARSER, _RESPONSE_PARSER_LOADED
    if not _RESPONSE_PARSER_LOADED:
        _RESPONSE_PARSER_LOADED = True
        _RESPONSE_PARSER = _resolve_hook(
            os.environ.get(RESPONSE_PARSER_PATH_ENV, "").strip()
        )
    return _RESPONSE_PARSER


def _parsed_response_dict(text: str, parser: Any) -> dict[str, Any] | None:
    """Run ``text`` through ``parser`` and return a JSON-able dict, or ``None``
    when there's no text or no parser."""
    if parser is None or not text:
        return None
    try:
        parsed = parser(text)
    except Exception:
        return None
    return {
        "content": getattr(parsed, "content", "") or "",
        "thinking": getattr(parsed, "thinking", None),
        "tool_calls": [
            {"name": tc.name, "arguments": tc.arguments}
            for tc in (getattr(parsed, "tool_calls", None) or [])
        ],
    }


def _trace_enabled() -> bool:
    return os.environ.get(CAPTURE_TRACE_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _trace_sample_limit() -> int:
    try:
        n = int(os.environ.get(TRACE_SAMPLE_LIMIT_ENV, "").strip())
    except (TypeError, ValueError):
        return _TRACE_SAMPLE_LIMIT_DEFAULT
    return max(0, n)


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trace_scalar(value: Any) -> Any:
    """Keep only small scalar attributes (timings/counts/flags). Drop lists,
    dicts, and long strings so trace payloads can't carry response/tool data
    that already lives on the Sample."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _TRACE_ATTR_STR_MAX else None
    return None


def _trace_attrs(raw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            scalar = _trace_scalar(value)
            if scalar is not None:
                out[str(key)] = scalar
    return out


def _normalize_span(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    get = raw.get
    name = get("name") or get("event") or get("span") or ""
    start = _coerce_float(get("start", get("start_time", get("ts", get("timestamp")))))
    end = _coerce_float(get("end", get("end_time")))
    if start is None:
        # An instant event may only carry an end/ts; fall back to it.
        start = end
    if start is None:
        return None
    parent = get("parent") or get("parent_name")
    return {
        "name": str(name),
        "start": start,
        "end": end if (end is not None and end != start) else None,
        "attributes": _trace_attrs(get("attributes") or get("attrs") or {}),
        "parent": str(parent) if parent else None,
    }


def _normalize_trace(raw: Any) -> list[dict[str, Any]] | None:
    """Coerce slime's per-sample trace (shape varies by version) into a list of
    ``{name, start, end, attributes, parent}`` spans rebased to start at 0.
    Returns ``None`` when there's nothing usable."""
    spans: list[Any] | None = None
    if isinstance(raw, list):
        spans = raw
    elif isinstance(raw, dict):
        for key in ("spans", "events", "timeline"):
            value = raw.get(key)
            if isinstance(value, list):
                spans = (spans or []) + value
        if spans is None:
            # A dict keyed by span name -> attrs/timing.
            spans = [{"name": k, **v} for k, v in raw.items() if isinstance(v, dict)]
    if not spans:
        return None
    out: list[dict[str, Any]] = []
    for entry in spans[:_TRACE_MAX_SPANS]:
        norm = _normalize_span(entry)
        if norm is not None:
            out.append(norm)
    if not out:
        return None
    base = min(s["start"] for s in out)
    for s in out:
        s["start"] = round(s["start"] - base, 6)
        if s["end"] is not None:
            s["end"] = round(s["end"] - base, 6)
    return out


def _extract_trace(sample: Any) -> Any:
    """Pull slime's trace carrier off a sample — it lives either as a ``trace``
    attribute/key or nested under ``metadata['trace']``, depending on version."""
    if isinstance(sample, dict):
        raw = sample.get("trace")
        if raw is None and isinstance(sample.get("metadata"), dict):
            raw = sample["metadata"].get("trace")
        return raw
    raw = getattr(sample, "trace", None)
    if raw is None:
        meta = getattr(sample, "metadata", None)
        if isinstance(meta, dict):
            raw = meta.get("trace")
    return raw


def _duck_get(obj: Any, key: str, default: Any = None) -> Any:
    """Unified field access for dict or object."""
    return (
        obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    )


def _extract_inference_metadata(sample: Any) -> dict[str, Any] | None:
    """Extract per-sample inference stats: token counts and prefix cache info."""
    prefix_info = _duck_get(sample, "prefix_cache_info")
    if prefix_info is None:
        return None

    total = _coerce_int(_duck_get(prefix_info, "total_prompt_tokens", 0))
    cached = _coerce_int(_duck_get(prefix_info, "cached_tokens", 0))
    resp_len = _duck_get(sample, "response_length")

    inference: dict[str, Any] = {
        "tokens_in": total,
        "cached_tokens": cached,
        "new_tokens": max(0, total - cached),
        "cache_hit_rate": cached / total if total else 0.0,
    }
    if resp_len is not None:
        inference["tokens_out"] = _coerce_int(resp_len)

    return inference


def _extract_audio_from_prompt(prompt: Any) -> str | None:
    """Pull a browser-playable audio data-URI out of a conversation-list prompt.

    Multimodal datasets keep audio as ``{"type": "audio", "audio": "<data-uri>"}``
    inside message content lists.  Returns ``None`` for plain-text prompts.
    """
    if not isinstance(prompt, list):
        return None
    for msg in prompt:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and (
                item.get("type") == "audio" or "audio" in item
            ):
                ref = item.get("audio") or item.get("audio_url")
                if isinstance(ref, str) and ref:
                    return ref
    return None


def _image_limit() -> int:
    """Max *distinct* images one rollout payload may carry (0 disables capture)."""
    try:
        n = int(os.environ.get(IMAGE_SAMPLE_LIMIT_ENV, "").strip())
    except (TypeError, ValueError):
        return _IMAGE_LIMIT_DEFAULT
    return max(0, n)


def _trajectory_sample_limit() -> int:
    try:
        n = int(os.environ.get(TRAJECTORY_SAMPLE_LIMIT_ENV, "").strip())
    except (TypeError, ValueError):
        return _TRAJECTORY_SAMPLE_LIMIT_DEFAULT
    return max(0, n)


def _compact_trajectory_messages(raw: Any) -> list[dict[str, Any]] | None:
    """Coerce a Sample's ``trajectory_messages`` into a compact, size-bounded
    list of ``{role, content}`` turns the dashboard can render. Caps the number
    of messages and truncates each long content (head+tail with an elision
    notice) so a single verbose episode can't blow up the rollout payload."""
    if not isinstance(raw, list) or not raw:
        return None
    out: list[dict[str, Any]] = []
    for msg in raw[:_TRAJECTORY_MAX_MESSAGES]:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            if content is None:
                content = ""
            else:
                try:
                    content = json.dumps(content)
                except TypeError:
                    content = str(content)
        if len(content) > _TRAJECTORY_MSG_CHARS_MAX:
            edge = _TRAJECTORY_MSG_CHARS_MAX // 2
            elided = len(content) - 2 * edge
            content = (
                f"{content[:edge]}\n\n… {elided} chars elided …\n\n{content[-edge:]}"
            )
        entry: dict[str, Any] = {
            "role": str(msg.get("role") or "unknown"),
            "content": content,
        }
        if msg.get("exit_status") is not None:
            entry["exit_status"] = msg["exit_status"]
        out.append(entry)
    return out or None


def _image_to_data_uri(value: Any) -> str | None:
    """Coerce an image reference into a browser-renderable, thumbnailed data-URI.

    Handles PIL Images (how slime's ``process_vision_info`` stores them on the
    Sample), raw bytes, and ``data:``/``http(s)`` strings. Downscales to a
    thumbnail and skips anything that's still too big, so a full-res screenshot
    can't blow up the rollout payload. Remote URLs pass through untouched (the
    browser fetches them — no payload cost). Returns ``None`` on any failure.
    """
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return value
    try:
        from PIL import Image

        img: Any = None
        if isinstance(value, str):
            if value.startswith("data:"):
                _, _, b64 = value.partition(",")
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
            else:
                img = Image.open(value)  # filesystem path
        elif isinstance(value, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(value)))
        elif hasattr(value, "save"):  # already a PIL Image
            img = value
        if img is None:
            return None

        img = img.convert("RGB")
        img.thumbnail((_IMAGE_MAX_DIM, _IMAGE_MAX_DIM))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        mime = "image/png"
        if len(data) > _IMAGE_MAX_BYTES:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            data = buf.getvalue()
            mime = "image/jpeg"
        if len(data) > _IMAGE_MAX_BYTES:
            return None
        return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    except Exception:
        # PIL missing or decode failed: pass through a small inline data-URI as-is.
        if (
            isinstance(value, str)
            and value.startswith("data:")
            and len(value) <= _IMAGE_MAX_BYTES * 2
        ):
            return value
        return None


def _image_candidates(sample: Any) -> list[Any]:
    """Unencoded image references on a slime Sample, best first.

    An image a reward/rollout function stashes on ``sample.metadata["image"]``
    (e.g. a rendering of the model's output) comes first. For image-modality
    runs slime's ``process_vision_info`` lifts the screenshot into
    ``sample.multimodal_inputs['images']`` (even when ``apply_chat_template``
    collapses the prompt to a string). Falls back to image items in a
    conversation-list prompt. Returns ``[]`` for non-image samples.
    """
    if isinstance(sample, dict):
        get = sample.get
    else:

        def get(key: str, default: Any = None) -> Any:
            return getattr(sample, key, default)

    candidates: list[Any] = []
    meta = get("metadata", None)
    if isinstance(meta, dict) and meta.get("image") is not None:
        candidates.append(meta["image"])

    mm = get("multimodal_inputs", None)
    if isinstance(mm, dict):
        imgs = mm.get("images") or mm.get("image")
        if imgs is not None:
            candidates.extend(imgs if isinstance(imgs, list) else [imgs])

    prompt = get("prompt", None)
    if isinstance(prompt, list):
        for msg in prompt:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if (
                    item.get("type") == "image"
                    or "image" in item
                    or "image_url" in item
                ):
                    ref = item.get("image") or item.get("image_url")
                    if isinstance(ref, dict):
                        ref = ref.get("url")
                    if ref is not None:
                        candidates.append(ref)

    return candidates


def _raw_image_key(value: Any) -> Any:
    """Hashable identity for an *unencoded* image candidate.

    Recognises a repeat before paying for a decode + thumbnail + base64 pass.
    Content-derived wherever the value exposes its bytes; ``id()`` is the last
    resort, and the store pins those objects so the id cannot be recycled.
    """
    if isinstance(value, str):
        if len(value) <= 512:
            return value
        return ("s", hashlib.md5(value.encode()).hexdigest())
    if isinstance(value, (bytes, bytearray)):
        return ("b", hashlib.md5(bytes(value)).hexdigest())
    tobytes = getattr(value, "tobytes", None)
    if callable(tobytes):
        try:
            return (
                "c",
                getattr(value, "mode", ""),
                getattr(value, "size", ()),
                hashlib.md5(tobytes()).hexdigest(),
            )
        except Exception:
            pass
    return ("o", id(value))


class RolloutImageStore:
    """Deduplicates images across one rollout's samples.

    Covers both input images (multimodal prompts) and output images a reward
    function attaches as ``sample.metadata["image"]``; every candidate is
    thumbnailed and size-capped before it rides on the payload.

    The bytes ride on the first sample that uses them as ``metadata["image"]``; the
    rest of the prompt group carries only ``metadata["image_ref"]`` naming it. Both
    keys live in sample metadata rather than a new rollout-level field, which a
    dashboard one deploy behind the trainer would drop while keeping the refs.

    ``limit`` bounds distinct images. Once spent, further images are skipped and
    their samples left unannotated, while samples sharing an already-captured image
    still resolve — coverage degrades by prompt group instead of dangling refs.

    Frameworks build a prompt group by copying one prepared sample (miles
    deep-copies it per ``n_samples_per_prompt``), so members hold equal-content but
    distinct image objects. Content keying is what collapses them, so every sample
    is resolved from its own candidates.
    """

    def __init__(self, limit: int) -> None:
        self._limit = max(0, limit)
        self._ref_by_uri: dict[str, str] = {}
        # "" marks a candidate already known to be unusable, so it is not retried
        # once per sample.
        self._ref_by_raw: dict[Any, str] = {}
        self._key_by_id: dict[int, Any] = {}
        self._pinned: list[Any] = []
        self._closed = False

    @property
    def count(self) -> int:
        """Distinct images captured so far."""
        return len(self._ref_by_uri)

    def _memo_key(self, candidate: Any) -> Any:
        """The key already computed for this exact object, or ``None``.

        Covers a prompt group that shares one image object, so the content key -- a
        full raw-pixel materialisation plus a hash for a PIL image -- is not paid
        once per sample. Content keying still decides equality: this only
        short-circuits the same object, which cannot have different content within
        one payload.
        """
        return self._key_by_id.get(id(candidate))

    def _key(self, candidate: Any) -> Any:
        """Key ``candidate`` by content, memoised on the object."""
        memo = self._memo_key(candidate)
        if memo is not None:
            return memo
        key = _raw_image_key(candidate)
        # Pinning keeps id() from being recycled onto another image while the
        # memo (or an identity-derived key) is live.
        self._pinned.append(candidate)
        self._key_by_id[id(candidate)] = key
        return key

    def annotate(self, sample: Any, metadata: dict[str, Any]) -> bool:
        """Write this sample's image keys onto ``metadata``; True if one resolves."""
        if not self._limit:
            return False
        return bool(self._resolve(sample, metadata))

    def _resolve(self, sample: Any, metadata: dict[str, Any]) -> str:
        """Annotate from ``sample``'s own candidates; the ref written, else ``""``."""
        for candidate in _image_candidates(sample):
            key = self._memo_key(candidate)
            if key is None:
                if self._closed:
                    continue
                key = self._key(candidate)
            cached = self._ref_by_raw.get(key)
            if cached is not None:
                if not cached:
                    continue
                metadata["image_ref"] = cached
                return cached
            if self.count >= self._limit:
                self._ref_by_raw[key] = ""
                self._closed = True
                continue
            uri = _image_to_data_uri(candidate)
            if not uri:
                self._ref_by_raw[key] = ""
                continue
            ref = self._ref_by_uri.get(uri)
            if ref is None:
                ref = hashlib.md5(uri.encode()).hexdigest()[:_IMAGE_REF_CHARS]
                self._ref_by_uri[uri] = ref
                metadata["image"] = uri
            self._ref_by_raw[key] = ref
            metadata["image_ref"] = ref
            return ref
        return ""


def _sample_to_dict(
    sample: Any,
    parser: Any = None,
    *,
    include_trace: bool = False,
    image_store: RolloutImageStore | None = None,
    include_trajectory: bool = False,
    n_samples_per_prompt: int = 1,
) -> dict[str, Any]:
    """Best-effort extraction of (prompt, response, reward, metadata) from a
    slime Sample-like object. Duck-typed so we don't import slime here."""
    if isinstance(sample, dict):
        attrs = sample
        get = sample.get
    else:
        attrs = None

        def get(key: str, default: Any = None) -> Any:
            return getattr(sample, key, default)

    prompt = get("prompt") if attrs is not None else get("prompt", "")
    response = get("response") if attrs is not None else get("response", "")
    reward = get("reward") if attrs is not None else get("reward", None)

    metadata: dict[str, Any] = {}
    for key in ("response_length", "prompt_length", "rollout_id", "rollout_idx"):
        value = get(key) if attrs is not None else get(key, None)
        if value is not None:
            metadata[key] = value
    if inference := _extract_inference_metadata(sample):
        metadata["inference"] = inference

    # Pull the sample's own metadata dict through to the dashboard so it can
    # render exit status, eval checks, custom reward-function tags, etc.
    # Reserved keys are handled separately below (their own compaction), and
    # oversized/non-serializable values are dropped rather than bloating the
    # rollout payload.
    sample_meta = get("metadata") if attrs is not None else get("metadata", None)
    if isinstance(sample_meta, dict):
        for key, value in sample_meta.items():
            if key in _RESERVED_METADATA_KEYS or value is None:
                continue
            if _is_small_json_value(value):
                metadata[key] = value
        # Multi-turn trajectory (capped to the first N samples): lets the
        # dashboard's ConversationView render the full agent conversation.
        # Without it, multi-turn rollouts collapse to a single flat block.
        if include_trajectory:
            trajectory = _compact_trajectory_messages(
                sample_meta.get("trajectory_messages")
            )
            if trajectory:
                metadata["trajectory_messages"] = trajectory
        # Store eval_report but only the compact summary/checks, not huge blobs
        eval_report = sample_meta.get("eval_report")
        if isinstance(eval_report, dict):
            compact: dict[str, Any] = {}
            if "check_summary" in eval_report:
                compact["check_summary"] = eval_report["check_summary"]
            if "checks" in eval_report and isinstance(eval_report["checks"], dict):
                compact["checks"] = {
                    name: {
                        "passed": bool(detail.get("passed")),
                        "status": str(detail.get("status", "")),
                        "score": detail.get("score"),
                        "errors": (detail.get("errors") or [])[:3],
                    }
                    for name, detail in eval_report["checks"].items()
                    if isinstance(detail, dict)
                }
            if compact:
                metadata["eval_report"] = compact

    sample_index = optional_int(get("index"))
    group_index = optional_int(get("group_index"))
    if group_index is None and sample_index is not None:
        group_index = sample_index // max(1, int(n_samples_per_prompt or 1))

    prompt_text = _coerce_text(prompt)
    if audio_uri := _extract_audio_from_prompt(prompt):
        metadata["_metadata_type"] = "audio"
        metadata["audio"] = audio_uri
    elif image_store is not None and image_store.annotate(sample, metadata):
        metadata["_metadata_type"] = "image"

    response_text = _coerce_text(response)
    # Score via gym Sample: numeric reward, else metadata["shaped_reward"] (OPD).
    numeric_reward = (
        float(reward)
        if isinstance(reward, (int, float)) and not isinstance(reward, bool)
        else None
    )
    score = _sample_score(Sample(metadata=metadata), numeric_reward)
    out: dict[str, Any] = {
        "score": score,
        "prompt": prompt_text,
        "response": response_text,
        "metadata": metadata,
    }
    rollout_index = optional_int(get("rollout_id"))
    if rollout_index is None:
        rollout_index = sample_index
    if rollout_index is not None:
        out["rollout_index"] = rollout_index
    if sample_index is not None:
        out["sample_index"] = sample_index
    if group_index is not None:
        out["group_index"] = group_index
    # Store raw + parsed (mirrors eval's EvalRowResult) so the dashboard can show
    # cleaned content without re-parsing. Parsing happens here, in the recorder.
    parsed = _parsed_response_dict(response_text, parser)
    if parsed is not None:
        out["parsed_response"] = parsed
    if include_trace:
        trace = _normalize_trace(_extract_trace(sample))
        if trace:
            out["trace"] = trace
    return out


def _metrics_to_dict(metrics: Any) -> dict[str, Any]:
    if isinstance(metrics, dict):
        return {str(k): v for k, v in metrics.items()}
    return {}
