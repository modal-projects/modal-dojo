"""slime rollout that drives an ASR model via /v1/audio/transcriptions.

slime's default rollout posts chat-formatted prompts to /v1/chat/completions.
That works for Qwen-Omni (a chat model with audio content blocks), but Qwen3-ASR
is served by SGLang as a Whisper-style model on a *different* endpoint:

    POST /v1/audio/transcriptions
      file=<wav bytes>, model=<served>, temperature=<T>, prompt=<optional>

This is the slime glue layer: it pulls audio off the slime Sample, POSTs the clip
to a local SGLang engine, and writes the transcript + training fields back onto the
Sample. Generic audio decode lives in :mod:`modal_dojo.common.audio`; the
Qwen3-ASR tokenization in :mod:`modal_dojo.common.models.qwen3_asr_1_7b`.

Wire it in via ``SlimeRecipe.custom_generate_function`` (``Qwen3_ASR_1_7B_Recipe``
already does). slime calls it ``n_samples_per_prompt`` times per prompt, sampling
at ``sampling_params["temperature"]`` to give GRPO its N samples to score.

Single node only: engines are addressed on the local Ray node. Multi-node rollout
(cross-node engine discovery) is not supported here.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any  # slime's runtime arg/Sample objects have no public type

from modal_dojo.common.audio import coerce_audio_to_bytes
from modal_dojo.common.models.qwen3_asr_1_7b import encode_training_inputs


def _iter_content_items(prompt: Any):
    """Yield each content item from a slime conversation-list prompt."""
    if not isinstance(prompt, list):
        return
    for msg in prompt:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            yield from content


def _audio_ref(sample: Any) -> Any:
    """Pull the raw audio reference (data-URI / bytes) off a slime Sample.

    The audio transcription dataset returns ``False`` from
    ``apply_chat_template()``, so slime keeps ``sample.prompt`` a conversation list
    and the audio rides in the message content as
    ``{"type": "audio", "audio": <data-uri>}`` (slime's ``process_vision_info``
    extracts only images/videos, so audio never reaches ``multimodal_inputs``).
    Fail loudly if it's missing — a silent miss would train on audio-free prompts.

    This is the slime ``Sample -> data`` seam; decoding ``data -> bytes`` is the
    framework-agnostic :func:`modal_dojo.common.audio.coerce_audio_to_bytes`.
    """
    for item in _iter_content_items(getattr(sample, "prompt", None)):
        if isinstance(item, dict) and (item.get("type") == "audio" or "audio" in item):
            ref = item.get("audio") or item.get("audio_url")
            if ref:
                return ref
    raise RuntimeError(
        "transcription_rollout: no audio on the slime Sample. Expected a "
        "conversation-list prompt with a {'type': 'audio', 'audio': <data-uri>} item."
    )


def _engine_bases(args: Any) -> list[str]:
    """Local SGLang engine base URLs to try, in order.

    The slime router doesn't proxy /v1/audio/transcriptions (404s on its fixed Rust
    routes), so we POST to engines directly. Colocated engines bind to the Ray node
    IP at ports ``base_port, base_port + 2, …`` (one per GPU). We list a few host
    candidates (Ray node IP, then localhost) × those ports and try them in order;
    they're interchangeable, so the first reachable one wins.
    """
    import socket

    base_port = int(getattr(args, "sglang_server_port", None) or 15000)
    n_engines = int(
        getattr(args, "rollout_num_gpus", 0)
        or getattr(args, "actor_num_gpus_per_node", 1)
        or 1
    )
    ports = [base_port + 2 * i for i in range(max(n_engines, 1) + 1)]

    hosts: list[str] = []
    try:
        import ray

        hosts.append(ray.util.get_node_ip_address())
    except (ImportError, RuntimeError):
        pass
    try:
        hosts.append(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    hosts.append("127.0.0.1")

    bases = [f"http://{host}:{port}" for host in hosts if host for port in ports]
    return list(dict.fromkeys(bases))  # de-dup, preserve order


def _build_form(audio_bytes: bytes, model: str, temperature: float, prompt: str) -> Any:
    import aiohttp

    form = aiohttp.FormData()
    form.add_field(
        "file", io.BytesIO(audio_bytes), filename="clip.wav", content_type="audio/wav"
    )
    form.add_field("model", model)
    form.add_field("temperature", str(temperature))
    if prompt:
        form.add_field("prompt", prompt[:200])
    return form


async def _transcribe(
    session: Any,
    url: str,
    audio_bytes: bytes,
    model: str,
    temperature: float,
    prompt: str,
) -> str | None:
    """POST one clip to one engine, retrying transient 5xx with backoff.

    Returns the transcript, or ``None`` if this engine should be skipped (wrong
    endpoint, or transient failures exhausted) so the caller can try the next one.
    """
    import aiohttp

    backoff = 1.0
    for _ in range(3):
        try:
            form = _build_form(audio_bytes, model, temperature, prompt)
            async with session.post(url, data=form) as r:
                if r.status == 404:
                    return None  # wrong endpoint/host — try the next engine
                if r.status in (502, 503, 504):
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                r.raise_for_status()
                return ((await r.json()).get("text") or "").strip()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(backoff)
            backoff *= 2
    return None


def _populate_training_fields(args: Any, sample: Any, audio_bytes: bytes) -> None:
    """Map the model's encoded inputs onto the slime Sample for training.

    Delegates Qwen3-ASR tokenization (prompt + expanded ``<audio_pad>``, mel
    features) to the model module, then writes slime's token / response-length /
    loss-mask / multimodal fields. ``response_length`` and ``loss_mask`` cover the
    response only.
    """
    checkpoint = getattr(args, "hf_checkpoint", None) or getattr(
        args, "model_name", None
    )
    encoded = encode_training_inputs(
        checkpoint, getattr(sample, "prompt", ""), sample.response or "", audio_bytes
    )
    sample.tokens = encoded.prompt_ids + encoded.response_ids
    sample.response_length = len(encoded.response_ids)
    sample.loss_mask = [1] * len(encoded.response_ids)
    if encoded.multimodal_inputs:
        sample.multimodal_train_inputs = encoded.multimodal_inputs


async def transcription_rollout(args: Any, sample: Any, sampling_params: dict) -> Any:
    """Drive Qwen3-ASR (served on /v1/audio/transcriptions) for one sample.

    slime calls this ``n_samples_per_prompt`` times per prompt; GRPO's sampling
    diversity comes from temperature. We POST the clip, set ``sample.response`` to
    the transcript, then populate the trainer's token + multimodal fields so the
    actor's log-probs are audio-conditioned.
    """
    import aiohttp

    audio_bytes = coerce_audio_to_bytes(_audio_ref(sample))  # Sample -> data -> bytes
    if audio_bytes is None:
        raise RuntimeError(
            "transcription_rollout: the Sample's audio reference did not decode to "
            "bytes (expected raw bytes or a base64 / data-URI string)."
        )
    model = getattr(args, "served_model_name", None) or "qwen3-asr"
    temperature = float(sampling_params.get("temperature", 1.0))
    hint = getattr(sample, "prompt_text", "") or ""
    hint = hint.replace("<audio>", "").strip() if isinstance(hint, str) else ""

    bases = _engine_bases(args)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120)
    ) as session:
        for base in bases:
            url = base.rstrip("/") + "/v1/audio/transcriptions"
            text = await _transcribe(
                session, url, audio_bytes, model, temperature, hint
            )
            if text is not None:
                sample.response = text
                _populate_training_fields(args, sample, audio_bytes)
                return sample

    raise RuntimeError(
        f"transcription_rollout: no local engine served /v1/audio/transcriptions "
        f"(tried {bases})."
    )
