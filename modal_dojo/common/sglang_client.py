"""Generic async client for SGLang ``/generate`` endpoints.

This module provides a reusable helper that any reward function, generate
function, or rollout function can use to call an SGLang server with
arbitrary request parameters.  Common SGLang fields are exposed as typed
keyword arguments; anything else is forwarded via ``**extra``.
"""

from __future__ import annotations

from typing import Any

import aiohttp


async def generate(
    url: str,
    *,
    input_ids: list[int] | None = None,
    text: str | None = None,
    sampling_params: dict[str, Any] | None = None,
    return_logprob: bool = False,
    logprob_start_len: int | None = None,
    top_logprobs_num: int | None = None,
    return_text_in_logprobs: bool | None = None,
    token_ids_logprob: list[int] | None = None,
    image_data: list[Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """POST to an SGLang ``/generate`` endpoint and return the JSON response.

    Parameters
    ----------
    url:
        Full URL of the ``/generate`` endpoint
        (e.g. ``http://host:port/generate``).
    input_ids:
        Token IDs to send.  Mutually exclusive with *text*.
    text:
        Raw text prompt.  Mutually exclusive with *input_ids*.
    sampling_params:
        SGLang sampling parameters (``temperature``, ``max_new_tokens``, …).
    return_logprob:
        Whether to return per-token log-probabilities.
    logprob_start_len:
        Position in the prompt from which to start returning logprobs.
        ``-1`` (SGLang default) returns logprobs for output tokens only.
    top_logprobs_num:
        Number of top-k logprobs to return at each position.
    return_text_in_logprobs:
        Detokenize tokens in the returned logprob entries.
    token_ids_logprob:
        Specific token IDs to return logprobs for.
    image_data:
        Multimodal image data (file paths, URLs, base64 strings, etc.).
    **extra:
        Any additional fields are merged directly into the JSON payload,
        allowing forward-compatibility with new SGLang parameters.
    """
    payload: dict[str, Any] = {}

    if input_ids is not None:
        payload["input_ids"] = input_ids
    if text is not None:
        payload["text"] = text
    if sampling_params is not None:
        payload["sampling_params"] = sampling_params
    if return_logprob:
        payload["return_logprob"] = True
    if logprob_start_len is not None:
        payload["logprob_start_len"] = logprob_start_len
    if top_logprobs_num is not None:
        payload["top_logprobs_num"] = top_logprobs_num
    if return_text_in_logprobs is not None:
        payload["return_text_in_logprobs"] = return_text_in_logprobs
    if token_ids_logprob is not None:
        payload["token_ids_logprob"] = token_ids_logprob
    if image_data is not None:
        payload["image_data"] = image_data

    payload.update(extra)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()
