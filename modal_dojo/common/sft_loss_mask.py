"""Validate pre-tokenized SFT rows or derive assistant masks from chat templates.

Template masks require stable conversation prefixes and tokenizer offsets.
Custom encoders supply explicit token IDs and masks instead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

# Rows carry these under ``metadata`` (Miles' ``--metadata-key``).
INPUT_IDS_KEY = "input_ids"
LOSS_MASK_KEY = "loss_mask"
# Miles' per-message switch: an assistant message with ``step_loss_mask`` != 1
# is rendered but not trained.
STEP_LOSS_MASK_KEY = "step_loss_mask"

Render = Callable[..., str]


class LossMaskError(ValueError):
    """A row whose loss mask cannot be built faithfully."""


def pretokenized(metadata: Any) -> tuple[list[int], list[int]] | None:
    """The row's own ``input_ids`` and ``loss_mask``, or None when it has none."""
    if not isinstance(metadata, dict):
        return None
    ids, mask = metadata.get(INPUT_IDS_KEY), metadata.get(LOSS_MASK_KEY)
    if ids is None and mask is None:
        return None
    if not isinstance(ids, list) or not isinstance(mask, list):
        raise LossMaskError(
            f"metadata.{INPUT_IDS_KEY} and metadata.{LOSS_MASK_KEY} must both be lists"
        )
    if len(ids) != len(mask):
        raise LossMaskError(
            f"metadata.{INPUT_IDS_KEY} has {len(ids)} tokens but "
            f"metadata.{LOSS_MASK_KEY} has {len(mask)} entries"
        )
    if any(type(t) is not int or t < 0 for t in ids):
        raise LossMaskError(f"metadata.{INPUT_IDS_KEY} must hold nonnegative integers")
    if any(not isinstance(m, (int, bool)) or m not in (0, 1) for m in mask):
        raise LossMaskError(f"metadata.{LOSS_MASK_KEY} must hold only 0 and 1")
    if 1 not in mask:
        raise LossMaskError(f"metadata.{LOSS_MASK_KEY} trains no token")
    if mask[0]:
        raise LossMaskError(
            "the first token needs loss_mask=0: it has no preceding context"
        )
    return list(ids), [int(m) for m in mask]


def template_render(tokenizer: Any, template_kwargs: dict | None = None) -> Render:
    """``render(messages, tools, add_generation_prompt)`` through the tokenizer's
    chat template."""
    if not getattr(tokenizer, "chat_template", None):
        raise LossMaskError(
            "the tokenizer has no chat template; pre-tokenize the rows "
            f"(metadata.{INPUT_IDS_KEY} and metadata.{LOSS_MASK_KEY}) or set "
            "chat_template_path"
        )
    kwargs = dict(template_kwargs or {})
    reserved = {
        "tools",
        "tokenize",
        "add_generation_prompt",
        "return_dict",
    } & kwargs.keys()
    if reserved:
        raise LossMaskError(
            "SFT controls these chat-template arguments: " + ", ".join(sorted(reserved))
        )

    def render(messages, tools=None, add_generation_prompt=False) -> str:
        return tokenizer.apply_chat_template(
            list(messages),
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    return render


def _trained(message: dict) -> bool:
    return (
        message.get("role") == "assistant" and message.get(STEP_LOSS_MASK_KEY, 1) == 1
    )


def assistant_spans(
    messages: Sequence[dict], render: Render, tools: list[dict] | None = None
) -> tuple[str, list[tuple[int, int]]]:
    """The rendered conversation and the character span of every trained
    assistant turn: from the end of the generation prompt before it to the end
    of the rendering through it (end-of-turn tokens included)."""
    full = render(messages, tools, add_generation_prompt=False)
    spans: list[tuple[int, int]] = []
    for i, message in enumerate(messages):
        if not _trained(message):
            continue
        before = render(messages[:i], tools, add_generation_prompt=True)
        through = render(messages[: i + 1], tools, add_generation_prompt=False)
        if not (through.startswith(before) and full.startswith(through)):
            raise LossMaskError(
                f"the chat template does not render message {i} as an extension "
                "of the messages before it (it rewrites earlier turns, e.g. by "
                "dropping past reasoning); pre-tokenize the rows instead"
            )
        if len(through) > len(before):
            spans.append((len(before), len(through)))
    return full, spans


def tokens_in_spans(
    tokenizer: Any, text: str, spans: Sequence[tuple[int, int]]
) -> tuple[list[int], list[int]]:
    """Mark every token overlapping a target, including boundary merges."""
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except NotImplementedError as err:  # slow tokenizers have no offsets
        raise LossMaskError("chat_template loss masks need a fast tokenizer") from err
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    mask = [0] * len(ids)
    j = 0
    ordered = sorted(spans)
    for k, (start, end) in enumerate(offsets):
        while j < len(ordered) and ordered[j][1] <= start:
            j += 1
        # Like Transformers' assistant masks, include a boundary token if any
        # of its characters belong to the target. Requiring full containment
        # silently drops answer text or merged whitespace (e.g. Qwen3.5's \n\n).
        if j < len(ordered) and end > start and end > ordered[j][0]:
            mask[k] = 1
    return list(ids), mask


def chat_template_loss_mask(
    tokenizer: Any,
    messages: Sequence[dict],
    tools: list[dict] | None = None,
    *,
    template_kwargs: dict | None = None,
) -> tuple[list[int], list[int]]:
    """Tokenize a conversation and mask its supervised assistant turns."""
    text, spans = assistant_spans(
        messages, template_render(tokenizer, template_kwargs), tools
    )
    if not spans:
        raise LossMaskError("the conversation has no assistant turn to train")
    ids, mask = tokens_in_spans(tokenizer, text, spans)
    if 1 not in mask:
        raise LossMaskError("no token falls inside an assistant turn")
    if mask[0]:
        raise LossMaskError("the first target token needs preceding context")
    return ids, mask
