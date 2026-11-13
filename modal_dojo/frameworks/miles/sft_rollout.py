"""Miles' SFT rollout with template-independent loss masks.

Drop-in for ``miles.rollout.sft_rollout.generate_rollout``: each sample is
tokenized and masked here, then handed to the trainer exactly as Miles' own
rollout would. Per sample, in order:

1. pre-tokenized rows (``metadata.input_ids`` and ``metadata.loss_mask``) train
   on those tokens as given;
2. with ``loss_mask_type="chat_template"`` (``modal_dojo_sft_loss_mask`` in the
   run's config), every assistant turn is marked through the tokenizer's chat
   template (``modal_dojo.common.sft_loss_mask``);
3. otherwise Miles' ``MultiTurnLossMaskGenerator`` runs with ``--loss-mask-type``,
   as before. It is built only when a sample needs it, so a tokenizer without a
   chat template no longer fails before the first pre-tokenized row.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from modal_dojo.common.sft_loss_mask import (
    LossMaskError,
    chat_template_loss_mask,
    pretokenized,
)

logger = logging.getLogger(__name__)

__all__ = ["generate_rollout", "SFT_LOSS_MASK_KEY"]

# The run-config key the recipe sets for masks Miles' --loss-mask-type flag
# cannot name (its argparse choices are fixed).
SFT_LOSS_MASK_KEY = "modal_dojo_sft_loss_mask"
CHAT_TEMPLATE = "chat_template"

_TOKENIZER: Any = None
_MASK_GENERATOR: Any = None
_SAMPLE_LOGGED = False


def _tokenizer(args):
    global _TOKENIZER
    if _TOKENIZER is None:
        from miles.utils.processing_utils import load_tokenizer

        _TOKENIZER = load_tokenizer(
            args.hf_checkpoint,
            chat_template_path=getattr(args, "chat_template_path", None),
            trust_remote_code=True,
        )
    return _TOKENIZER


def _miles_mask(args, messages, tools):
    global _MASK_GENERATOR
    if _MASK_GENERATOR is None:
        from miles.utils.mask_utils import MultiTurnLossMaskGenerator

        _MASK_GENERATOR = MultiTurnLossMaskGenerator(
            _tokenizer(args), tokenizer_type=args.loss_mask_type
        )
    return _MASK_GENERATOR.get_loss_mask(messages, tools=tools)


def mask_mode(args) -> str | None:
    mode = getattr(args, SFT_LOSS_MASK_KEY, None)
    for container in ("extra_config", "custom_config"):
        values = getattr(args, container, None)
        if mode is None and isinstance(values, dict):
            mode = values.get(SFT_LOSS_MASK_KEY)
    return mode


def tokens_and_mask(args, sample) -> tuple[list[int], list[int], str]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if given := pretokenized(metadata):
        return (*given, "pretokenized")
    if mask_mode(args) == "pretokenized":
        raise LossMaskError(
            "pretokenized SFT requires metadata.input_ids and metadata.loss_mask"
        )
    tools = metadata.get("tools")
    if mask_mode(args) == CHAT_TEMPLATE:
        kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
        if isinstance(kwargs, str):
            kwargs = json.loads(kwargs)
        return (
            *chat_template_loss_mask(
                _tokenizer(args), sample.prompt, tools, template_kwargs=kwargs
            ),
            CHAT_TEMPLATE,
        )
    return (*_miles_mask(args, sample.prompt, tools), args.loss_mask_type)


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Samples for one SFT step, as ``miles.rollout.sft_rollout.generate_rollout``."""
    del rollout_id
    if evaluation or not args.rollout_global_dataset:
        raise ValueError("SFT replay requires the training dataset")

    global _SAMPLE_LOGGED
    samples = data_buffer.get_samples(args.rollout_batch_size)
    if len(samples) != args.rollout_batch_size:
        raise ValueError(
            "Incomplete SFT batch; provide at least one full batch of rows"
        )
    for i, group in enumerate(samples):
        (sample,) = group
        token_ids, loss_mask, source = tokens_and_mask(args, sample)
        if 1 not in loss_mask:
            raise LossMaskError(f"SFT sample {i} has no target token")
        context = getattr(args, "rollout_max_context_len", None)
        if context is not None and len(token_ids) > context:
            raise LossMaskError(
                f"SFT sample {i} has {len(token_ids)} tokens, exceeding "
                f"rollout_max_context_len={context}; split it before training"
            )
        response_length = len(loss_mask) - loss_mask.index(1)
        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]
        if i == 0 and not _SAMPLE_LOGGED:
            logger.info(
                "modal_dojo sft_rollout: %s mask, %d tokens, %d trained, response_length=%d",
                source,
                len(token_ids),
                sum(loss_mask),
                response_length,
            )
            _SAMPLE_LOGGED = True
    return samples
