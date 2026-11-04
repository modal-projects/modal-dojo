"""Template-independent SFT loss masks and Dojo's Miles SFT rollout."""

from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from modal_dojo.common.sft_loss_mask import (
    LossMaskError,
    chat_template_loss_mask,
    pretokenized,
)
from modal_dojo.frameworks.miles import sft_rollout
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

# Rendering a chat template needs jinja2, which Dojo does not depend on.
needs_jinja = pytest.mark.skipif(
    __import__("importlib").util.find_spec("jinja2") is None,
    reason="jinja2 not installed",
)

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
# Renders only the last assistant turn's content, as reasoning templates do
# with past thinking: earlier turns change when a new one is appended.
LAST_ONLY = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% if m.role != 'assistant' or loop.last %}{{ m.content }}{% endif %}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _tokenizer(template: str | None = CHATML) -> PreTrainedTokenizerFast:
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)
    fast.add_special_tokens(
        {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
    )
    fast.chat_template = template
    return fast


MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Bye"},
    {"role": "assistant", "content": "See you."},
]


def _trained_text(tok, ids, mask) -> list[str]:
    spans, cur = [], []
    for t, m in zip(ids, mask):
        if m:
            cur.append(t)
        elif cur:
            spans.append(tok.decode(cur))
            cur = []
    if cur:
        spans.append(tok.decode(cur))
    return spans


@needs_jinja
def test_chat_template_mask_marks_every_assistant_turn() -> None:
    tok = _tokenizer()
    ids, mask = chat_template_loss_mask(tok, MESSAGES)
    assert (
        ids
        == tok(
            tok.apply_chat_template(MESSAGES, tokenize=False), add_special_tokens=False
        )["input_ids"]
    )
    assert _trained_text(tok, ids, mask) == [
        "Hello!<|im_end|>\n",
        "See you.<|im_end|>\n",
    ]


@needs_jinja
def test_step_loss_mask_zero_turn_is_context_only() -> None:
    tok = _tokenizer()
    messages = [dict(m) for m in MESSAGES]
    messages[2]["step_loss_mask"] = 0
    ids, mask = chat_template_loss_mask(tok, messages)
    assert _trained_text(tok, ids, mask) == ["See you.<|im_end|>\n"]


@needs_jinja
def test_template_that_rewrites_past_turns_is_refused() -> None:
    with pytest.raises(LossMaskError, match="pre-tokenize"):
        chat_template_loss_mask(_tokenizer(LAST_ONLY), MESSAGES)


def test_tokenizer_without_template_is_refused() -> None:
    with pytest.raises(LossMaskError, match="no chat template"):
        chat_template_loss_mask(_tokenizer(None), MESSAGES)


def test_custom_render_replaces_the_template() -> None:
    tok = _tokenizer(None)

    def render(messages, tools=None, add_generation_prompt=False):
        text = "".join(f"[{m['role']}]{m['content']}[end]" for m in messages)
        return text + ("[assistant]" if add_generation_prompt else "")

    ids, mask = chat_template_loss_mask(tok, MESSAGES, render=render)
    assert _trained_text(tok, ids, mask) == ["Hello![end]", "See you.[end]"]


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({"input_ids": [1, 2], "loss_mask": [0]}, "has 2 tokens"),
        ({"input_ids": [1, 2], "loss_mask": [0, 2]}, "only 0 and 1"),
        ({"input_ids": [1, 2], "loss_mask": [0, 0]}, "trains no token"),
        ({"input_ids": [1, 2]}, "both be lists"),
    ],
)
def test_pretokenized_rows_are_validated(metadata, match) -> None:
    with pytest.raises(LossMaskError, match=match):
        pretokenized(metadata)


def test_pretokenized_absent() -> None:
    assert pretokenized({"tools": []}) is None
    assert pretokenized(None) is None


class _Buffer:
    def __init__(self, samples):
        self.samples = samples

    def get_samples(self, n):
        return [[s] for s in self.samples[:n]]


def _args(**kw):
    return SimpleNamespace(
        **{
            "rollout_global_dataset": True,
            "rollout_batch_size": 2,
            "loss_mask_type": "qwen",
            **kw,
        }
    )


def test_rollout_uses_pretokenized_rows_without_a_tokenizer(monkeypatch) -> None:
    monkeypatch.setattr(
        sft_rollout, "_tokenizer", lambda args: pytest.fail("tokenizer loaded")
    )
    sample = SimpleNamespace(
        prompt=[], metadata={"input_ids": [5, 6, 7, 8], "loss_mask": [0, 1, 0, 1]}
    )
    (group,) = sft_rollout.generate_rollout(
        _args(rollout_batch_size=1), 0, _Buffer([sample])
    )
    assert group[0].tokens == [5, 6, 7, 8]
    assert group[0].response_length == 3
    assert group[0].loss_mask == [1, 0, 1]


@needs_jinja
def test_rollout_chat_template_mode(monkeypatch) -> None:
    tok = _tokenizer()
    monkeypatch.setattr(sft_rollout, "_tokenizer", lambda args: tok)
    args = _args(modal_dojo_sft_loss_mask="chat_template")
    samples = [
        SimpleNamespace(prompt=MESSAGES, metadata={}),
        SimpleNamespace(prompt=MESSAGES[:3], metadata={}),
    ]
    out = sft_rollout.generate_rollout(args, 0, _Buffer(samples))
    first = out[0][0]
    assert sum(first.loss_mask) == sum(chat_template_loss_mask(tok, MESSAGES)[1])
    assert len(first.loss_mask) == first.response_length


def test_rollout_reads_mode_from_custom_config() -> None:
    args = _args(custom_config={"modal_dojo_sft_loss_mask": "chat_template"})
    assert sft_rollout.mask_mode(args) == "chat_template"


def _dataset():
    from modal_dojo.common.dataset import HuggingFaceDataset

    return HuggingFaceDataset(
        hf_repo="some/dataset",
        input_column="prompt",
        output_column="answer",
        input_format="text",
    )


def test_recipe_routes_chat_template_mask_through_config() -> None:
    recipe = MilesRecipe(loss_type="sft_loss", loss_mask_type="chat_template")
    assert recipe.extra_config["modal_dojo_sft_loss_mask"] == "chat_template"
    args = recipe.cli_args(dataset=_dataset())
    values = dict(zip(args, args[1:]))
    assert (
        values["--rollout-function-path"] == sft_rollout.__name__ + ".generate_rollout"
    )
    assert "--loss-mask-type" not in args


def test_lora_sft_is_train_only() -> None:
    args = MilesRecipe(loss_type="sft_loss", lora_rank=8).cli_args(dataset=_dataset())
    assert "--lora-train-only" in args
    rl = MilesRecipe(lora_rank=8).cli_args(dataset=_dataset())
    assert "--lora-train-only" not in rl


def test_recipe_keeps_native_mask_types_on_the_flag() -> None:
    recipe = MilesRecipe(loss_type="sft_loss", loss_mask_type="qwen3")
    assert "modal_dojo_sft_loss_mask" not in (recipe.extra_config or {})
    args = recipe.cli_args(dataset=_dataset())
    assert args[args.index("--loss-mask-type") + 1] == "qwen3"
