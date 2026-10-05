"""GPT-OSS-120B model and recipe wiring."""

from __future__ import annotations

from modal_dojo.frameworks.miles.modal_helpers.utils import (
    get_checkpoint_conversion_policy,
)
from modal_dojo.common.models import GPT_OSS_120B, parse_gpt_oss_response
from modal_dojo.train_recipes.miles_recipe import (
    GPT_OSS_120B_LoRA_Recipe,
    MilesRecipe,
)


def _flags(args: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for i, token in enumerate(args):
        if token.startswith("--"):
            nxt = args[i + 1] if i + 1 < len(args) else None
            out[token] = None if nxt is None or nxt.startswith("--") else nxt
    return out


def test_base_recipe_is_the_lora_recipe() -> None:
    recipe = MilesRecipe.get_base_recipe(GPT_OSS_120B())
    assert isinstance(recipe, GPT_OSS_120B_LoRA_Recipe)
    assert recipe.megatron_to_hf_mode == "bridge"
    assert recipe.lora_rank == recipe.sglang_max_lora_rank == 32


def test_recipe_emits_gpt_oss_attention_flags() -> None:
    model = GPT_OSS_120B()
    flags = _flags(GPT_OSS_120B_LoRA_Recipe().cli_args(model=model))
    assert flags["--hf-checkpoint"] == "openai/gpt-oss-120b"
    assert flags["--num-layers"] == "36"
    assert flags["--num-experts"] == "128"
    assert flags["--moe-router-topk"] == "4"
    assert flags["--softmax-type"] == "learnable"
    assert flags["--window-size"] == "128,0"
    assert flags["--window-attn-skip-freq"] == "2"
    assert flags["--max-position-embeddings"] == "131072"
    assert flags["--rotary-base"] == "150000"
    assert flags["--qkv-format"] == "bshd"
    assert "--use-dynamic-batch-size" not in flags
    assert flags["--micro-batch-size"] == "1"
    # GPT-OSS keeps linear biases and has no QK layernorm.
    assert "--disable-bias-linear" not in flags
    assert "--qk-layernorm" not in flags
    assert "--ref-load" not in flags


def test_conversion_args_carry_sliding_window_flags() -> None:
    recipe = GPT_OSS_120B_LoRA_Recipe(megatron_to_hf_mode="raw")
    _, _, extra = get_checkpoint_conversion_policy(recipe, model=GPT_OSS_120B())
    joined = " ".join(extra)
    assert "--softmax-type learnable" in joined
    assert "--window-size 128,0" in joined
    assert "--window-attn-skip-freq 2" in joined
    assert "--max-position-embeddings 131072" in joined


def test_parse_harmony_channels() -> None:
    text = (
        "<|channel|>analysis<|message|>Compute 2+2.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>\\boxed{4}<|return|>"
    )
    parsed = parse_gpt_oss_response(text)
    assert parsed.content == "\\boxed{4}"
    assert parsed.thinking == "Compute 2+2."
    assert parsed.tool_calls == []


def test_parse_harmony_tool_call() -> None:
    text = (
        "<|channel|>analysis<|message|>Need weather.<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.get_weather "
        '<|constrain|>json<|message|>{"city": "Paris"}<|call|>'
    )
    parsed = parse_gpt_oss_response(text)
    assert parsed.content == ""
    assert parsed.thinking == "Need weather."
    assert [(c.name, c.arguments) for c in parsed.tool_calls] == [
        ("get_weather", {"city": "Paris"})
    ]


def test_parse_without_special_tokens() -> None:
    parsed = parse_gpt_oss_response("analysisThink hard.assistantfinalThe answer is 4.")
    assert parsed.content == "The answer is 4."
    assert parsed.thinking == "Think hard."
    assert parse_gpt_oss_response("plain text").content == "plain text"


def test_parse_truncated_mid_channel() -> None:
    parsed = parse_gpt_oss_response("<|channel|>analysis<|message|>Still thinking")
    assert parsed.content == ""
    assert parsed.thinking == "Still thinking"
