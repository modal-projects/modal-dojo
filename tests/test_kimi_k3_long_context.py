import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from configs.kimi_k3_long_context import (
    generate_with_context_limit,
    remaining_response_tokens,
)
from scripts.kimi_k3_validation import append_tokens, observation_prefix


@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize("context", [65536, 131072])
@pytest.mark.parametrize("previous_tokens", [0, 1024, None])
@pytest.mark.parametrize("prompt_length", [2048, 131071])
def test_generation_bounds_requests_and_preserves_resumed_tokens(
    monkeypatch, evaluation, context, previous_tokens, prompt_length
):
    if previous_tokens is None:
        previous_tokens = max(0, context - prompt_length - 1)
    calls = []

    async def generate(args, sample, params, *, evaluation):
        if sample.response:
            params["max_new_tokens"] -= len(sample.tokens) - prompt_length
        calls.append((len(sample.tokens), params["max_new_tokens"], evaluation))
        return sample

    for name, attrs in {
        "miles.rollout.base_types": {"GenerateFnOutput": SimpleNamespace},
        "miles.rollout.sglang_rollout": {"generate": generate},
        "miles.utils.types": {
            "Sample": SimpleNamespace(Status=SimpleNamespace(TRUNCATED="truncated"))
        },
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    tokens = list(range(prompt_length + previous_tokens))
    sample = SimpleNamespace(
        prompt="problem", response="partial" if previous_tokens else "", tokens=tokens
    )
    params = {"max_new_tokens": context, "temperature": 1.0}
    input = SimpleNamespace(
        args=SimpleNamespace(sglang_context_length=context),
        state=SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda *a, **kw: range(prompt_length))
        ),
        sample=sample,
        sampling_params=params,
        evaluation=evaluation,
    )
    if prompt_length >= context - 1:
        with pytest.raises(ValueError, match="Prompt leaves no generation space"):
            asyncio.run(generate_with_context_limit(input))
        assert not calls
        return
    result = asyncio.run(generate_with_context_limit(input))
    assert result.samples is sample
    assert sample.tokens is tokens
    assert params["max_new_tokens"] == context
    if previous_tokens >= context - prompt_length - 1:
        assert not calls
        assert sample.status == "truncated"
    else:
        assert calls == [(len(tokens), context - len(tokens) - 1, evaluation)]


def test_budget_reserves_scheduler_boundary_and_accounts_for_history():
    assert remaining_response_tokens(131072, 2048, 131072) == 129023
    assert remaining_response_tokens(131072, 131070, 131072) == 1
    assert remaining_response_tokens(131072, 131071, 131072) == 0
    assert remaining_response_tokens(131072, 131073, 131072) == 0
    assert remaining_response_tokens(131072, 2048, 1024) == 1024


def test_observations_preserve_generated_ids_and_mask_alignment():
    sample = SimpleNamespace(
        tokens=[11, 12], response_length=0, rollout_log_probs=[], loss_mask=[]
    )
    append_tokens(sample, [31, 32], [-0.1, -0.2], generated=True)
    append_tokens(sample, [77, 78, 79], [0.0] * 3, generated=False)
    append_tokens(sample, [41], [-0.3], generated=True)
    assert sample.tokens == [11, 12, 31, 32, 77, 78, 79, 41]
    assert sample.loss_mask == [1, 1, 0, 0, 0, 1]
    assert sample.rollout_log_probs == [-0.1, -0.2, 0, 0, 0, -0.3]
    assert sample.response_length == 6
    with pytest.raises(ValueError, match="Non-finite"):
        append_tokens(sample, [99], [float("nan")], generated=True)
    assert sample.response_length == 6


def test_truncated_thinking_is_closed_but_completed_message_is_not_duplicated():
    closing = "<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>"
    user = '<|open|>message role="user"<|sep|>observation'
    suffix = closing + user
    assert observation_prefix(suffix, "still thinking").startswith(
        "<|close|>think<|sep|>"
    )
    assert observation_prefix(suffix, "answer" + closing) == user
    assert (
        observation_prefix(suffix, "<|close|>think<|sep|><|open|>response<|sep|>answer")
        == suffix
    )
