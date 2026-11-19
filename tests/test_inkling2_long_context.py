"""Numerical/gradient parity; run with `uv run --with torch pytest ...`."""

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from modal_dojo.frameworks.miles import inkling2_long_context as lc  # noqa: E402


@pytest.mark.parametrize("temperature", [0.7, 1.0])
@pytest.mark.parametrize("length", [2, 8, 9])
def test_streamed_scores_match_dense_values_and_lora_gradients(length, temperature):
    torch.manual_seed(31)
    hidden = torch.randn(length, 1, 5, requires_grad=True)
    base = torch.randn(13, 5)
    a = torch.randn(2, 5, requires_grad=True)
    b = torch.randn(13, 2, requires_grad=True)
    targets = torch.randint(0, 11, (1, length))
    calls = []

    def project(x):
        calls.append(len(x))
        return torch.nn.functional.linear(x, base) + (x @ a.T) @ b.T

    def scale(x):
        x = 3 * torch.tanh(x / 3)
        return x.masked_fill(torch.arange(13) >= 11, torch.finfo(x.dtype).min)

    dense_logits = scale(project(hidden))[:, 0, :11] / temperature
    lp = dense_logits.log_softmax(-1)
    dense = torch.stack(
        (
            lp[:-1].gather(-1, targets[0, 1:, None]).squeeze(-1),
            -(lp[:-1].exp() * lp[:-1]).sum(-1),
        ),
        -1,
    )
    dense = torch.cat((dense, hidden[-1:, 0, :1].expand(-1, 2) * 0)).unsqueeze(0)
    weights = torch.randn_like(dense)
    expected_grad = torch.autograd.grad(
        (dense * weights).sum(), (hidden, a, b), allow_unused=True
    )
    calls.clear()
    saved = []

    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        actual = lc.stream_token_scores(
            hidden,
            targets,
            project,
            scale,
            temperature=temperature,
            chunk_size=3,
            vocab_size=11,
        )
    actual_grad = torch.autograd.grad(
        (actual * weights).sum(), (hidden, a, b), allow_unused=True
    )
    torch.testing.assert_close(actual, dense)
    for actual_g, expected_g in zip(actual_grad, expected_grad, strict=True):
        if actual_g is None or expected_g is None:
            assert actual_g is expected_g
        else:
            torch.testing.assert_close(actual_g, expected_g, atol=2e-6, rtol=2e-5)
    assert all(rows <= 3 for rows in calls)
    assert not any(shape and shape[-1] == 13 for shape in saved)
    assert actual._modal_dojo_inkling2_scores
    with torch.no_grad():
        torch.testing.assert_close(
            lc.stream_token_scores(
                hidden,
                targets,
                project,
                scale,
                temperature=temperature,
                chunk_size=3,
                vocab_size=11,
            ),
            dense,
        )


def _residual_reference(sources, queries, *, normalize_values, partial_addend=None):
    values = torch.stack([s.float() for s in sources])
    if partial_addend is not None:
        values = torch.cat(
            (values[:-1], (values[-1] + partial_addend.float()).unsqueeze(0))
        )
    inv_rms = torch.rsqrt(values.square().mean(dim=-1) + 1e-6)
    scaled_queries = queries.float() * queries.shape[-1] ** -0.5
    logits = torch.einsum("n...d,qd,n...->q...n", values, scaled_queries, inv_rms)
    probabilities = torch.softmax(logits, dim=-1)
    if normalize_values:
        values = values * inv_rms.unsqueeze(-1)
    return list(torch.einsum("q...n,n...d->q...d", probabilities, values).unbind(0))


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("addend", [True, False])
def test_residual_chunks_preserve_values_and_gradients(normalize, addend):
    torch.manual_seed(7)
    sources = [torch.randn(11, 1, 5, requires_grad=True) for _ in range(4)]
    queries = torch.randn(3, 5, requires_grad=True)
    extra = torch.randn(11, 1, 5, requires_grad=True) if addend else None
    inputs = [*sources, queries, *([extra] if extra is not None else [])]
    expected = _residual_reference(
        sources, queries, normalize_values=normalize, partial_addend=extra
    )
    expected_grad = torch.autograd.grad(sum(x.square().sum() for x in expected), inputs)
    actual = lc.chunked_residual_mix(
        _residual_reference,
        sources,
        queries,
        normalize_values=normalize,
        partial_addend=extra,
        chunk_size=3,
    )
    actual_grad = torch.autograd.grad(sum(x.square().sum() for x in actual), inputs)
    for x, y in zip(actual, expected, strict=True):
        torch.testing.assert_close(x, y)
    for x, y in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-5)


def test_checkpoint_context_stores_each_completed_block_once():
    store = lc.CheckpointStore()
    blocks = [torch.randn(7, 1, 5) for _ in range(4)]
    saved = [store.pack_context(torch.stack(blocks[:n], dim=-2)) for n in (1, 2, 2, 4)]
    assert len(store.blocks) == 4
    assert saved[0][1][0] is saved[-1][1][0]
    assert sum(x.numel() for x in store.blocks) == 4 * 7 * 5
    for item, n in zip(saved, (1, 2, 2, 4), strict=True):
        torch.testing.assert_close(
            store.unpack_context(item), torch.stack(blocks[:n], dim=-2)
        )
    with pytest.raises(ValueError, match="append"):
        store.pack_context(torch.stack(blocks[:1], dim=-2))


def test_cpu_checkpoint_keeps_gradients_and_context(monkeypatch):
    from torch.utils.checkpoint import checkpoint

    def megatron_checkpoint(fn, distribute, *args):
        assert not distribute
        return checkpoint(fn, *args, use_reentrant=True)

    monkeypatch.setitem(
        sys.modules,
        "megatron.core",
        SimpleNamespace(
            tensor_parallel=SimpleNamespace(checkpoint=megatron_checkpoint)
        ),
    )
    h = torch.randn(3, 1, 5, requires_grad=True)
    c = torch.randn(3, 1, 2, 5, requires_grad=True)

    def fn(hidden, mask, context):
        return (hidden + context.sum(-2)).square()

    expected = torch.autograd.grad(fn(h, None, c).sum(), (h, c))
    token = lc._checkpoint_store.set(lc.CheckpointStore())
    try:
        result = lc.activation_checkpoint(fn, False, h, None, c)
    finally:
        lc._checkpoint_store.reset(token)
    result.sum().backward()
    actual = (h.grad, c.grad)
    for x, y in zip(actual, expected, strict=True):
        torch.testing.assert_close(x, y)


@pytest.mark.parametrize("window", [None, 2])
def test_query_chunks_preserve_full_attention_and_packed_boundaries(
    monkeypatch, window
):
    import torch.nn.attention.flex_attention as flex

    def dense_flex(q, k, v, *, score_mod, block_mask, scale, **kwargs):
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        scores = (q @ k.transpose(-1, -2)) * scale
        h = torch.arange(q.shape[1])[None, :, None, None]
        qi = torch.arange(q.shape[2])[None, None, :, None]
        ki = torch.arange(k.shape[2])[None, None, None, :]
        scores = score_mod(scores, 0, h, qi, ki)
        scores = scores.masked_fill(~block_mask.mask_mod(0, h, qi, ki), -torch.inf)
        return scores.softmax(-1) @ v, SimpleNamespace(lse=scores.logsumexp(-1))

    monkeypatch.setattr(
        flex, "create_block_mask", lambda fn, **kwargs: SimpleNamespace(mask_mod=fn)
    )
    monkeypatch.setattr(lc, "_flex_compiled", lambda: dense_flex)
    torch.manual_seed(5)
    q, k, v = [torch.randn(11, h, 4, requires_grad=True) for h in (4, 2, 2)]
    r = torch.randn(11, 4, 3, requires_grad=True)
    proj = torch.randn(3, 5, requires_grad=True)
    inputs = (q, k, v, r, proj)
    # A chunk boundary cuts across the two packed sequences (lengths 5 and 6).
    bias = torch.einsum("thd,de->the", r, proj).permute(1, 0, 2)
    qi, ki = torch.arange(11)[:, None], torch.arange(11)[None, :]
    distance = qi - ki
    segment = torch.tensor([0] * 5 + [1] * 6)
    mask = (distance >= 0) & (segment[:, None] == segment[None, :])
    if window is not None:
        mask &= distance <= window
    raw = torch.einsum("thd,shd->hts", q, k.repeat_interleave(2, 1)) / 2
    relative = bias[:, qi.expand(11, 11), distance.clamp(0, 4)]
    raw = (raw + torch.where(distance < 5, relative, 0)).masked_fill(~mask, -torch.inf)
    expected = torch.einsum("hts,shd->thd", raw.softmax(-1), v.repeat_interleave(2, 1))
    expected_lse = raw.logsumexp(-1).T
    expected_grad = torch.autograd.grad(
        expected.square().sum() + expected_lse.sum(), inputs
    )
    cache = {}
    actual, actual_lse = lc.relative_attention(
        *inputs, [5, 6], window=window, block_mask_cache=cache, chunk_size=3
    )
    actual_grad = torch.autograd.grad(actual.square().sum() + actual_lse.sum(), inputs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_lse, expected_lse)
    for x, y in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)
    assert len(cache) == 5  # four chunks and the layout key
