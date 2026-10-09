from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_dojo.frameworks.slime.modal_helpers.patches import (
    patch_entropy_no_grad as patcher,
)

FIXTURES = Path(__file__).parent / "testdata/slime"


@pytest.fixture
def sources(tmp_path):
    ppo = tmp_path / "slime/utils/ppo_utils.py"
    loss = tmp_path / "slime/backends/megatron_utils/loss.py"
    for path, name in ((ppo, "ppo_utils"), (loss, "loss")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((FIXTURES / f"{name}.entropy.input").read_text())
    return tmp_path, ppo, loss


def test_patch_is_idempotent_and_command_embeds_script(sources):
    root, ppo, loss = sources
    patcher.patch(root)
    once = ppo.read_text(), loss.read_text()
    patcher.patch(root)
    assert (ppo.read_text(), loss.read_text()) == once
    command = patcher.image_patch_command()
    assert base64.b64decode(command.split()[1]) == Path(patcher.__file__).read_bytes()


def test_source_drift_fails_before_writing_either_file(sources):
    root, ppo, loss = sources
    loss.write_text(loss.read_text().replace("log_prob_keep_mask=top_p_keep_mask,", ""))
    before = ppo.read_text(), loss.read_text()
    with pytest.raises(RuntimeError, match="anchor changed"):
        patcher.patch(root)
    assert (ppo.read_text(), loss.read_text()) == before


def _load(torch, ppo_source, loss_source, entropy_saved):
    # Exercise the pinned entropy autograd implementation on CPU/one TP rank.
    # Only distributed collectives, compilation, and Megatron CE are substituted.
    namespace = {
        "torch": torch,
        "Namespace": SimpleNamespace,
        "dist": SimpleNamespace(
            ProcessGroup=object,
            ReduceOp=SimpleNamespace(MAX=None),
            all_reduce=lambda *args, **kwargs: None,
        ),
        "mpu": SimpleNamespace(get_tensor_model_parallel_group=lambda: None),
    }
    exec(compile(ppo_source, "ppo_utils.py", "exec"), namespace)
    entropy_fn = namespace["compute_entropy_from_logits"]

    def entropy(logits, group):
        def save(tensor):
            entropy_saved.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(save, lambda x: x):
            return entropy_fn(logits, group)

    namespace["compute_entropy_from_logits"] = entropy
    namespace["compute_log_probs"] = lambda logits, tokens, group, keep_mask=None: (
        logits.log_softmax(-1).gather(-1, tokens[:, None])
    )
    namespace["_build_shifted_tokens"] = lambda T, device, tokens, *args: tokens[0]
    namespace["_extract_per_sample"] = lambda logp, ent, *args: ([logp], [ent])
    exec(compile(loss_source, "loss.py", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("chunk_size", [-1, 1, 4])
@pytest.mark.parametrize("entropy_coef", [0.0, 0.01])
def test_loss_gradients_and_entropy_saved_tensors(
    sources, monkeypatch, chunk_size, entropy_coef
):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch, "compile", lambda **kwargs: lambda fn: fn)
    root, ppo, loss = sources
    original = ppo.read_text(), loss.read_text()
    patcher.patch(root)
    patched = ppo.read_text(), loss.read_text()
    torch.manual_seed(7)
    initial = torch.randn(1, 11, 17)
    tokens = torch.randint(17, (11,))
    outcomes = []
    for code in (original, patched):
        saved = []
        ns = _load(torch, *code, saved)
        logits = initial.clone().requires_grad_()
        _, result = ns["get_log_probs_and_entropy"](
            logits,
            args=SimpleNamespace(
                entropy_coef=entropy_coef,
                log_probs_chunk_size=chunk_size,
                rollout_temperature=0.7,
                allgather_cp=False,
            ),
            unconcat_tokens=[tokens],
            total_lengths=[11],
            response_lengths=[11],
            with_entropy=True,
        )
        logp, entropy = result["log_probs"][0], result["entropy"][0]
        objective = -logp.mean() - entropy_coef * entropy.mean()
        objective.backward()
        torch.testing.assert_close(logits.detach(), initial)
        outcomes.append(
            (logp.detach(), entropy.detach(), logits.grad, saved, entropy.requires_grad)
        )
    baseline, fixed = outcomes
    for expected, actual in zip(baseline[:3], fixed[:3], strict=True):
        torch.testing.assert_close(actual, expected)
    assert baseline[
        3
    ]  # The original implementation saves entropy tensors even at zero.
    if entropy_coef == 0:
        assert fixed[3] == []
        assert not fixed[4]
    else:
        assert fixed[3] == baseline[3]
        assert fixed[4]


def test_outer_no_grad_and_empty_input(sources, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch, "compile", lambda **kwargs: lambda fn: fn)
    root, ppo, loss = sources
    patcher.patch(root)
    saved = []
    ns = _load(torch, ppo.read_text(), loss.read_text(), saved)
    for length in (0, 11):
        logits = torch.randn(length, 17, requires_grad=True)
        before = logits.detach().clone()
        with torch.no_grad():
            logp, entropy = ns["calculate_log_probs_and_entropy"](
                logits,
                torch.zeros(length, dtype=torch.long),
                None,
                with_entropy=True,
                chunk_size=4,
            )
        assert not logp.requires_grad and not entropy.requires_grad
        torch.testing.assert_close(logits.detach(), before)
    assert not saved
