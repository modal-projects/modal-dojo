"""Avoid saved entropy tensors when entropy is only a training metric.

Applied explicitly by the coding-agent tutorial to its pinned Slime fork.
Validate every source anchor before writing either file so upstream drift fails
the image build instead of silently leaving the memory issue unfixed.
"""

from __future__ import annotations

import base64
from pathlib import Path


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(new) == 1 and old not in source.replace(new, "", 1):
        return source
    if source.count(old) == 1:
        return source.replace(old, new, 1)
    raise RuntimeError("Entropy patch anchor changed; review the pinned Slime source")


def patch(slime_root: Path) -> None:
    ppo_path = slime_root / "slime/utils/ppo_utils.py"
    loss_path = slime_root / "slime/backends/megatron_utils/loss.py"
    ppo = ppo_path.read_text()
    loss = loss_path.read_text()

    ppo = _replace_once(
        ppo,
        "    logits, tokens, tp_group, with_entropy: bool = False, "
        "chunk_size: int = -1, log_prob_keep_mask=None\n",
        "    logits, tokens, tp_group, with_entropy: bool = False, "
        "chunk_size: int = -1, log_prob_keep_mask=None,\n"
        "    *, entropy_requires_grad: bool = True,\n",
    )
    ppo = _replace_once(
        ppo,
        "    entropy = None\n    if logits.size(0) != 0:\n",
        "    entropy = None\n\n"
        "    def entropy_for_logits(values):\n"
        "        if entropy_requires_grad and torch.is_grad_enabled():\n"
        "            return compute_entropy_from_logits(values.clone(), tp_group)\n"
        "        # Entropy forward does not mutate its input; only backward does.\n"
        "        with torch.no_grad():\n"
        "            return compute_entropy_from_logits(values.detach(), tp_group)\n\n"
        "    if logits.size(0) != 0:\n",
    )
    ppo = _replace_once(
        ppo,
        "                    entropy_input = logits_chunk.clone()\n"
        "                    entropys.append(compute_entropy_from_logits(entropy_input, tp_group))\n",
        "                    entropys.append(entropy_for_logits(logits_chunk))\n",
    )
    ppo = _replace_once(
        ppo,
        "                entropy_input = logits.clone()\n"
        "                entropy = compute_entropy_from_logits(entropy_input, tp_group)\n",
        "                entropy = entropy_for_logits(logits)\n",
    )
    loss = _replace_once(
        loss,
        "        log_prob_keep_mask=top_p_keep_mask,\n",
        "        log_prob_keep_mask=top_p_keep_mask,\n"
        "        entropy_requires_grad=args.entropy_coef != 0,\n",
    )
    compile(ppo, str(ppo_path), "exec")
    compile(loss, str(loss_path), "exec")
    ppo_path.write_text(ppo)
    loss_path.write_text(loss)


def image_patch_command() -> str:
    """Embed the standalone patch in the tutorial's image build."""
    encoded = base64.b64encode(Path(__file__).read_bytes()).decode()
    return f"echo {encoded} | base64 -d | python3"


if __name__ == "__main__":
    patch(Path("/root/slime"))
