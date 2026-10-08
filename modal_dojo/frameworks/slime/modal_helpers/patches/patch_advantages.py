"""Patch compute_advantages_and_returns to handle missing tensor references.

When a custom generate function is used with kl_coef=0 and no critic,
the ``can_reuse_log_probs_in_loss`` optimisation in ``train_actor``
skips computing log_probs before advantage calculation.  Standard
rollouts still provide ``rollout_log_probs`` as a fallback, but custom
generate functions do not collect them.  This leaves all three tensor
references (log_probs, rollout_log_probs, values) as None, crashing the
``[torch.zeros_like(x, ...) for x in xs]`` list comprehension with
``TypeError: 'NoneType' object is not iterable``.

This patch makes the None-xs path fall back to creating zero-KL tensors sized
to the *context-parallel-local* response chunk (via ``slice_log_prob_with_cp``).
Sizing them to the full ``response_lengths`` instead would make GRPO advantages
full-length while the policy ratio is CP-sharded, raising a shape mismatch in
``compute_policy_loss`` under ``context_parallel_size > 1``. At cp_size=1 the
slicer is a no-op, so this is behaviour-neutral there.

It also makes the on-policy-distillation (OPD) KL term in
``apply_opd_kl_to_advantages`` NaN-safe: a per-sample ``teacher_log_probs``
tensor containing any NaN is treated as "no teacher signal for this
trajectory", so its ``reverse_kl`` is set to zero (equivalent to a
``student - student`` self-reference) instead of subtracting NaN/garbage.
This lets a rollout deliberately disable OPD for individual trajectories
(e.g. when a remote teacher / reward server is unavailable) by writing a
NaN ``teacher_log_probs`` tensor, while keeping that trajectory's policy
(GRPO) gradient intact. It is behaviour-neutral for normal runs because
real teacher logprobs never contain NaN.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

p = pathlib.Path("/root/slime/slime/backends/megatron_utils/loss.py")
src = p.read_text()

old = """\
        xs = log_probs or rollout_log_probs or values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]"""

new = """\
        xs = log_probs or rollout_log_probs or values
        if xs is not None:
            kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
        else:
            _dev = loss_masks[0].device if loss_masks else torch.cuda.current_device()
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp as _tg_cp_slice
            kl = [
                _tg_cp_slice(torch.zeros(rl, dtype=torch.float32, device=_dev), tl, rl)
                for rl, tl in zip(response_lengths, total_lengths)
            ]"""

if old in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
    print("[patch_advantages] patched None-xs KL fallback")
else:
    print("[patch_advantages] WARNING: None-xs KL fallback target not found; skipped")

opd_old = """\
        reverse_kl = student_log_probs[i] - teacher_log_probs[i]"""

opd_new = """\
        if torch.isnan(teacher_log_probs[i]).any():
            reverse_kl = torch.zeros_like(student_log_probs[i])
        else:
            reverse_kl = student_log_probs[i] - teacher_log_probs[i]"""

if opd_old in src:
    p.write_text(src.replace(opd_old, opd_new, 1))
    print("[patch_advantages] patched OPD reverse_kl to be NaN-safe")
else:
    print("[patch_advantages] WARNING: OPD reverse_kl target not found; skipped")
