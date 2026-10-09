"""Patch ``_compute_zero_std_metrics`` to ignore aborted (reward is None) samples.

slime's rollout metric logging path
(``RolloutManager.generate`` -> ``_log_rollout_data`` ->
``compute_metrics_from_samples`` -> ``_compute_zero_std_metrics``) runs on the
kept training batch *before* the reward post-process converts samples to train
data.  Multi-turn / agentic rollouts can yield ``ABORTED`` samples whose
``reward`` is still ``None``: in ``generate_and_rm`` a fan-out group where *any*
member is aborted returns early with no reward computed for the whole group.

Cross-tokenizer OPD goes further: its ``custom_rm_function`` returns the
teacher's raw ``/generate`` response (a ``dict`` of logprobs) as ``sample.reward``
during the rollout, and only the later ``post_process`` step reduces that to a
scalar.  The zero-std metric runs *in between*, so ``get_reward_value`` hands
back either a ``dict`` (normal sample) or ``None`` (aborted sample), and
``round(...)`` crashes with::

    TypeError: type NoneType doesn't define __round__
    TypeError: type dict doesn't define __round__

This patch makes the metric numeric-safe: samples/groups whose reward is not an
``int``/``float`` are skipped.  It is behaviour-neutral for plain scalar-reward
runs because those rewards are always numeric.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

p = pathlib.Path("/root/slime/slime/ray/rollout.py")
src = p.read_text()

old_is_zero_std = """\
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)"""

new_is_zero_std = """\
        rewards = [
            r
            for sample in samples
            if isinstance((r := sample.get_reward_value(args)), (int, float))
        ]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)"""

if old_is_zero_std in src:
    src = src.replace(old_is_zero_std, new_is_zero_std, 1)
    print("[patch_zero_std_metrics] patched _is_zero_std to ignore non-numeric rewards")
else:
    print("[patch_zero_std_metrics] WARNING: _is_zero_std target not found; skipped")

old_interesting = (
    "    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) "
    "for g in interesting_sample_groups]"
)

new_interesting = """\
    interesting_rewards = [
        str(round(r, 1))
        for g in interesting_sample_groups
        if isinstance((r := g[0].get_reward_value(args)), (int, float))
    ]"""

if old_interesting in src:
    src = src.replace(old_interesting, new_interesting, 1)
    print(
        "[patch_zero_std_metrics] patched interesting_rewards to ignore non-numeric rewards"
    )
else:
    print(
        "[patch_zero_std_metrics] WARNING: interesting_rewards target not found; skipped"
    )

p.write_text(src)
