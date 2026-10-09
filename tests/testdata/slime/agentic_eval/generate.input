"""Per-sample agentic-RL rollout hook for slime.

Wire-up: ``--custom-generate-function-path agentic_rl.core.generate.generate``.

One call = one episode: pick the env by ``metadata.task_type``, run the
in-process mini-swe loop (the token-recording ``RecordingModel`` + a bash-only
Modal ``Sandbox``) in a worker thread, and build token-faithful training
Sample(s). Each turn carries the ``weight_version`` live at generation, so
fully-async off-policy correction (TIS/clipping) and the async-health metrics
work. A weight-update abort mid-generation -> ``Status.ABORTED`` -> slime
recycles the episode. Reward is computed inline so slime's reward-model step is
skipped. Train and eval take the same path.

Episode limits come from ``args`` (``--custom-config-path``): ``agentic_max_steps``,
``agentic_episode_timeout``, ``agentic_exec_timeout``, ``agentic_grade_timeout``,
``agentic_eval_timeout``, ``agentic_query_timeout``, ``agentic_max_empty_turns``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import dataclasses
import logging
import threading
import time
import uuid
from typing import Any

from agentic_rl.envs.base import EnvMetadataError, EpisodeLimits, RewardResult, load_env

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from .model import RecordingModel

logger = logging.getLogger("agentic_rl")

# Episodes are long I/O-bound chains; asyncio.to_thread caps at min(32, cpu+4)
# threads, which throttles in-flight episodes regardless of sglang concurrency.
# Use a wide dedicated pool sized to the rollout's serving concurrency.
_episode_pool: concurrent.futures.ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _episode_executor(args) -> concurrent.futures.ThreadPoolExecutor:
    global _episode_pool
    if _episode_pool is None:
        with _pool_lock:
            if _episode_pool is None:
                from slime.utils.http_utils import get_rollout_num_engines

                n = getattr(args, "sglang_server_concurrency", 128) * get_rollout_num_engines(args)
                n = min(max(n, 32), 2048)
                _episode_pool = concurrent.futures.ThreadPoolExecutor(max_workers=n, thread_name_prefix="episode")
                logger.info("[agentic_rl] episode thread pool: max_workers=%d", n)
    return _episode_pool


def _limits(args) -> EpisodeLimits:
    return EpisodeLimits(
        max_steps=getattr(args, "agentic_max_steps", 30),
        episode_timeout=getattr(args, "agentic_episode_timeout", 1800),
        exec_timeout=getattr(args, "agentic_exec_timeout", 120),
        grade_timeout=getattr(args, "agentic_grade_timeout", 1800),
        eval_timeout=getattr(args, "agentic_eval_timeout", None),
    )


def _sampling_params(state: GenerateState, overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Base on the rollout's sampling defaults; let slime's per-call params (eval
    temperature/top_p, max_new_tokens) override."""
    params = dict(state.sampling_params)
    for k in ("temperature", "top_p", "top_k", "max_new_tokens", "stop", "stop_token_ids"):
        if overrides and overrides.get(k) is not None:
            params[k] = overrides[k]
    return params


def _session_id(sample: Sample, md: dict[str, Any]) -> str:
    if sample.index is not None and sample.group_index is not None:
        return f"agent-{md['instance_id']}-{sample.index}-{sample.group_index}"
    return f"agent-{md['instance_id']}-{uuid.uuid4().hex[:8]}"


def _run_episode(env, md: dict[str, Any], model: RecordingModel, limits: EpisodeLimits) -> RewardResult:
    try:
        return env.rollout(md, model=model, limits=limits)
    except Exception as e:
        logger.exception("[agentic_rl] episode failed (instance=%s)", md.get("instance_id"))
        return RewardResult(
            reward=0.0,
            is_solved=False,
            extra={"error": "episode_exception", "grading_error": str(e)},
            grading_status="timeout" if isinstance(e, TimeoutError) else "infrastructure_error",
        )


async def generate(args, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    state = GenerateState(args)
    if state.aborted:
        sample.status = Sample.Status.ABORTED
        return sample

    try:
        env = load_env((sample.metadata or {}).get("task_type"))
        md = env.normalize_metadata(sample)
    except EnvMetadataError as e:
        return _abort(sample, str(e))
    except (ValueError, TypeError) as e:
        return _abort(sample, f"env_dispatch:{type(e).__name__}:{e}")

    limits = _limits(args)
    sample.metadata = {**(sample.metadata or {}), "budgets": dataclasses.asdict(limits)}
    session_id = sample.session_id or _session_id(sample, md)
    sample.session_id = session_id

    model = RecordingModel(
        state.tokenizer,
        _sampling_params(state, sampling_params),
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
        session_id,
        tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
        reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
        max_context_len=int(getattr(args, "rollout_max_context_len", 0) or 0),
        query_timeout=getattr(args, "agentic_query_timeout", 600),
        max_empty_turns=getattr(args, "agentic_max_empty_turns", 3),
        # Forced think closure on length-capped no-tool-call turns (A/B knob; see
        # RecordingModel.query). Off by default: flips via custom_config_path.
        close_think_on_length=bool(getattr(args, "agentic_close_think_on_length", False)),
        max_think_closures=int(getattr(args, "agentic_max_think_closures", 2) or 0),
        think_closure_budget=int(getattr(args, "agentic_think_closure_budget", 4096)),
    )

    t0 = time.time()
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(_episode_executor(args), _run_episode, env, md, model, limits)
    except asyncio.CancelledError:
        if asyncio.current_task().cancelling():
            raise
        return _abort(sample, "exception:CancelledError")
    except Exception as e:  # noqa: BLE001
        logger.error("[agentic_rl] %s: rollout failed: %s", md["instance_id"], e)
        return _abort(sample, f"exception:{type(e).__name__}")

    if model.aborted and not evaluation:  # weight update aborted a turn mid-flight -> recycle (train only)
        sample.status = Sample.Status.ABORTED
        return sample

    return _build_samples(
        sample, model, result, state.tokenizer, md, args, elapsed=time.time() - t0, evaluation=evaluation
    )


def _build_samples(sample, model, result, tokenizer, md, args, *, elapsed: float, evaluation: bool = False):
    key = sample.label or md.get("instance_id") or ""
    usable = [c for c in model.chains if c.has_response]
    if not usable:
        # No surviving generation -- two flavors, told apart by the model's terminal
        # reason: it ran but every turn was a no-tool-call rolled back out of the stream
        # (ContextLengthExceeded / NoProgress), or it never reached an LLM call at all
        # (empty chains => boot/image/provision/sandbox-death failure). Carry the real
        # reason plus the rolled-back generation/prompt so the dashboard shows *why*
        # there's no trajectory instead of a blanket "ImageUnusable".
        last = model.chains[-1] if model.chains else None
        reason = model.exit_status or "ImageUnusable"
        tail = last.truncated_tail if last else None
        full_prompt = last.full_prompt if last else None
        # Ship a well-formed, fully-masked reward-0 sample (remove_sample=True), never a
        # bare ABORTED. Neither eval nor the train loop (generate_rollout_async) recycles
        # ABORTED -- the loop appends every finished group straight to the training batch
        # -- so a bare ABORTED leaks reward=None into _post_process_rewards and crashes the
        # step (observed: rollout_0 dump, a sandbox that died before the first LLM call ->
        # empty chains -> 2/256 samples reward=None). _ship_null also keeps the group at
        # n_samples_per_prompt, which slime's GRPO reshape requires. Train == eval here.
        if not evaluation:
            logger.warning("[agentic_rl] %s unusable (%s); shipping masked reward-0", key, reason)
        null = _ship_null(sample, tokenizer, md, reason, tail=tail, full_prompt=full_prompt, elapsed=elapsed)
        if result is not None:
            null.metadata["agentic"].update(result.extra)
            null.metadata["agentic"]["grading_status"] = result.grading_status
        if getattr(model, "n_think_closures", 0):  # closure attempted but the episode still nulled
            null.metadata["agentic"]["think_closures"] = model.n_think_closures
        return null

    k = len(usable)
    stats = {
        "turns": sum(len(c.versions) for c in usable),
        "format_errors": model.n_format_errors,
        "think_closures": getattr(model, "n_think_closures", 0),
        "output_tokens": sum(sum(c.loss_mask) for c in usable),
        "response_tokens": sum(len(c.tokens) - c.prompt_len for c in usable),
        "chains": k,
        "gen_timestamp": time.time(),
        # Per-episode wall time, measured here so it survives even when the env's own
        # timing is lost (an episode that raised ships extra={"error":...}); the basis
        # for the tail/straggler metrics in metrics.py.
        "elapsed_sec": round(elapsed, 1),
        "is_solved": result.is_solved,
        **{key_: val for key_, val in result.extra.items() if key_ != "harbor_step_results"},
        "grading_status": result.grading_status,
    }
    # A context-length/no-progress terminal turn is rolled back, but earlier
    # valid tool-call turns remain trainable. Preserve the terminal reason on
    # those non-null prefixes too; otherwise ContextLengthExceeded monitoring
    # only sees first-turn nulls and incorrectly implies every occurrence was
    # fully masked.
    if model.exit_status and not stats.get("exit_status"):
        stats["exit_status"] = model.exit_status
    # LLM-inference seconds are the generation portion of the env's "agent" leg, not a
    # disjoint phase -- fold them into the timing dict as "generate" so they live
    # alongside boot/prep/agent/verifier (metrics.py logs agentic/timing/generate, and
    # the dashboard splits the agent bar into generate + agent/tools off this key).
    stats["timing"] = {**(stats.get("timing") or {}), "generate": round(model.gen_time, 1)}
    logger.info(
        "[agentic_rl] %s: reward=%.2f solved=%s turns=%d chains=%d elapsed=%.1fs",
        md["instance_id"],
        result.reward,
        result.is_solved,
        stats["turns"],
        k,
        elapsed,
    )

    samples = []
    for i, c in enumerate(usable):
        s = sample if i == 0 else copy.copy(sample)
        s.tokens = c.tokens
        s.response_length = len(c.tokens) - c.prompt_len
        s.loss_mask = c.loss_mask[c.prompt_len :]
        s.remove_sample = sample.remove_sample or result.grading_status != "valid"
        if s.remove_sample:
            s.loss_mask = [0] * s.response_length
        s.rollout_log_probs = c.logprobs[c.prompt_len :]
        s.weight_versions = [v for v in c.versions if v is not None]
        s.reward = 0.0 if s.remove_sample else result.reward / k
        s.response = tokenizer.decode(c.tokens[c.prompt_len :], skip_special_tokens=False)
        s.status = Sample.Status.COMPLETED
        s.rollout_id = sample.index  # siblings share rollout_id so reducers don't over-count
        # Per-chain debug surfaces for the dashboard: the exact rendered prompt, and a
        # rolled-back terminal generation (length-truncated think, etc.) if any.
        agentic = {**stats, "full_prompt": c.full_prompt}
        # Per-turn token spans (response-relative) + generation-end timestamps:
        # the join keys turn-level reward shaping needs to align mid-episode judge
        # submissions (server ts) with the turn whose tool call produced them.
        if c.turn_spans:
            agentic["turn_spans"] = [[s - c.prompt_len, e - c.prompt_len] for s, e in c.turn_spans]
            agentic["turn_ts"] = list(c.turn_ts)
        if c.truncated_tail is not None:
            agentic["truncated_tail"] = c.truncated_tail
        s.metadata = {**(sample.metadata or {}), "instance_id": md["instance_id"], "agentic": agentic}
        samples.append(s)

    # prefix-cache info is per episode; attribute it to the primary sample.
    samples[0].prefix_cache_info.cached_tokens = model.cached_tokens
    samples[0].prefix_cache_info.total_prompt_tokens = model.input_tokens
    return samples[0] if k == 1 else samples


def _ship_null(
    sample: Sample,
    tokenizer,
    md: dict[str, Any],
    reason: str,
    *,
    tail: dict | None = None,
    full_prompt: str | None = None,
    elapsed: float | None = None,
) -> Sample:
    """A valid but fully-masked reward-0 sample for a permanently-failing task:
    ships to leave the buffer, contributes no gradient (remove_sample=True).

    ``reason`` is the real terminal cause (the model's ``exit_status``, or an
    ``ImageUnusable`` fallback when no LLM call ran). ``tail``/``full_prompt`` carry
    the rolled-back terminal generation and the exact prompt so the dashboard can
    show what the episode did before it was discarded -- convert.py renders the tail
    as a synthetic turn, so a turn-1 runaway ``<think>`` is visible instead of blank."""
    ptoks = tokenizer.encode(md.get("problem_statement", "") or reason, add_special_tokens=False)[:512]
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else (ptoks[-1] if ptoks else 0)
    sample.tokens = ptoks + [eos]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.weight_versions = []
    sample.reward = 0.0
    sample.status = Sample.Status.COMPLETED
    sample.remove_sample = True
    sample.rollout_id = sample.index
    agentic: dict[str, Any] = {"exit_status": reason, "turns": 0, "gen_timestamp": time.time()}
    if elapsed is not None:
        agentic["elapsed_sec"] = round(elapsed, 1)
    if tail is not None:
        agentic["truncated_tail"] = tail
    if full_prompt is not None:
        agentic["full_prompt"] = full_prompt
    sample.metadata = {**(sample.metadata or {}), "agentic": agentic}
    return sample


def _abort(sample: Sample, reason: str) -> Sample:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.rollout_id = sample.index
    sample.status = Sample.Status.ABORTED
    sample.remove_sample = True
    sample.metadata = {
        **(sample.metadata or {}),
        "abort_reason": reason,
        "agentic": {"grading_status": "infrastructure_error"},
    }
    logger.warning("[agentic_rl] aborted: %s", reason)
    return sample
