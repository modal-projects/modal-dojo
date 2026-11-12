"""A tinker-cookbook RL config as a TinkerProgram, reporting to Dojo."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.status import SpindleStatus
from modal_dojo.common.training_rollout import TrainingRolloutSample

from .reporting import report_rollouts, report_status, report_timing

logger = logging.getLogger(__name__)
_training_rollout = ContextVar("spindle_training_rollout", default=False)

# The rl_train functions whose calls are the run's phases.
_PHASE_FUNCTIONS = {
    SpindleStatus.DO_GROUP_ROLLOUT_AND_FILTER_CONSTANT_REWARD: (
        "do_group_rollout_and_filter_constant_reward"
    ),
    SpindleStatus.TRAIN_STEP: "train_step",
    SpindleStatus.SAVE_CHECKPOINT_AND_GET_SAMPLING_CLIENT: (
        "save_checkpoint_and_get_sampling_client"
    ),
}


def cookbook_rl(config) -> CookbookRL:
    """Wrap a tinker-cookbook ``rl_train.Config`` for ``SpindleRecipe`` to run."""
    from tinker_cookbook.rl.train import Config

    if not isinstance(config, Config):
        raise DojoConfigError("cookbook_rl takes a tinker-cookbook rl_train.Config")
    return CookbookRL(config)


class CookbookRL:
    """tinker-cookbook's ``rl_train.main(config)`` as a TinkerProgram."""

    def __init__(self, config):
        if not callable(config.dataset_builder):
            raise DojoConfigError(
                "dataset_builder must be a tinker-cookbook builder; build one from a "
                "Dojo dataset with modal_dojo.tinker.rl_dataset_builder"
            )
        if (
            config.async_config is not None
            or config.stream_minibatch_config is not None
        ):
            raise DojoConfigError(
                "SpindleRecipe v0 supports synchronous tinker-cookbook RL only"
            )
        if config.max_tokens <= 0:
            raise DojoConfigError("max_tokens must be positive")
        if config.max_steps is not None and config.max_steps <= 0:
            raise DojoConfigError("max_steps must be positive")
        self.config = config
        self.model_name = config.model_name

    async def run(self, client, ctx):
        import chz
        from tinker_cookbook.rl import train as rl_train
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        # rl_train.main opens its own ServiceClient, which reaches Spindle the
        # same way this one does.
        del client
        tokenizer = await asyncio.to_thread(get_tokenizer, self.model_name)
        _Reporter(ctx, tokenizer).install(rl_train)
        try:
            # Retries resume from checkpoints.jsonl in this directory.
            await rl_train.main(chz.replace(self.config, log_path=ctx.log_dir))
        finally:
            _record_checkpoints(ctx)


def _record_checkpoints(ctx):
    checkpoints = Path(ctx.log_dir) / "checkpoints.jsonl"
    if not checkpoints.exists():
        return
    try:
        ctx.run.metadata["spindle_checkpoints"] = [
            json.loads(line)
            for line in checkpoints.read_text().splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as error:
        ctx.run.metadata["checkpoint_read_error"] = str(error)
        logger.warning("Could not read tinker-cookbook checkpoint records: %s", error)


def _decode_turns(tokenizer, trajectory) -> list[dict[str, str]]:
    """Each turn's new prompt tokens and its response.

    A turn's prompt usually repeats the whole conversation so far; decoding
    only what follows it keeps long multi-turn episodes linear in size.
    """
    turns = []
    context: list[int] = []
    for transition in trajectory.transitions:
        prompt = transition.ob.to_ints()
        shared = 0
        for seen, token in zip(context, prompt):
            if seen != token:
                break
            shared += 1
        turns.append(
            {
                "prompt": tokenizer.decode(prompt[shared:]),
                "response": tokenizer.decode(transition.ac.tokens),
            }
        )
        context = prompt + list(transition.ac.tokens)
    return turns


class _Reporter:
    """Phases, metrics, timing and training rollouts from tinker-cookbook RL."""

    def __init__(self, ctx, tokenizer):
        self.ctx = ctx
        self.run = ctx.run
        self.tokenizer = tokenizer
        self.groups = []

    def install(self, rl_train):
        # The driver owns its single-use container, so patch modules in place.
        from tinker_cookbook.rl import rollouts

        reporter = self
        ml_log, trace = rl_train.ml_log, rl_train.trace
        setup_logging = ml_log.setup_logging
        trace_iteration = trace.trace_iteration
        do_group_rollout = rollouts.do_group_rollout

        class Logger(ml_log.Logger):
            def log_hparams(self, config):
                pass

            def log_metrics(self, metrics, step=None):
                reporter.log_metrics(metrics, step)

        @wraps(setup_logging)
        def setup_logging_with_dojo(*args, **kwargs):
            result = setup_logging(*args, **kwargs)
            result.loggers.append(Logger())
            return result

        @wraps(trace_iteration)
        @contextmanager
        def trace_iteration_with_dojo(step):
            with trace_iteration(step) as window:
                try:
                    yield window
                finally:
                    reporter.report_timing(step, window)

        @wraps(do_group_rollout)
        async def capture_group(env_group_builder, *args, **kwargs):
            group = await do_group_rollout(env_group_builder, *args, **kwargs)
            if _training_rollout.get():
                reporter.groups.append((env_group_builder.logging_tags(), group))
            return group

        def report_phase(phase, operation):
            @wraps(operation)
            async def wrapped(*args, **kwargs):
                if reporter.run.framework_status != phase:
                    reporter.phase(phase, is_active=True)
                token = _training_rollout.set(
                    phase is SpindleStatus.DO_GROUP_ROLLOUT_AND_FILTER_CONSTANT_REWARD
                )
                try:
                    return await operation(*args, **kwargs)
                finally:
                    _training_rollout.reset(token)

            return wrapped

        ml_log.setup_logging = setup_logging_with_dojo
        trace.trace_iteration = trace_iteration_with_dojo
        rollouts.do_group_rollout = capture_group
        for phase, name in _PHASE_FUNCTIONS.items():
            setattr(rl_train, name, report_phase(phase, getattr(rl_train, name)))

    def phase(self, phase: SpindleStatus, **kwargs):
        report_status(self.run, phase, **kwargs)

    def log_metrics(self, metrics, step=None):
        # Only training metrics mark a batch; the final weights' evaluation is
        # logged after the last batch with just its step.
        if "progress/batch" not in metrics:
            if step is not None:
                self.ctx.log_metrics(step, **metrics)
            return
        step = int(metrics["progress/batch"])
        self.run.metrics = {}
        self.ctx.log_metrics(step, **metrics)
        try:
            self.flush_groups(step, self.run.metrics)
        except Exception:
            # tinker-cookbook calls loggers unguarded; reporting must not stop training.
            logger.exception("Could not report tinker-cookbook rollouts")
            self.groups.clear()
        self.phase(
            self.run.framework_status,
            is_active=True,
            progress_current=step + 1,
            # tinker-cookbook stops at max_steps or when the dataset runs out.
            progress_total=round((step + 1) / metrics["progress/done_frac"]),
            progress_unit="step",
            rollout_id=step,
        )

    def report_timing(self, step, window):
        report_timing(self.run, step, window.get_span_dicts())

    def flush_groups(self, step, metrics):
        samples = []
        for group_index, (tags, group) in enumerate(self.groups):
            metrics_G = group.metrics_G or [{}] * len(group.trajectories_G)
            for trajectory, final_reward, sample_metrics in zip(
                group.trajectories_G, group.final_rewards_G, metrics_G, strict=True
            ):
                turns = _decode_turns(self.tokenizer, trajectory)
                rewards = [transition.reward for transition in trajectory.transitions]
                samples.append(
                    TrainingRolloutSample(
                        rollout_index=len(samples),
                        sample_index=len(samples),
                        group_index=group_index,
                        prompt=turns[0]["prompt"] if turns else "",
                        # Later turns' prompts hold only their new messages, so
                        # this is the episode's transcript after the first prompt.
                        response=turns[0]["response"]
                        + "".join(
                            turn["prompt"] + turn["response"] for turn in turns[1:]
                        )
                        if turns
                        else "",
                        score=float(final_reward + sum(rewards)),
                        metadata={
                            "transition_rewards": rewards,
                            "final_reward": final_reward,
                            "tags": tags,
                            "metrics": sample_metrics,
                        },
                    )
                )
        self.groups.clear()
        report_rollouts(self.run, step, samples, metrics)
