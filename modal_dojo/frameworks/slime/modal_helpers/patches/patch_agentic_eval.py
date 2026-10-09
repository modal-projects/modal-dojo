"""Recover gradeable eval artifacts after a model transport failure.

Explicit tutorial patch for the pinned agentic Slime fork. All anchors are
validated and compiled before writing; recovery remains opt-in and eval-only.
"""

from __future__ import annotations

import base64
from pathlib import Path


ERROR_SOURCE = '''"""Typed generation transport failure, separate from verifier failure."""
import urllib.error


class GenerationRequestError(RuntimeError):
    phase = "generation"

    def __init__(self, error, diagnostics):
        super().__init__(str(error))
        self.diagnostics = diagnostics
        self.is_timeout = isinstance(error, TimeoutError) or (
            isinstance(error, urllib.error.URLError)
            and isinstance(error.reason, TimeoutError)
        )
        self.recoverable = self.is_timeout or isinstance(error, ConnectionError)
        if isinstance(error, urllib.error.HTTPError):
            self.recoverable = error.code in (502, 503, 504)


def evaluation_diagnostics(samples):
    records = [(s.metadata or {}).get("agentic", {}) for s in samples]
    if not any(records):
        return {}
    graded = [r for r in records if r.get("grading_status") == "valid"]
    metrics = {
        "graded_count": len(graded),
        "ungraded_count": len(records) - len(graded),
        "generation_error_count": sum(bool(r.get("generation_request_errors"))
            or r.get("failure_phase") == "generation" for r in records),
        "recovered_grade_count": sum(bool(r.get("graded_after_generation_error"))
            and r.get("grading_status") == "valid" for r in records),
    }
    if graded:
        metrics["solve_rate_on_graded"] = sum(bool(r.get("is_solved")) for r in graded) / len(graded)
    return metrics
'''


def _replace(source: str, old: str, new: str) -> str:
    if source.count(new) == 1 and old not in source.replace(new, "", 1):
        return source
    if source.count(old) != 1:
        raise RuntimeError(f"Agentic eval patch anchor changed: {old[:90]!r}")
    return source.replace(old, new, 1)


def patch(root: Path) -> None:
    paths = {
        "model": root / "agentic_rl/core/model.py",
        "generate": root / "agentic_rl/core/generate.py",
        "env": root / "agentic_rl/envs/harbor/env.py",
        "rollout": root / "slime/ray/rollout.py",
    }
    sources = {name: path.read_text() for name, path in paths.items()}
    model = sources["model"]
    model = _replace(
        model,
        "from .prompts import BASH_TOOL, FORMAT_ERROR_TEMPLATE, OBSERVATION_TEMPLATE\n",
        "from .prompts import BASH_TOOL, FORMAT_ERROR_TEMPLATE, OBSERVATION_TEMPLATE\n"
        "from .request_errors import GenerationRequestError\n",
    )
    model = _replace(
        model,
        "        self.gen_time = 0.0\n",
        "        self.gen_time = 0.0\n        self.request_errors = []\n",
    )
    model = _replace(
        model,
        "        with urllib.request.urlopen(req, timeout=self.query_timeout) as resp:\n"
        "            data = json.loads(resp.read())\n"
        "        self.gen_time += time.perf_counter() - t0\n",
        """        started_at = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.query_timeout) as resp:
                data = json.loads(resp.read())
        except Exception as error:
            diagnostics = {
                "phase": "generation", "error_type": type(error).__name__,
                "message": str(error)[:500],
                "seconds": round(time.perf_counter() - t0, 3),
                "started_at": started_at, "ended_at": time.time(),
                "routing_key": self.headers.get("X-SMG-Routing-Key"),
                "input_tokens": len(input_ids),
                "max_new_tokens": sp.get("max_new_tokens"),
                "timeout_seconds": self.query_timeout,
            }
            self.request_errors.append(diagnostics)
            raise GenerationRequestError(error, diagnostics) from error
        finally:
            self.gen_time += time.perf_counter() - t0
""",
    )

    env = sources["env"]
    env = _replace(
        env,
        "from agentic_rl.core.timing import PhaseTimer\n",
        "from agentic_rl.core.timing import PhaseTimer\n"
        "from agentic_rl.core.request_errors import GenerationRequestError\n",
    )
    env = _replace(
        env,
        "        grading_error = None\n",
        "        grading_error = None\n        agent_error = None\n",
    )
    env = _replace(
        env,
        '                with timer.phase("agent"):\n'
        '                    run_leg(sb, step["instruction"], remaining)\n',
        """                with timer.phase("agent"):
                    try:
                        run_leg(sb, step["instruction"], remaining)
                    except GenerationRequestError as error:
                        if not md.get("grade_after_generation_error") or not error.recoverable:
                            raise
                        agent_error = error.diagnostics
                        logger.exception("[harbor] %s: generation failed; grading existing sandbox", md["instance_id"])
""",
    )
    env = _replace(
        env,
        '                if not _meets_min_reward(rewards, step.get("min_reward")):\n',
        "                if agent_error is not None:\n"
        "                    break  # Never continue a multi-step task after recovery.\n"
        '                if not _meets_min_reward(rewards, step.get("min_reward")):\n',
    )
    env = _replace(
        env,
        '                "grading_error": grading_error,\n',
        '                "grading_error": grading_error,\n'
        '                "agent_error": agent_error,\n'
        '                "graded_after_generation_error": agent_error is not None,\n'
        '                "failure_phase": "verifier" if grading_error else ("generation" if agent_error else None),\n',
    )

    generate = sources["generate"]
    generate = _replace(
        generate,
        '            extra={"error": "episode_exception", "grading_error": str(e)},\n'
        '            grading_status="timeout" if isinstance(e, TimeoutError) else "infrastructure_error",\n',
        """            extra={"error": "episode_exception", "grading_error": str(e),
                   "failure_phase": getattr(e, "phase", "episode"),
                   "exception_type": type(e).__name__,
                   "request_error": getattr(e, "diagnostics", None)},
            grading_status="timeout" if isinstance(e, TimeoutError) or getattr(e, "is_timeout", False) else "infrastructure_error",
""",
    )
    generate = _replace(
        generate,
        "async def generate(args, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):\n",
        """_eval_limit = None


async def generate(args, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    # The outer Slime semaphore covers train and eval. This additional limit
    # applies only to eval episodes, including their sandbox/tool lifetime.
    limit = int(getattr(args, "agentic_eval_concurrency", 0) or 0)
    if evaluation and limit > 0:
        global _eval_limit
        loop = asyncio.get_running_loop()
        if _eval_limit is None or _eval_limit[:2] != (loop, limit):
            _eval_limit = (loop, limit, asyncio.Semaphore(limit))
        async with _eval_limit[2]:
            return await _generate_episode(args, sample, sampling_params, evaluation)
    return await _generate_episode(args, sample, sampling_params, evaluation)


async def _generate_episode(args, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
""",
    )
    generate = _replace(
        generate,
        "    limits = _limits(args)\n",
        "    limits = _limits(args)\n"
        '    md["grade_after_generation_error"] = bool(evaluation and getattr(args, "agentic_eval_recover_generation_errors", False))\n',
    )
    generate = _replace(
        generate,
        '        "gen_timestamp": time.time(),\n',
        '        "gen_timestamp": time.time(),\n'
        '        "generation_request_errors": list(getattr(model, "request_errors", [])),\n',
    )
    rollout = _replace(
        sources["rollout"],
        '        if (samples := data[key].get("samples")) is not None:\n',
        '        if (samples := data[key].get("samples")) is not None:\n'
        "            from agentic_rl.core.request_errors import evaluation_diagnostics\n"
        '            log_dict |= dict_add_prefix(evaluation_diagnostics(samples), f"eval/{key}/")\n',
    )
    sources.update(model=model, env=env, generate=generate, rollout=rollout)
    for name, source in sources.items():
        compile(source, str(paths[name]), "exec")
    errors = root / "agentic_rl/core/request_errors.py"
    compile(ERROR_SOURCE, str(errors), "exec")
    for name, source in sources.items():
        paths[name].write_text(source)
    errors.write_text(ERROR_SOURCE)


def image_patch_command() -> str:
    encoded = base64.b64encode(Path(__file__).read_bytes()).decode()
    return f"echo {encoded} | base64 -d | python3"


if __name__ == "__main__":
    patch(Path("/root/slime"))
