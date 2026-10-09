"""HarborEnv: harbor-format tasks (USACO, SWE-rebench-V2-as-harbor, ...) as RL
episodes on Modal.

The converter (``convert2slime/harbor.py``) bakes everything into ``metadata`` so
this never reads ``task.toml`` at rollout. Episode (harbor "shared" verifier
semantics): boot the sandbox, then per step write the instruction, run one agent
leg, verify IN-PLACE (upload tests/, run test.sh, parse reward.{json,txt}), and
gate on min_reward. Per-step rewards aggregate (mean | final) to a scalar. Tests
are uploaded only AFTER the agent leg so the agent can't read them.

In-place grading is intrinsic to harbor: the deliverable is the sandbox's
filesystem state (there is no patch to transplant to a fresh sandbox). Needs
``ASYNC_RL_TASK_ROOT`` (dir relative ``task_path``s resolve against). Oracle
check (no model):

    python -m agentic_rl.envs.harbor.env out/usaco.jsonl --task-root out --limit 3
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import shlex
import time
from pathlib import Path
from typing import Any

from agentic_rl.core.sandbox import DockerfileImage, Sandbox
from agentic_rl.core.timing import PhaseTimer
from agentic_rl.rewards import rewards as rewards_mod
from ..base import EnvMetadataError, EpisodeLimits, RewardResult, RolloutEnv, coerce_prompt

logger = logging.getLogger("agentic_rl")

TASK_ROOT_ENV = "ASYNC_RL_TASK_ROOT"

# ${VAR} / ${VAR:-default} in verifier env values, resolved against the head's os.environ.
_ENV_TEMPLATE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*))?\}$")


def _resolve_env_templates(env: dict[str, str] | None) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, value in (env or {}).items():
        m = _ENV_TEMPLATE.fullmatch(str(value))
        if not m:
            resolved[key] = str(value)
            continue
        name, default = m.group(1), m.group(2)
        actual = os.environ.get(name, default)
        if actual is None:
            logger.warning("[harbor] env %s=${%s} unresolvable on this host; skipping", key, name)
            continue
        resolved[key] = actual
    return resolved


def _meets_min_reward(rewards: dict[str, Any] | None, min_reward: float | dict[str, float] | None) -> bool:
    if min_reward is None:
        return True
    if isinstance(min_reward, dict):
        return all(
            rewards is not None and k in rewards and float(rewards[k]) >= float(v) for k, v in min_reward.items()
        )
    return rewards is not None and "reward" in rewards and float(rewards["reward"]) >= float(min_reward)


class HarborEnv(RolloutEnv):
    name = "harbor"
    # Drop the step instruction into the workdir as PROBLEM_STATEMENT.md before each
    # leg. FrontierCsEnv turns this off — it folds the statement into the prompt.
    writes_problem_statement_file = True

    def normalize_metadata(self, sample) -> dict[str, Any]:
        m = sample.metadata or {}
        task_path = m.get("task_path")
        if not task_path:
            raise EnvMetadataError("missing_task_path")
        task_dir = self._resolve_task_dir(task_path)

        docker_image = m.get("docker_image")
        dockerfile = m.get("dockerfile")
        if not docker_image and not dockerfile:
            raise EnvMetadataError("missing_docker_image_or_dockerfile")
        if dockerfile and not (task_dir / dockerfile).is_file():
            raise EnvMetadataError(f"dockerfile_missing:{task_dir / dockerfile}")

        steps = m.get("steps") or None
        if steps:
            for step in steps:
                if not (task_dir / step["tests_path"] / "test.sh").is_file():
                    raise EnvMetadataError(f"tests_missing:{step['tests_path']}")
        elif not (task_dir / "tests" / "test.sh").is_file():
            raise EnvMetadataError("tests_missing:tests")

        return {
            "instance_id": m.get("instance_id") or sample.label or task_dir.name,
            "task_dir": str(task_dir),
            "docker_image": docker_image,
            "dockerfile": dockerfile,
            "workdir": m.get("workdir"),
            "problem_statement": m.get("problem_statement") or coerce_prompt(sample.prompt),
            "verifier": m.get("verifier") or {},
            "steps": steps,
            "reward_strategy": m.get("reward_strategy"),
            "cpus": m.get("cpus"),
            "memory_mb": m.get("memory_mb"),
            "reward_shape": m.get("reward_shape"),
            "outcome_reward": m.get("outcome_reward"),
        }

    @staticmethod
    def _resolve_task_dir(task_path: str) -> Path:
        p = Path(task_path)
        if not p.is_absolute():
            root = os.environ.get(TASK_ROOT_ENV)
            if not root:
                raise EnvMetadataError(f"{TASK_ROOT_ENV}_unset")
            p = Path(root) / p
        if not p.is_dir():
            raise EnvMetadataError(f"task_dir_missing:{p}")
        return p

    def _image(self, md: dict[str, Any]) -> str | DockerfileImage:
        if md["docker_image"]:
            return md["docker_image"]
        path = Path(md["task_dir"]) / md["dockerfile"]
        return DockerfileImage(path=str(path), context_dir=str(path.parent))

    def _sandbox(self, md: dict[str, Any], *, lifetime: int, exec_timeout: int) -> Sandbox:
        return Sandbox(
            self._image(md),
            cwd=md["workdir"] or "/",
            lifetime=lifetime,
            exec_timeout=exec_timeout,
            cpu=md.get("cpus"),
            memory_mb=md.get("memory_mb"),
            env=md.get("agent_extra_env") or None,
            vm_runtime=os.environ.get("AGENTIC_SANDBOX_VM_RUNTIME", "0").strip().lower() in ("1", "true", "yes"),
        )

    def _step_specs(self, md: dict[str, Any]) -> list[dict[str, Any]]:
        if md["steps"]:
            return md["steps"]
        return [
            {
                "name": None,
                "instruction": md["problem_statement"],
                "tests_path": "tests",
                "verifier": md["verifier"],
                "min_reward": None,
            }
        ]

    def rollout(self, md: dict[str, Any], *, model, limits: EpisodeLimits) -> RewardResult:
        def agent_leg(sb, instruction: str, budget_sec: int) -> dict:
            return self.run_agent_leg(model, sb, instruction, max_steps=limits.max_steps, wall_time_sec=budget_sec)

        return self._episode(md, run_leg=agent_leg, agent_budget_sec=limits.episode_timeout, limits=limits)

    def _episode(self, md: dict[str, Any], *, run_leg, agent_budget_sec: int, limits: EpisodeLimits) -> RewardResult:
        """Shared by the RL rollout and the oracle check: only the leg differs."""
        task_dir = Path(md["task_dir"])
        steps = self._step_specs(md)
        step_results: list[dict[str, Any]] = []
        timer = PhaseTimer()
        grading_status = "valid"
        grading_error = None

        t0 = time.monotonic()
        artifacts: dict[str, Any] = {}
        sandbox_exec: dict[str, int] = {}
        sb, workdir = self._prepare_sandbox(md, steps[0], agent_budget_sec, limits, timer)
        with sb:

            # Start the agent clock only after boot+prep: a cold image pull can take
            # minutes, and charging it against the agent budget would exhaust the
            # window before any step runs.
            deadline = time.monotonic() + agent_budget_sec
            for index, step in enumerate(steps):
                remaining = int(deadline - time.monotonic())
                if remaining <= 0:
                    logger.warning(
                        "[harbor] %s: agent budget exhausted before step %r", md["instance_id"], step["name"]
                    )
                    break
                leg_md = {**md, "workdir": workdir}
                if index:
                    with timer.phase("prep"):
                        if self.writes_problem_statement_file:
                            self.write_problem_file(sb, workdir, step["instruction"])
                        self._pre_agent_setup(sb, task_dir, leg_md)
                with timer.phase("agent"):
                    run_leg(sb, step["instruction"], remaining)
                with timer.phase("verifier"):
                    try:
                        rewards = self._verify(
                            sb,
                            tests_dir=task_dir / step["tests_path"],
                            workdir=workdir,
                            verifier={**md["verifier"], **(step.get("verifier") or {})},
                            eval_timeout_sec=limits.eval_timeout,
                            instance_id=md["instance_id"],
                        )
                        if not rewards:
                            raise ValueError("verifier returned no grade")
                        signal = rewards_mod.signal_from_reward_dict(rewards)
                        if not math.isfinite(signal.score_raw):
                            raise ValueError("verifier returned a non-finite grade")
                    except Exception as e:
                        grading_status = "timeout" if isinstance(e, TimeoutError) else "infrastructure_error"
                        grading_error = str(e)
                        logger.exception("[harbor] %s: grading failed (%s)", md["instance_id"], grading_status)
                        break
                shaped = rewards_mod.shape(signal, rewards_mod.resolve_shape(md))
                step_results.append(
                    {"name": step["name"], "rewards": rewards, "reward": shaped, "is_solved": signal.is_solved}
                )
                if not _meets_min_reward(rewards, step.get("min_reward")):
                    logger.info(
                        "[harbor] %s: step %r below min_reward; aborting remaining steps",
                        md["instance_id"],
                        step["name"],
                    )
                    break

            # Pull post-episode artifacts off the still-live sandbox (the env tears
            # it down on __exit__); must never fail the episode.
            try:
                artifacts = self._collect_artifacts(sb, workdir, md)
            except Exception:  # noqa: BLE001
                logger.exception("[harbor] %s: artifact collection failed", md["instance_id"])
            # Per-command walls distinguish actual sandbox/tool latency from the
            # configured timeout ceiling. Keep the raw, compact duration vector so
            # rollout metrics can emit p50/p90/p99 rather than hiding a slow tail.
            sandbox_exec = {
                "exec_count": sb.exec_count,
                "exec_time": round(sb.exec_time, 3),
                "exec_timeouts": sb.exec_timeouts,
                "exec_durations": [round(value, 3) for value in sb.exec_durations],
            }

        reward = self._aggregate(steps, step_results, md["reward_strategy"])
        is_solved = (
            bool(step_results) and len(step_results) == len(steps) and all(r.get("is_solved") for r in step_results)
        )
        # Episode-level outcome shaping (rewards.py): fold the mid-episode judge
        # submissions (frontier_cs artifacts) into the training scalar per the
        # ASYNC_RL_OUTCOME_REWARD / ASYNC_RL_SOLVED_BONUS switches. Defaults are
        # the identity (reward == the aggregate above); no submissions -> ditto.
        outcome = rewards_mod.episode_outcome_from_artifacts(reward, is_solved, artifacts)
        reward, outcome_info = rewards_mod.shape_outcome(outcome, rewards_mod.resolve_outcome(md))
        timer.record("episode", time.monotonic() - t0)
        return RewardResult(
            reward=reward if grading_status == "valid" else 0.0,
            is_solved=is_solved and grading_status == "valid",
            grading_status=grading_status,
            extra={
                "grading_error": grading_error,
                "outcome": outcome_info,
                "harbor_step_results": step_results,
                "harbor_steps_completed": len(step_results),
                "harbor_steps_total": len(steps),
                "timing": timer.as_dict(),
                **sandbox_exec,
                **artifacts,
            },
        )

    def _prepare_sandbox(self, md, step, agent_budget_sec, limits, timer):
        retries = Sandbox._rpc_retries_from_env()
        for attempt in range(retries):
            sb = None
            try:
                with timer.phase("boot"):
                    sb = self._sandbox(
                        md, lifetime=agent_budget_sec + limits.grade_timeout + 300, exec_timeout=limits.exec_timeout
                    )
                with timer.phase("prep"):
                    workdir = md["workdir"] or self._detect_workdir(sb)
                    sb.exec(
                        f"mkdir -p {shlex.quote(workdir)} /logs/agent /logs/verifier /logs/artifacts",
                        check=True,
                        timeout=60,
                        idempotent=True,
                    )
                    if self.writes_problem_statement_file:
                        self.write_problem_file(sb, workdir, step["instruction"])
                    self._pre_agent_setup(sb, Path(md["task_dir"]), {**md, "workdir": workdir})
                return sb, workdir
            except Exception as error:
                if sb is not None:
                    sb.terminate()
                if attempt + 1 == retries or not Sandbox._is_transient_rpc_error(error):
                    raise
                logger.warning("[harbor] %s: retrying sandbox setup: %s", md["instance_id"], error)
                ceiling = min(Sandbox.rpc_backoff_cap_sec, Sandbox.rpc_backoff_base_sec * (2**attempt))
                time.sleep(random.uniform(0.0, ceiling))

    @staticmethod
    def _aggregate(steps: list[dict], results: list[dict], strategy: str | None) -> float:
        if not results:
            return 0.0
        if len(steps) == 1:
            return results[0]["reward"]
        if (strategy or "mean") == "final":
            return results[-1]["reward"] if len(results) == len(steps) else 0.0
        return sum(r["reward"] for r in results) / len(steps)

    @staticmethod
    def _detect_workdir(sb) -> str:
        ec, out, _ = sb.exec("pwd", check=False, timeout=30)
        detected = (out or "").strip().splitlines()[-1] if ec == 0 and (out or "").strip() else ""
        return detected or "/app"

    def _pre_agent_setup(self, sb, task_dir: Path, md: dict[str, Any]) -> None:
        """Hook run in the workdir before each agent leg. Default no-op;
        FrontierCsEnv overrides it to stage statement.txt / submit.sh into /app."""

    def _collect_artifacts(self, sb, workdir: str, md: dict[str, Any]) -> dict[str, Any]:
        """Hook run on the still-live sandbox after the agent leg(s); its return is
        merged into ``RewardResult.extra`` (→ ``sample.metadata``). Default none;
        FrontierCsEnv overrides it to pull the episode's judge-server submission
        record (``md`` carries its per-episode attribution id)."""
        return {}

    def _verify(
        self,
        sb,
        *,
        tests_dir: Path,
        workdir: str,
        verifier: dict[str, Any],
        eval_timeout_sec: int | None,
        instance_id: str,
    ) -> dict[str, Any] | None:
        # A config-level agentic_eval_timeout (eval_timeout_sec) OVERRIDES the per-task
        # verifier timeout when set; otherwise fall back to the task's baked timeout_sec
        # (the converter bakes 1800s, which lets a hung test suite burn the full budget
        # and gate the whole sync step). Final fallback 600s if neither is present.
        timeout = int(eval_timeout_sec or verifier.get("timeout_sec") or 600)
        self.upload_dir(sb, tests_dir, "/tests")
        sb.exec(
            "chmod +x /tests/test.sh && rm -f /logs/verifier/reward.json /logs/verifier/reward.txt",
            check=True,
            timeout=60,
        )
        env = _resolve_env_templates(verifier.get("env"))
        q = shlex.quote
        ec, out, err = sb.exec(
            f"cd {q(workdir)} && bash /tests/test.sh", env=env or None, timeout=timeout, check=False
        )
        if os.environ.get("HARBOR_VERIFY_DEBUG"):
            logger.info(
                "[harbor-verify-debug] %s: test.sh exit=%s\nstdout:\n%s\nstderr:\n%s",
                instance_id,
                ec,
                (out or "")[-3000:],
                (err or "")[-3000:],
            )
        if ec in (-1, 124):
            raise TimeoutError(f"verifier timed out after {timeout}s (exit={ec})")

        raw_json = sb.read_file("/logs/verifier/reward.json")
        if raw_json.strip():
            try:
                return dict(json.loads(raw_json))
            except (ValueError, TypeError):
                logger.warning("[harbor] %s: unparseable reward.json: %.200s", instance_id, raw_json)
                return None
        raw_txt = sb.read_file("/logs/verifier/reward.txt")
        if raw_txt.strip():
            try:
                return {"reward": float(raw_txt.strip())}
            except ValueError:
                logger.warning("[harbor] %s: unparseable reward.txt: %.200s", instance_id, raw_txt)
                return None
        # No reward file: terminal-bench-style tasks end test.sh with a bare pytest run.
        if ec in (0, 1):
            return {"reward": 1.0 if ec == 0 else 0.0, "graded_from": "exit_code"}
        logger.warning(
            "[harbor] %s: test.sh exit=%s wrote no reward file; stderr: %s", instance_id, ec, (err or "")[-400:]
        )
        raise RuntimeError(f"verifier exited {ec} without a reward file: {(err or '')[-400:]}")

    # oracle check (reference solution through the exact rollout path) -------
    def oracle_episode(self, md: dict[str, Any], *, solve_timeout_sec: int, eval_timeout_sec: int) -> RewardResult:
        task_dir = Path(md["task_dir"])
        steps = self._step_specs(md)
        state = {"i": 0}

        def leg(sb, instruction: str, budget_sec: int) -> dict:
            step = steps[state["i"]]
            state["i"] += 1
            solution_dir = task_dir / "steps" / step["name"] / "solution" if step["name"] else task_dir / "solution"
            if not (solution_dir / "solve.sh").is_file():
                solution_dir = task_dir / "solution"
            if not (solution_dir / "solve.sh").is_file():
                raise FileNotFoundError(f"no solution/solve.sh under {task_dir}")
            self.upload_dir(sb, solution_dir, "/solution")
            q = shlex.quote
            sb.exec("chmod +x /solution/solve.sh", check=False, timeout=30)
            ec, _, err = sb.exec(
                f"cd {q(md['workdir'])} && bash /solution/solve.sh",
                timeout=min(budget_sec, solve_timeout_sec),
                check=False,
            )
            if ec != 0:
                logger.warning("[harbor-oracle] solve.sh exit=%d stderr: %s", ec, (err or "")[-400:])
            return {}

        limits = EpisodeLimits(
            max_steps=0,
            episode_timeout=solve_timeout_sec * max(1, len(steps)),
            grade_timeout=eval_timeout_sec,
            eval_timeout=eval_timeout_sec,
        )
        return self._episode(md, run_leg=leg, agent_budget_sec=limits.episode_timeout, limits=limits)


def _oracle_main() -> int:
    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(
        description="Run harbor reference solutions through the rollout path (reward should be 1.0)."
    )
    parser.add_argument("jsonl", help="converted slime prompt JSONL (convert2slime/harbor.py output)")
    parser.add_argument("--task-root", help=f"task root (default: ${TASK_ROOT_ENV} or the JSONL's directory)")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--index", type=int)
    parser.add_argument("--solve-timeout", type=int, default=600)
    parser.add_argument("--eval-timeout", type=int, default=int(os.environ.get("AGENT_EVAL_TIMEOUT_SEC", "600")))
    parser.add_argument("--vm-runtime", action="store_true")
    args = parser.parse_args()
    if args.vm_runtime:
        os.environ["AGENTIC_SANDBOX_VM_RUNTIME"] = "1"

    os.environ[TASK_ROOT_ENV] = (
        args.task_root or os.environ.get(TASK_ROOT_ENV) or str(Path(args.jsonl).resolve().parent)
    )
    rows = [json.loads(line) for line in open(args.jsonl, encoding="utf-8") if line.strip()]
    picked = [rows[args.index]] if args.index is not None else rows[: args.limit]

    env = HarborEnv()
    failures = 0
    for row in picked:
        sample = SimpleNamespace(metadata=row.get("metadata"), prompt=row.get("prompt"), label=row.get("label"))
        md = env.normalize_metadata(sample)
        t0 = time.monotonic()
        result = env.oracle_episode(md, solve_timeout_sec=args.solve_timeout, eval_timeout_sec=args.eval_timeout)
        status = "OK " if result.is_solved else "FAIL"
        print(
            f"[{status}] {md['instance_id']}: reward={result.reward:.2f} t={time.monotonic() - t0:.0f}s {result.extra}"
        )
        failures += 0 if result.is_solved else 1
    return 1 if failures else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(_oracle_main())
