from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import modal
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.training_group import TrainingGroup
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

app = modal.App("toolathlon")

# Launcher-only deps. The slime rollout workers import THIS module just to load
# toolathlon_generate / reward_func, so let the module import even where modal /
# modal_training_gym aren't installed.
try:
    from modal_training_gym import HarborDataset, Qwen3_6_35B, TrainConfig, WandbConfig
except ImportError:
    HarborDataset = object  # type: ignore[assignment,misc]


# -----------------------------------------------------------------------------
# Tuned parameters
# -----------------------------------------------------------------------------
TEMPERATURE = 1.0
QWEN3_TOOL_STOP_TOKEN_IDS = (151329, 151336, 151338)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GIT_URL = "https://github.com/hkust-nlp/Toolathlon.git"
GIT_COMMIT = "main"
REPO_SUBDIR = "tasks/finalpool"
INSTRUCTION_FILE = "docs/task.md"
TASK_CONFIG_FILE = "task_config.json"

MINI_SWE_SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MINI_SWE_COMMAND_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}
NEXT_RESPONSE_CONTRACT_RE = re.compile(
    r"\n?<next_response_contract>.*?</next_response_contract>\n?",
    re.DOTALL,
)

# Wall-clock ceiling for a single live Mini SWE episode.
EPISODE_TIMEOUT_SEC = 300
MINI_SWE_STEP_LIMIT = 20
# Keep the reasoning pass on. thinking_budget=0 (skip the separate reasoning
# call) was tried to cut latency/transcript length, but eval showed it CRIPPLES
# task completion — the model flails and never submits (0/15 pass vs 3/15 with
# reasoning restored). Reasoning is worth the latency here.
MINI_SWE_THINKING_BUDGET = 4096
# Cap observations to bound the training transcript: a real run truncated ~19%
# of samples at the 32k-token ceiling (raw file dumps ballooned the transcript;
# truncated samples earned 0.32 reward vs 0.52). 4k was too aggressive — it
# starved data tasks of file content (0/15 pass). 16k keeps episodes under the
# ~32k sequence cap (truncation ~0% in eval) while leaving enough context to work.
MINI_SWE_OBSERVATION_CHAR_LIMIT = 16000
MINI_SWE_OBSERVATION_EDGE_CHARS = 8000

TASK_LIMIT = None
ROLLOUT_RESPONSE_LEN = 8192
ROLLOUT_CONTEXT_LEN = 32768
MAX_TOKENS_PER_GPU = 8192

# Length penalty on the model's OWN generated (assistant) tokens — the
# loss-masked turns the policy actually controls — NOT total response_length,
# which is ~80% tool-observation tokens the model doesn't write. Applied as an
# additive cost to passes AND fails so verbosity is discouraged regardless of
# task success: cost is 0 up to LENGTH_PENALTY_FREE_TOKENS, then grows linearly to
# LENGTH_PENALTY_MAX_COST at LENGTH_PENALTY_MAX_TOKENS (clamped beyond).
# NOTE: initial thresholds — calibrate against toolathlon_assistant_tokens_mean.
LENGTH_PENALTY_FREE_TOKENS = 4000
LENGTH_PENALTY_MAX_TOKENS = 16000
LENGTH_PENALTY_MAX_COST = 0.25


class ToolathlonExitStatus(StrEnum):
    SUBMITTED = "Submitted"
    SUCCESS = "ok"
    FAIL = "fail"
    HARNESS_ERROR = "harness-error"
    TIMEOUT = "timeout"
    MISSING_TASK_NAME = "missing-task-name"


def _normalize_exit_status(status: str) -> str:
    if status == "ok":
        return ToolathlonExitStatus.SUCCESS
    return status


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: scan the OTS coding tasks from git, one record per task
# ─────────────────────────────────────────────────────────────────────────────


def _clone_tasks() -> Path:
    """Shallow-clone the OTS task repo and return its `tasks/` root."""
    data_root = (
        Path("/data")
        if os.access("/data", os.W_OK)
        else Path.cwd() / ".toolathlon-data"
    )
    checkout = data_root / "toolathlon-tasks" / GIT_COMMIT

    def has_task_files() -> bool:
        return any((checkout / REPO_SUBDIR).glob(f"*/{INSTRUCTION_FILE}"))

    if checkout.exists() and not has_task_files():
        shutil.rmtree(checkout, ignore_errors=True)

    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                GIT_COMMIT,
                GIT_URL,
                checkout.as_posix(),
            ],
            check=True,
        )
    if not has_task_files():
        raise FileNotFoundError(
            f"Cloned {GIT_URL}@{GIT_COMMIT}, but found no {INSTRUCTION_FILE} under "
            f"{checkout / REPO_SUBDIR}"
        )
    return checkout / REPO_SUBDIR


# Explicit per-task labels Toolathlon may stamp under ``task_config.json:meta``.
# Surfaced verbatim onto each sample's metadata when present so runs can be
# sliced/filtered by task type in the dashboard.
HARBOR_META_FIELDS = ("task_type", "category", "difficulty", "tags", "description")


def _task_config(task_dir: Path) -> dict[str, Any]:
    config_file = task_dir / TASK_CONFIG_FILE
    if not config_file.is_file():
        return {}
    try:
        return json.loads(config_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _task_metadata(
    task_dir: Path, prompt: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Per-sample metadata: the task identity/request plus the Harbor labels that
    say *what kind* of task it is. ``task_type`` prefers an explicit
    ``meta.task_type``/``meta.category`` label and otherwise falls back to the
    set of MCP servers the task exercises (its reliable type signature)."""
    needed = sorted(config.get("needed_mcp_servers") or [])
    meta = config.get("meta") if isinstance(config.get("meta"), dict) else {}
    metadata: dict[str, Any] = {
        "task_name": task_dir.name,
        "task_request": prompt,
        "needed_mcp_servers": needed,
        "needed_local_tools": sorted(config.get("needed_local_tools") or []),
        "task_type": meta.get("task_type")
        or meta.get("category")
        or ("+".join(needed) or "unknown"),
    }
    metadata.update({k: meta[k] for k in HARBOR_META_FIELDS if k in meta})
    return metadata


class ToolathlonDataset(HarborDataset):
    """Tier-A Toolathlon tasks as a :class:`HarborDataset` over the git-cloned
    ``tasks/finalpool``.

    Harbor's scanning (glob tasks → read instruction → build label → write
    splits) is reused wholesale; only three Toolathlon-specific bits are
    overridden: cloning the repo (Harbor pulls registry datasets, not arbitrary
    git URLs), the Tier-A MCP filter, and reshaping ``task_config.json`` into the
    derived metadata the rollout/eval read back from the label.
    """

    label_key = "label"
    output_format = "jsonl"
    always_prepare = True
    instruction_path = INSTRUCTION_FILE

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            dataset_name="toolathlon-tasks",
            input_key="messages",
            apply_chat_template=True,
            **kwargs,
        )

    def _resolve_task_root(self) -> Path:
        # Toolathlon ships as a git repo, not a Harbor-registry dataset: clone it
        # and point Harbor's task scanning at tasks/finalpool.
        return _clone_tasks()

    def _candidate_task_dirs(self, task_root: Path) -> list[Path]:
        # Only Tier-A tasks (MCP state confined to the workspace) can run in the
        # decoupled sandbox env; tasks backed by external services (google-cloud,
        # canvas, email, …) are out of scope and are skipped here.
        from modal_training_gym.common.environments.toolathlon import TIER_A_MCPS

        kept = []
        for task_dir in super()._candidate_task_dirs(task_root):
            if not (task_dir / INSTRUCTION_FILE).is_file():
                continue
            needed = set(_task_config(task_dir).get("needed_mcp_servers") or [])
            if needed and not needed <= TIER_A_MCPS:
                continue
            kept.append(task_dir)
            if TASK_LIMIT and len(kept) >= TASK_LIMIT:
                break
        return kept

    def _build_label(self, task_root: Path, task_dir: Path) -> dict[str, Any]:
        # Reshape task_config.json into the derived metadata (task_type fallback,
        # sorted MCP servers, surfaced Harbor labels) the rollout/eval read back.
        prompt = (task_dir / INSTRUCTION_FILE).read_text().strip()
        return _task_metadata(task_dir, prompt, _task_config(task_dir))


MINI_SWE_SYSTEM_TEMPLATE = f"""\
You are a coding agent that can interact with a computer.

Every response must contain exactly ONE fenced bash action block.
Never answer with prose only. Never omit the action block.
The action block must be tagged exactly `mswea_bash_command`.
Final prose summaries are invalid. To finish, run the submit echo command.

Correct format:

THOUGHT: Briefly explain the next shell action.

```mswea_bash_command
one_bash_command_here
```

If the task is complete, your only command must be:

```mswea_bash_command
echo {MINI_SWE_SUBMIT_SENTINEL}
```
"""

MINI_SWE_INSTANCE_TEMPLATE = f"""\
Please solve this issue: {{{{task}}}}

You can execute bash commands and edit files to implement the necessary changes.

## Required workflow

1. Inspect the workspace.
2. Reproduce or understand the issue.
3. Edit files to implement the fix.
4. Run targeted verification.
5. Submit by issuing exactly this command in the required action block:
   `echo {MINI_SWE_SUBMIT_SENTINEL}`

## Hard formatting rules

- Every response must include exactly one fenced bash action block.
- The fence tag must be `mswea_bash_command`.
- Do not answer in plain text without a command block.
- After every observation, continue with another action block or submit.
- If tests pass or the fix appears complete, submit immediately with
  `echo {MINI_SWE_SUBMIT_SENTINEL}`.

<system_information>
{{{{system}}}} {{{{release}}}} {{{{version}}}} {{{{machine}}}}
</system_information>
"""


def _build_observation_template(char_limit: int, edge_chars: int | None = None) -> str:
    """Mini-SWE observation template with a configurable truncation budget.

    Output longer than ``char_limit`` chars is replaced by head+tail slices of
    ``edge_chars`` each (default ``char_limit // 2``) plus an elision notice.
    Keeping this budget tight bounds how fast the per-step transcript (and thus
    prefill cost) grows over a long agent loop.
    """
    edge = edge_chars if edge_chars is not None else char_limit // 2
    return f"""\
{{% if output.exception_info -%}}
<exception>{{{{output.exception_info}}}}</exception>
{{% endif -%}}
<returncode>{{{{output.returncode}}}}</returncode>
{{% if output.output | length <= {char_limit} -%}}
<output>
{{{{ output.output -}}}}
</output>
{{%- else -%}}
<warning>
The output of your last command was too long.
Please try a different command that produces less output.
</warning>
<output_head>
{{{{ output.output[:{edge}] }}}}
</output_head>
<elided_chars>
{{{{ output.output | length - {edge * 2} }}}} characters elided
</elided_chars>
<output_tail>
{{{{ output.output[-{edge}:] }}}}
</output_tail>
{{%- endif %}}

<next_response_contract>
Your next response must contain exactly one fenced bash action block:

```mswea_bash_command
one_bash_command_here
```

Do not respond with prose only. If done, run:

```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
If your next response would summarize completion, run the submit command instead.
</next_response_contract>
"""


MINI_SWE_OBSERVATION_TEMPLATE = _build_observation_template(
    MINI_SWE_OBSERVATION_CHAR_LIMIT, MINI_SWE_OBSERVATION_EDGE_CHARS
)

MINI_SWE_FORMAT_ERROR_TEMPLATE = """\
Your previous response was rejected because it contained {{actions|length}} fenced bash actions; exactly one is required.

Reply now using exactly this structure:

THOUGHT: Briefly describe the next shell command.

```mswea_bash_command
one_bash_command_here
```

If the task is complete, the command must be:

```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""


def _ensure_v1_suffix(base_url: str) -> str:
    split = urlsplit(base_url.rstrip("/"))
    if split.path.endswith("/v1"):
        return urlunsplit(split)
    path = f"{split.path.rstrip('/')}/v1" if split.path else "/v1"
    return urlunsplit((split.scheme, split.netloc, path, split.query, split.fragment))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _exception_summary(exc: BaseException, *, limit: int = 1000) -> str:
    message = str(exc).strip() or repr(exc).strip()
    name = type(exc).__name__
    if not message or message == f"{name}()":
        return name
    if message.startswith(f"{name}:") or message.startswith(f"{name}("):
        return message[:limit]
    return f"{name}: {message}"[:limit]


def _trajectory_messages(trajectory: Any) -> list[dict[str, Any]]:
    if not isinstance(trajectory, dict):
        return []
    raw_messages = trajectory.get("messages")
    if not isinstance(raw_messages, list):
        return []

    messages: list[dict[str, Any]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        content = raw.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = json.dumps(content)
        message: dict[str, Any] = {
            "role": str(raw.get("role") or "unknown"),
            "content": content,
        }
        extra = raw.get("extra")
        if isinstance(extra, dict) and "exit_status" in extra:
            message["exit_status"] = extra["exit_status"]
        messages.append(message)
    return messages


def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role", "")).lower() != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


@lru_cache(maxsize=1)
def _budget_forcing_model_cls():
    import litellm
    from minisweagent.models.litellm_textbased_model import (  # type: ignore[import-not-found]
        LitellmTextbasedModel,
    )

    class _BudgetForcingTextbasedModel(LitellmTextbasedModel):
        def __init__(
            self,
            *,
            thinking_budget: int,
            force_phrase: str,
            max_tokens_total: int,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self._thinking_budget = max(0, int(thinking_budget))
            self._force_phrase = force_phrase
            self._max_tokens_total = max(256, int(max_tokens_total))

        @staticmethod
        def _raw_reasoning_extra_body(base: dict[str, Any]) -> dict[str, Any]:
            extra_body = dict(base.get("extra_body") or {})
            extra_body["separate_reasoning"] = False
            return extra_body

        def _think_prefix(
            self, messages: list[dict[str, str]], merged: dict[str, Any]
        ) -> tuple[str, int]:
            if self._thinking_budget <= 0:
                return "<think>\n</think>\n\n", 0

            think_kwargs = dict(merged)
            think_kwargs["max_tokens"] = self._thinking_budget
            think_kwargs["stop"] = sorted(
                {*(think_kwargs.get("stop") or []), "</think>"}
            )
            think_kwargs["extra_body"] = self._raw_reasoning_extra_body(merged)
            response = litellm.completion(
                model=self.config.model_name,
                messages=messages,
                **think_kwargs,
            )
            choice = response.choices[0]
            think_inner = (choice.message.content or "").split("</think>")[0].rstrip()
            if "<think>" not in think_inner:
                think_inner = "<think>\n" + think_inner
            usage = getattr(response, "usage", None)
            think_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            if str(getattr(choice, "finish_reason", "")) == "length":
                return f"{think_inner}\n{self._force_phrase}\n\n", think_tokens
            return f"{think_inner}\n</think>\n\n", think_tokens

        def _query(self, messages: list[dict[str, str]], **kwargs):
            merged = self.config.model_kwargs | kwargs
            prefix, think_tokens = self._think_prefix(messages, merged)

            answer_kwargs = dict(merged)
            answer_kwargs["max_tokens"] = max(
                256, self._max_tokens_total - think_tokens
            )
            answer_kwargs.pop("stop", None)
            extra_body = self._raw_reasoning_extra_body(merged)
            extra_body["continue_final_message"] = True
            extra_body["add_generation_prompt"] = False
            answer_kwargs["extra_body"] = extra_body
            response = litellm.completion(
                model=self.config.model_name,
                messages=[*messages, {"role": "assistant", "content": prefix}],
                **answer_kwargs,
            )
            response.choices[0].message.content = prefix + (
                response.choices[0].message.content or ""
            )
            return response

        # Fence tags the model actually uses for its action block, in preference
        # order. Eval diagnostics (run windy-inversion) showed the dominant
        # failure was NOT unclosed blocks but a *mis-tagged fence*: the model
        # emits a valid command in ```bash / ```bash_command instead of the
        # required ```mswea_bash_command, so the strict parser finds 0 actions and
        # re-prompts — burning ~every such step into LimitsExceeded (reward 0).
        _ACTION_FENCE_TAGS = (
            "mswea_bash_command",
            "bash_command",
            "bash",
            "shell",
            "sh",
        )

        def _parse_actions(self, response):
            """Fence-tolerant action parsing.

            Accepts the intended ```mswea_bash_command tag and the mis-tagged
            fences the model actually emits (```bash, ```bash_command, …), taking
            the FIRST matching block and recovering an unclosed one. The <think>
            block is stripped first so its markdown fences (e.g. a directory-tree
            example) are never parsed as the action. A genuinely command-free
            response still falls through to the stock FormatError re-prompt.
            """
            content = response.choices[0].message.content or ""
            body = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            for tag in self._ACTION_FENCE_TAGS:
                fence = rf"```{tag}[ \t]*\n"
                closed = re.findall(fence + r"(.*?)\n```", body, re.DOTALL)
                if closed:
                    return [{"command": closed[0].strip()}]
                m = re.search(fence + r"(.*)\Z", body, re.DOTALL)
                if m:
                    cmd = re.sub(r"\n[ \t]*`{1,3}[ \t]*\Z", "", m.group(1)).strip()
                    if cmd:
                        return [{"command": cmd}]
            # Generic fallback: any fence tag containing "bash" — catches typos of
            # the command tag seen in eval (```tswea_bash_command,
            # ```msheia_bash_command). Restricted to bash-ish tags so stray
            # markdown fences in the prose aren't mistaken for actions.
            gen = r"```\S*bash\S*[ \t]*\n"
            closed = re.findall(gen + r"(.*?)\n```", body, re.DOTALL)
            if closed:
                return [{"command": closed[0].strip()}]
            m = re.search(gen + r"(.*)\Z", body, re.DOTALL)
            if m:
                cmd = re.sub(r"\n[ \t]*`{1,3}[ \t]*\Z", "", m.group(1)).strip()
                if cmd:
                    return [{"command": cmd}]
            # Diagnostic: mini-swe raises FormatError before storing the response,
            # so a failing turn is invisible in the saved trajectory. Log what the
            # model actually emitted so any residual failure shape stays visible.
            print(
                "[toolathlon] format-reject-content "
                f"len={len(content)} tail={content[-300:]!r}",
                flush=True,
            )
            return super()._parse_actions(response)

    return _BudgetForcingTextbasedModel


class _MiniSweSandboxEnvironment:
    def __init__(self, env: Any, *, timeout: int) -> None:
        self._env = env
        self.cwd = env.config.workspace_path
        self.timeout = timeout
        self.config = None
        self.tool_execution_latencies: list[float] = []

    def execute(
        self, action: dict, cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        command = str(action.get("command") or "")
        run_cwd = cwd or self.cwd
        run_timeout = int(timeout or self.timeout)
        exports = " ".join(
            f"{shlex.quote(key)}={shlex.quote(value)}"
            for key, value in MINI_SWE_COMMAND_ENV.items()
        )
        script = (
            f"cd {shlex.quote(run_cwd)} && "
            f"env {exports} timeout --preserve-status {run_timeout}s bash -lc "
            f"{shlex.quote(command)}"
        )
        started = time.time()
        # text=False + lenient decode: these tasks routinely cat binary files
        # (xlsx/pdf/pptx/images), and modal's default text=True decodes stdout as
        # strict UTF-8 — non-UTF-8 bytes would raise UnicodeDecodeError and kill
        # the whole episode (reward 0). Decode with errors="replace" instead.
        proc = self._env.sandbox.exec("bash", "-lc", script, text=False)
        proc.wait()
        self.tool_execution_latencies.append(time.time() - started)
        stdout = (proc.stdout.read() or b"").decode("utf-8", "replace")
        stderr = (proc.stderr.read() or b"").decode("utf-8", "replace")
        output = stdout + stderr if stdout and stderr else (stdout or stderr)
        result = {
            "output": output,
            "returncode": proc.returncode,
            "exception_info": "",
        }
        self._check_finished(result)
        return result

    def _check_finished(self, output: dict) -> None:
        from minisweagent.exceptions import Submitted  # type: ignore[import-not-found]

        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == MINI_SWE_SUBMIT_SENTINEL
            and output["returncode"] == 0
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {
                        "exit_status": ToolathlonExitStatus.SUBMITTED,
                        "submission": submission,
                    },
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {
            "cwd": self.cwd,
            "system": "Linux",
            "release": "",
            "version": "",
            "machine": "x86_64",
            **kwargs,
        }

    def serialize(self) -> dict:
        return {"info": {"environment": {"cwd": self.cwd, "timeout": self.timeout}}}


def _create_fresh_toolathlon_env(task_name: str) -> Any:
    from modal_training_gym.common.environments.toolathlon import (
        DEFAULT_CONFIG,
        ToolathlonEnvironment,
        _seed_workspace,
        build_env_image,
    )

    config = DEFAULT_CONFIG
    sandbox = modal.Sandbox._experimental_create(
        "sleep",
        "infinity",
        image=build_env_image(config),
        app=modal.App.lookup(config.sandbox_app, create_if_missing=True),
        timeout=60 * 30,
        cpu=2.0,
        memory=4096,
    )
    try:
        # _seed_workspace seeds the workspace, synthesizes the eval traj_log, and
        # starts the MCP gateway (required for the tasks and env.evaluate()) —
        # mirrors ToolathlonEnvPool.acquire_fresh.
        _seed_workspace(
            sandbox,
            config,
            f"finalpool/{task_name}",
        )
        return ToolathlonEnvironment(
            sandbox,
            config,
            task_name,
        )
    except Exception:
        try:
            sandbox.terminate()
        except Exception:
            pass
        raise


def _mini_swe_model(
    *,
    base_url: str,
    model_name: str,
    sampling_params: dict[str, Any],
    max_new_tokens: int,
    extra_model_kwargs: dict[str, Any] | None = None,
    thinking_budget: int = MINI_SWE_THINKING_BUDGET,
    observation_template: str = MINI_SWE_OBSERVATION_TEMPLATE,
):
    from minisweagent.models.litellm_textbased_model import (  # type: ignore[import-not-found]
        LitellmTextbasedModel,
    )

    requested_max_tokens = (
        _optional_int(sampling_params.get("max_new_tokens"))
        or _optional_int(sampling_params.get("max_tokens"))
        or max_new_tokens
    )
    max_tokens = min(int(requested_max_tokens), int(max_new_tokens))
    model_kwargs: dict[str, Any] = {
        "api_base": _ensure_v1_suffix(base_url),
        "api_key": "local",
        "drop_params": True,
        "temperature": _optional_float(sampling_params.get("temperature"))
        or TEMPERATURE,
        "max_tokens": int(max_tokens),
        "timeout": EPISODE_TIMEOUT_SEC,
    }
    top_p = _optional_float(sampling_params.get("top_p"))
    if top_p is not None:
        model_kwargs["top_p"] = top_p
    model_kwargs.update(extra_model_kwargs or {})
    model_arg = model_name if "/" in model_name else f"openai/{model_name}"
    common_kwargs = dict(
        model_name=model_arg,
        observation_template=observation_template,
        format_error_template=MINI_SWE_FORMAT_ERROR_TEMPLATE,
        cost_tracking="ignore_errors",
    )
    if thinking_budget >= 0:
        return _budget_forcing_model_cls()(
            thinking_budget=thinking_budget,
            force_phrase="I must now output an answer.</think>",
            max_tokens_total=max_tokens,
            model_kwargs=model_kwargs,
            **common_kwargs,
        )
    return LitellmTextbasedModel(model_kwargs=model_kwargs, **common_kwargs)


async def run_mini_swe_toolathlon_episode(
    *,
    base_url: str,
    model_name: str,
    metadata: dict[str, Any],
    sampling_params: dict[str, Any],
    extra_model_kwargs: dict[str, Any] | None = None,
    step_limit: int = MINI_SWE_STEP_LIMIT,
    max_new_tokens: int = ROLLOUT_RESPONSE_LEN,
    mini_swe_command_timeout_sec: int = 120,
    thinking_budget: int = MINI_SWE_THINKING_BUDGET,
    obs_char_limit: int = MINI_SWE_OBSERVATION_CHAR_LIMIT,
) -> dict[str, Any]:
    from minisweagent.agents.default import DefaultAgent  # type: ignore[import-not-found]

    task_name = metadata.get("task_name")
    task_request = metadata.get("task_request") or metadata.get("prompt") or ""
    if not task_name:
        return {
            "reward": 0.0,
            "exit_status": ToolathlonExitStatus.MISSING_TASK_NAME,
            "trajectory_messages": [],
        }

    episode_started = time.time()
    print(
        "[toolathlon] episode start "
        f"task={task_name!r} timeout={EPISODE_TIMEOUT_SEC}s "
        f"step_limit={step_limit} max_new_tokens={max_new_tokens} "
        f"thinking_budget={thinking_budget} obs_char_limit={obs_char_limit}",
        flush=True,
    )

    acquire_started = time.time()
    try:
        env = await asyncio.to_thread(_create_fresh_toolathlon_env, task_name)
    except Exception as exc:  # noqa: BLE001
        return {
            "reward": 0.0,
            "exit_status": f"acquire: {_exception_summary(exc)}",
            "trajectory_messages": [],
        }
    acquire_latency = time.time() - acquire_started
    print(
        "[toolathlon] sandbox acquired "
        f"task={task_name!r} latency={acquire_latency:.2f}s",
        flush=True,
    )

    mini_env = _MiniSweSandboxEnvironment(env, timeout=mini_swe_command_timeout_sec)
    model = _mini_swe_model(
        base_url=base_url,
        model_name=model_name,
        sampling_params=sampling_params,
        max_new_tokens=max_new_tokens,
        extra_model_kwargs=extra_model_kwargs,
        thinking_budget=thinking_budget,
        observation_template=_build_observation_template(obs_char_limit),
    )
    agent = DefaultAgent(
        model,
        mini_env,
        system_template=MINI_SWE_SYSTEM_TEMPLATE,
        instance_template=MINI_SWE_INSTANCE_TEMPLATE,
        step_limit=step_limit,
        cost_limit=0.0,
        wall_time_limit_seconds=EPISODE_TIMEOUT_SEC,
        output_path=None,
    )

    started = time.time()
    run_error: Exception | None = None
    exit_status: str = ToolathlonExitStatus.SUBMITTED
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mini-swe")
    agent_started = time.time()
    try:
        print(
            "[toolathlon] agent run start "
            f"task={task_name!r} request_chars={len(str(task_request))}",
            flush=True,
        )
        result_extra = await loop.run_in_executor(executor, agent.run, task_request)
        exit_status = _normalize_exit_status(
            str(
                (result_extra or {}).get("exit_status")
                or ToolathlonExitStatus.SUBMITTED
            )
        )
    except Exception as exc:  # noqa: BLE001
        run_error = exc
        exit_status = type(exc).__name__
        print(
            "[toolathlon] agent run exception "
            f"task={task_name!r} status={exit_status} error={_exception_summary(exc)}",
            flush=True,
        )
    finally:
        executor.shutdown(wait=False)
    agent_run_latency = time.time() - agent_started
    print(
        "[toolathlon] agent run end "
        f"task={task_name!r} status={exit_status} "
        f"latency={agent_run_latency:.2f}s",
        flush=True,
    )

    try:
        trajectory = agent.serialize()
    except Exception:
        trajectory = {}
    trajectory_messages = _trajectory_messages(trajectory)
    response_text = _last_assistant_text(trajectory_messages)

    evaluate_started = time.time()
    # Why the task was graded the way it was: the evaluator's human-readable
    # failure detail (or the grading exception). Surfaced on the dashboard so a
    # failed rollout shows *why* it failed, not just a bare "fail" status.
    eval_detail = ""
    try:
        print(f"[toolathlon] evaluate start task={task_name!r}", flush=True)
        verdict = await asyncio.to_thread(env.evaluate)
    except Exception as exc:  # noqa: BLE001
        verdict = None
        eval_detail = _exception_summary(exc)
        exit_status = f"evaluate: {eval_detail}"
        print(
            "[toolathlon] evaluate exception "
            f"task={task_name!r} error={_exception_summary(exc)}",
            flush=True,
        )
    finally:
        evaluate_latency = time.time() - evaluate_started
        await asyncio.to_thread(env.close)
        print(
            "[toolathlon] sandbox released "
            f"task={task_name!r} evaluate_latency={evaluate_latency:.2f}s",
            flush=True,
        )

    if verdict is None:
        reward = 0.0
    else:
        reward = 1.0 if verdict.passed else 0.0
        if verdict.detail:
            eval_detail = verdict.detail
        if exit_status == ToolathlonExitStatus.SUBMITTED:
            exit_status = (
                ToolathlonExitStatus.SUCCESS
                if verdict.passed
                else ToolathlonExitStatus.FAIL
            )
        elif verdict.harness_error:
            exit_status = ToolathlonExitStatus.HARNESS_ERROR

    tool_execution_latency = sum(mini_env.tool_execution_latencies)
    print(
        "[toolathlon] episode end "
        f"task={task_name!r} status={exit_status} reward={reward:.1f} "
        f"total_latency={time.time() - episode_started:.2f}s "
        f"agent_latency={agent_run_latency:.2f}s "
        f"tool_calls={len(mini_env.tool_execution_latencies)} "
        f"tool_latency={tool_execution_latency:.2f}s",
        flush=True,
    )
    return {
        "reward": reward,
        "exit_status": exit_status,
        "eval_detail": eval_detail,
        "trajectory_messages": trajectory_messages,
        "response_text": response_text,
        "model_latency": time.time() - started,
        "acquire_latency": acquire_latency,
        "agent_run_latency": agent_run_latency,
        "evaluate_latency": evaluate_latency,
        "tool_execution_latency": tool_execution_latency,
        "tool_execution_latency_mean": (
            tool_execution_latency / len(mini_env.tool_execution_latencies)
            if mini_env.tool_execution_latencies
            else 0.0
        ),
        "tool_execution_count": len(mini_env.tool_execution_latencies),
        "mini_swe_exit_code": 0 if run_error is None else -1,
        "mini_swe_exit_status": exit_status,
        "mini_swe_trajectory": trajectory,
    }


# ─────────────────────────────────────────────────────────────────────────────
# slime hooks: rollout (toolathlon_generate) + reward (reward_func)
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=2)
def _tokenizer(hf_checkpoint: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)


def _sample_metadata(sample) -> dict[str, Any]:
    metadata = dict(getattr(sample, "metadata", {}) or {})
    label = getattr(sample, "label", None)
    if isinstance(label, str):
        try:
            label = json.loads(label)
        except json.JSONDecodeError:
            label = None
    if isinstance(label, dict):
        metadata = {**label, **metadata}
    return metadata


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(
            str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in content
        )
    return str(content) if content else ""


def _training_segments(
    trajectory: list[dict[str, Any]],
) -> tuple[list[tuple[str, int]], list[str]]:
    segments: list[tuple[str, int]] = []
    assistant_turns: list[str] = []
    saw_assistant = False
    for message in trajectory:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        content = _message_text(message)
        if not content:
            continue
        if role == "assistant":
            if segments:
                segments.append(("\n\n", 0))
            segments.append((content, 1))
            assistant_turns.append(content)
            saw_assistant = True
        elif saw_assistant:
            content = NEXT_RESPONSE_CONTRACT_RE.sub("\n", content).strip()
            if not content:
                continue
            label = {
                "tool": "OBSERVATION",
                "user": "OBSERVATION",
                "system": "SYSTEM",
                "developer": "DEVELOPER",
            }.get(role, role.upper() or "OBSERVATION")
            segments.append((f"\n\n{label}:\n{content}", 0))
    return segments, assistant_turns


async def toolathlon_generate(
    args,
    sample,
    sampling_params,
    evaluation: bool = False,
    step_limit: int = MINI_SWE_STEP_LIMIT,
    max_new_tokens: int = ROLLOUT_RESPONSE_LEN,
    mini_swe_command_timeout_sec: int = 120,
):
    """Run a live Toolathlon episode and pack it into a slime Sample."""
    from slime.utils.types import Sample  # type: ignore[import-not-found]

    del evaluation

    metadata = _sample_metadata(sample)

    try:
        payload = await asyncio.wait_for(
            run_mini_swe_toolathlon_episode(
                base_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
                model_name="model",
                metadata=metadata,
                sampling_params=sampling_params,
                step_limit=step_limit,
                max_new_tokens=max_new_tokens,
                mini_swe_command_timeout_sec=mini_swe_command_timeout_sec,
            ),
            timeout=EPISODE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        print(
            "[toolathlon] episode timeout "
            f"task={metadata.get('task_name')!r} timeout={EPISODE_TIMEOUT_SEC}s",
            flush=True,
        )
        payload = {
            "reward": 0.0,
            "exit_status": ToolathlonExitStatus.TIMEOUT,
            "trajectory_messages": [],
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(
            "[toolathlon] episode exception "
            f"task={metadata.get('task_name')!r} error={_exception_summary(exc)}",
            flush=True,
        )
        payload = {
            "reward": 0.0,
            "exit_status": f"exception: {_exception_summary(exc)}",
            "trajectory_messages": [],
        }

    tokenizer = _tokenizer(args.hf_checkpoint)
    prompt_text = (
        sample.prompt
        if isinstance(sample.prompt, str)
        else json.dumps(sample.prompt, sort_keys=True)
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    response_tokens: list[int] = []
    loss_mask: list[int] = []
    trajectory_messages = payload.get("trajectory_messages", [])
    if not isinstance(trajectory_messages, list):
        trajectory_messages = []
    segments, assistant_turns = _training_segments(trajectory_messages)
    for text, trainable in segments:
        ids = tokenizer.encode(text, add_special_tokens=False)
        response_tokens.extend(ids)
        loss_mask.extend([trainable] * len(ids))

    if not response_tokens:
        eos = getattr(tokenizer, "eos_token_id", None) or 0
        response_tokens, loss_mask = [eos], [0]

    # Megatron context-parallel packing needs (prompt + response) divisible by 2*CP.
    cp = getattr(args, "context_parallel_size", 1) or 1
    max_seq = getattr(args, "max_tokens_per_gpu", 8192) * cp
    raw_response_length = len(response_tokens)
    response_budget = max(1, max_seq - len(prompt_tokens))
    response_tokens = response_tokens[:response_budget]
    loss_mask = loss_mask[: len(response_tokens)]
    align = 2 * cp
    remainder = (len(prompt_tokens) + len(response_tokens)) % align
    if remainder:
        eos = getattr(tokenizer, "eos_token_id", None) or 0
        if len(prompt_tokens) + len(response_tokens) + (align - remainder) <= max_seq:
            response_tokens += [eos] * (align - remainder)
            loss_mask += [0] * (align - remainder)
        else:
            response_tokens = response_tokens[: len(response_tokens) - remainder]
            loss_mask = loss_mask[: len(loss_mask) - remainder]

    sample.response = "\n\n".join(assistant_turns)
    sample.prompt_length = len(prompt_tokens)
    sample.response_length = len(response_tokens)
    sample.tokens = prompt_tokens + response_tokens
    sample.loss_mask = loss_mask
    # Apply the length penalty HERE, not in reward_func. toolathlon_generate sets
    # sample.reward + status=COMPLETED, so slime's rm step (generate_and_rm) treats
    # the reward as already computed and SKIPS custom_rm_function — a penalty applied
    # in reward_func never reaches GRPO. sample.reward is the value GRPO optimizes.
    # Penalize on the model's OWN generated tokens (loss_mask=1), not total length.
    assistant_tokens = sum(loss_mask)
    task_reward = float(payload.get("reward", 0.0))
    sample.reward = _length_penalized_reward(task_reward, assistant_tokens)
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        **metadata,
        **payload,
        # Raw 0/1 task reward (true pass/fail); sample.reward carries the penalty.
        "reward": task_reward,
        "task_reward": task_reward,
        "penalized_reward": sample.reward,
        "training_assistant_tokens": assistant_tokens,
        "exit_status": payload.get("exit_status"),
        "training_response_source": "assistant_action_turns",
        "training_assistant_turns": len(assistant_turns),
        "trajectory_message_count": len(trajectory_messages),
        "training_token_limit": max_seq,
        "training_raw_response_length": raw_response_length,
        "training_tokens_truncated": raw_response_length > len(response_tokens),
    }
    return sample


def _length_penalized_reward(reward: float, assistant_tokens: int) -> float:
    """Subtract a verbosity cost that grows with the model's own generated
    (assistant) token count — applied to passes AND fails so long trajectories
    are discouraged regardless of task success.

    Cost is 0 up to ``LENGTH_PENALTY_FREE_TOKENS``, then grows linearly to
    ``LENGTH_PENALTY_MAX_COST`` at ``LENGTH_PENALTY_MAX_TOKENS`` (clamped beyond).
    ``assistant_tokens`` is ``sum(loss_mask)`` — the trainable turns the policy
    controls, not total ``response_length`` (dominated by tool observations).
    """
    if assistant_tokens <= LENGTH_PENALTY_FREE_TOKENS:
        return reward
    span = LENGTH_PENALTY_MAX_TOKENS - LENGTH_PENALTY_FREE_TOKENS
    frac = min(1.0, (assistant_tokens - LENGTH_PENALTY_FREE_TOKENS) / span)
    return reward - LENGTH_PENALTY_MAX_COST * frac


async def reward_func(args, samples, **kwargs):
    """Pass through ``sample.reward`` (already length-penalized in
    toolathlon_generate); slime computes the GRPO advantage.

    NOTE: for toolathlon this is effectively never called — toolathlon_generate
    sets ``sample.reward`` + ``status=COMPLETED``, so slime's ``generate_and_rm``
    skips ``custom_rm_function`` for samples that already have a reward. Reward
    shaping MUST happen in toolathlon_generate, not here.
    """
    if isinstance(samples, (list, tuple)):
        return [float(getattr(s, "reward", 0.0) or 0.0) for s in samples]
    return float(getattr(samples, "reward", 0.0) or 0.0)


def toolathlon_rollout_log(
    rollout_id, args, samples, rollout_extra_metrics, rollout_time
) -> bool:
    del args, rollout_time

    latencies = []
    task_rewards = []  # raw 0/1 task reward -> true pass rate
    penalized_rewards = []  # length-penalized reward GRPO actually optimizes
    response_lengths = []
    assistant_tokens_list = []  # model's own generated tokens (penalty target)
    exit_status_counts: dict[str, int] = {}
    truncated_count = 0
    for sample in samples:
        penalized_rewards.append(float(getattr(sample, "reward", 0.0) or 0.0))
        rl = getattr(sample, "response_length", None)
        if isinstance(rl, (int, float)):
            response_lengths.append(float(rl))
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        task_rewards.append(
            float(metadata.get("task_reward", metadata.get("reward", 0.0)) or 0.0)
        )
        at = metadata.get("training_assistant_tokens")
        if isinstance(at, (int, float)):
            assistant_tokens_list.append(float(at))
        exit_status = str(metadata.get("exit_status") or "unknown")
        exit_status_counts[exit_status] = exit_status_counts.get(exit_status, 0) + 1
        if metadata.get("training_tokens_truncated"):
            truncated_count += 1
        try:
            latencies.append(float(metadata["tool_execution_latency"]))
        except (KeyError, TypeError, ValueError):
            continue

    if not latencies and not penalized_rewards:
        return False

    metrics: dict[str, float] = {}
    if task_rewards:
        metrics["toolathlon_pass_rate"] = sum(task_rewards) / len(task_rewards)
    if penalized_rewards:
        metrics["toolathlon_mean_penalized_reward"] = sum(penalized_rewards) / len(
            penalized_rewards
        )
        metrics["toolathlon_truncation_rate"] = truncated_count / len(penalized_rewards)
    if response_lengths:
        metrics["toolathlon_response_length_mean"] = sum(response_lengths) / len(
            response_lengths
        )
    if assistant_tokens_list:
        metrics["toolathlon_assistant_tokens_mean"] = sum(assistant_tokens_list) / len(
            assistant_tokens_list
        )
    if latencies:
        metrics["tool_execution_latency"] = sum(latencies) / len(latencies)
    for status, count in exit_status_counts.items():
        metrics[f"toolathlon_exit_status/{status}"] = count

    if isinstance(rollout_extra_metrics, dict):
        rollout_extra_metrics.update(metrics)

    try:
        wandb = __import__("wandb")
    except ImportError:
        return False

    if getattr(wandb, "run", None) is not None:
        wandb.log(
            metrics,
            step=rollout_id,
            commit=False,
        )
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Eval: same live Toolathlon grading as training, against a deployed endpoint
# ─────────────────────────────────────────────────────────────────────────────


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    label = row.get("label")
    if isinstance(label, str):
        try:
            label = json.loads(label)
        except json.JSONDecodeError:
            label = None
    if isinstance(label, dict):
        metadata = {**label, **metadata}
    return metadata


def _eval_transcript(trajectory: list[dict[str, Any]]) -> str:
    parts = []
    for message in trajectory:
        role = str(message.get("role", "")).upper()
        content = message.get("content")
        if content:
            parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def make_toolathlon_eval_fn(
    *,
    sampling_params: dict[str, Any] | None = None,
    step_limit: int = MINI_SWE_STEP_LIMIT,
    max_new_tokens: int = ROLLOUT_RESPONSE_LEN,
    mini_swe_command_timeout_sec: int = 120,
    thinking_budget: int = MINI_SWE_THINKING_BUDGET,
    obs_char_limit: int = MINI_SWE_OBSERVATION_CHAR_LIMIT,
):
    """Build the single Toolathlon grader — an ``EvalConfig``-compatible
    ``eval_fn(deployment, example) -> EvalRowResult``.

    This is the one place the eval is defined. It wraps the same
    :func:`run_mini_swe_toolathlon_episode` scorer the training rollout
    (:func:`toolathlon_generate`) uses, so a task graded during a standalone
    ``eval`` run and the same task graded by the in-training slime eval go
    through identical live Mini-SWE episodes. Pass a callable produced here (or
    your own with the same signature) to :class:`EvalConfig` to override it in
    one place.
    """
    sampling = sampling_params or {
        "temperature": TEMPERATURE,
        "top_p": 1.0,
        "max_new_tokens": max_new_tokens,
    }

    def eval_fn(deployment, example):
        # Imported lazily (not at module import) so rollout workers that import
        # this module without modal_training_gym installed still load — see the
        # ImportError guard at the top of the file.
        from modal_training_gym.common.deployment import _modal_proxy_auth_headers
        from modal_training_gym.common.eval import EvalRowResult

        metadata = _row_metadata(example)

        async def _run():
            return await asyncio.wait_for(
                run_mini_swe_toolathlon_episode(
                    base_url=deployment.url,
                    model_name=deployment.deployment_config.served_model_name,
                    metadata=metadata,
                    sampling_params=sampling,
                    extra_model_kwargs={"extra_headers": _modal_proxy_auth_headers()},
                    step_limit=step_limit,
                    max_new_tokens=max_new_tokens,
                    mini_swe_command_timeout_sec=mini_swe_command_timeout_sec,
                    thinking_budget=thinking_budget,
                    obs_char_limit=obs_char_limit,
                ),
                timeout=EPISODE_TIMEOUT_SEC,
            )

        try:
            payload = asyncio.run(_run())
        except asyncio.TimeoutError:
            payload = {
                "reward": 0.0,
                "exit_status": ToolathlonExitStatus.TIMEOUT,
                "trajectory_messages": [],
            }
        return EvalRowResult(
            score=float(payload.get("reward", 0.0)),
            response=_eval_transcript(payload.get("trajectory_messages", [])),
            prompt=str(metadata.get("task_request") or metadata.get("prompt") or ""),
            metadata={
                "task_name": metadata.get("task_name"),
                "task_type": metadata.get("task_type"),
                "needed_mcp_servers": metadata.get("needed_mcp_servers"),
                "exit_status": payload.get("exit_status"),
                # Why it failed: the evaluator's failure detail (empty on pass).
                "eval_detail": payload.get("eval_detail"),
                # Structured multi-turn trajectory so the dashboard can render the
                # full conversation via ConversationView (the flat `response` is
                # kept as a fallback). Only ~15 eval rows, so no cap needed.
                "trajectory_messages": payload.get("trajectory_messages", []),
                **{k: metadata[k] for k in HARBOR_META_FIELDS if k in metadata},
            },
        )

    return eval_fn


#: The default shared grader, usable directly as an ``EvalConfig.eval_fn``.
toolathlon_eval_fn = make_toolathlon_eval_fn()


def _resolve_eval_fn(eval_fn_path: str):
    """Resolve a ``module.attr`` dotted path to an eval callable, or return the
    built-in :data:`toolathlon_eval_fn` when *eval_fn_path* is empty."""
    if not eval_fn_path:
        return toolathlon_eval_fn
    import importlib

    module_path, _, attr = eval_fn_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"eval_fn_path {eval_fn_path!r} must be a dotted 'module.attr' path"
        )
    fn = getattr(importlib.import_module(module_path), attr)
    if not callable(fn):
        raise TypeError(f"eval_fn_path {eval_fn_path!r} did not resolve to a callable")
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Recipe / launch
# ─────────────────────────────────────────────────────────────────────────────


def _image_overlay(image):
    return image.run_commands(
        "uv pip install --system 'modal>=1.4.0'",
        "uv pip install --system git+https://github.com/BerriAI/litellm.git --no-deps",
        "uv pip install --system 'mini-swe-agent==2.3.0'",
    )


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class ToolathlonQwen3_6_35bRecipe(SlimeRecipe):
    """Qwen3.6-35B-A3B (MoE) on 5x8xH200 with TP1/PP2/CP4/EP4."""

    # -- Infra flags ───────────────────────────────────────────────────────
    gpu_type: str = "H200"
    slime_model_script: str = "scripts/models/qwen3.5-35B-A3B.sh"
    hf_checkpoint: str = "Qwen/Qwen3.6-35B-A3B"
    async_mode: bool = True
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )

    colocate: bool = False
    # 1 actor node (8 GPUs)
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    # 4 rollout nodes (8 GPUs per node)
    rollout_num_gpus: int = 32

    # ── Rollout ───────────────────────────────────────────────────────────
    num_rollout: int = 40
    rollout_batch_size: int = 8
    # tp_size = engine_number = rollout_num_gpus / rollout_num_gpus_per_engine
    rollout_num_gpus_per_engine: int = 8
    rollout_max_response_len: int = ROLLOUT_RESPONSE_LEN
    rollout_max_context_len: int = ROLLOUT_CONTEXT_LEN
    rollout_temperature: float = 1.0
    rollout_stop_token_ids: list[int] | None = field(
        default_factory=lambda: list(QWEN3_TOOL_STOP_TOKEN_IDS)
    )
    global_batch_size: int = 64
    sglang_mem_fraction_static: float = 0.75
    sglang_ep_size: int | None = 4
    sglang_cuda_graph_bs: list[int] | None = field(
        default_factory=lambda: [1, 2, 4, 8] + list(range(16, 257, 8))
    )

    # EAGLE/MTP speculative decoding disabled: the base Qwen3.6-35B-A3B ships no
    # MTP weights, and converting a randomly-initialized MTP head at any tp/pp>1
    # corrupts the torch_dist save (duplicate keys in determine_global_metadata) —
    # the same MTP/checkpoint incompatibility that forced GLM-4.7 to disable it.
    sglang_speculative_algorithm: str | None = None
    sglang_speculative_num_steps: int | None = None
    sglang_speculative_eagle_topk: int | None = None
    sglang_speculative_num_draft_tokens: int | None = None
    sglang_mamba_scheduler_strategy: str = "extra_buffer"
    mtp_num_layers: int | None = None
    enable_mtp_training: bool = False
    mtp_loss_scaling_factor: float | None = None
    sglang_max_running_requests: int | None = 256

    # Long context specific flags that are commonly set ---------------------
    # HiCache: tier the radix prefix cache GPU→host so idle multi-turn prefixes survive
    # eviction (keeps ~0.8 prefix-hit under load). ratio=1.0: host pool = GPU pool — 3.0
    # host-OOMs on H200 (its GPU KV pool is ~2× H100's).
    sglang_enable_hierarchical_cache = True
    sglang_hicache_ratio = 1.0
    sglang_hicache_write_policy = "write_through"
    sglang_page_size = 64  # HiCache transfers are page-granular

    # -- Here are recipes that can be tuned ────────────────────────────────
    # ── Parallelism ───────────────────────────────────────────────────────
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 2
    context_parallel_size: int = 4
    expert_model_parallel_size: int = 4
    expert_tensor_parallel_size: int = 1

    # ── Training ──────────────────────────────────────────────────────────
    n_samples_per_prompt: int = 8
    lr: float = 1e-6
    max_tokens_per_gpu: int = MAX_TOKENS_PER_GPU
    calculate_per_token_loss: bool = True
    moe_token_dispatcher_type: str = "flex"
    moe_enable_deepep: bool = True

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # ── Attention ─────────────────────────────────────────────────────────
    attention_backend: str = "flash"
    no_save_optim: bool = True
    no_load_optim: bool = True
    over_sampling_batch_size: int | None = 48
    # Bounded variant of slime's check_reward_nonzero_std: slime's
    # dynamic-sampling loop has no cap, so an all-equal-reward step (e.g. the
    # policy fails every task) would oversample forever and wedge the run. The
    # bounded filter force-accepts after a streak of consecutive rejections.
    dynamic_sampling_filter_path: str | None = (
        "modal_training_gym.train_recipes.slime_recipe.dynamic_sampling.check_reward_nonzero_std_bounded"
    )

    # ── Checkpointing / eval ──────────────────────────────────────────────
    megatron_to_hf_mode: str = ""
    ref_load: str = "/checkpoints/Qwen3.6-35B-A3B_torch_dist_toolathlon"
    save_interval: int = 5
    # In-training eval every 5 rollouts (aligned with save_interval). slime's
    # eval reuses the rollout `custom_generate_function` (toolathlon_generate)
    # with evaluation=True against the eval split, so the grading path is the
    # same live Mini-SWE episode the standalone `eval` entrypoint runs — see
    # make_toolathlon_eval_fn / run_mini_swe_toolathlon_episode.
    eval_interval: int | None = 5

    # ── Chat template ─────────────────────────────────────────────────────
    apply_chat_template_kwargs: dict | str = field(
        default_factory=lambda: {"enable_thinking": True}
    )
    extra_config: dict | None = field(
        default_factory=lambda: {
            "rl_parallel_generation_tasks": 64,
        }
    )

    # ── Environment ───────────────────────────────────────────────────────
    environment: dict = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        }
    )

    # -- Custom functions ───────────────────────────────────────────────────
    custom_generate_function: Any = toolathlon_generate
    custom_rm_function: Any = reward_func
    image_overlay: Any = _image_overlay
    # The single shared grader: an EvalConfig-compatible
    # ``eval_fn(deployment, example) -> EvalRowResult`` read by the standalone
    # ``eval`` entrypoint. Defined once here so standalone eval and the
    # in-training slime eval grade with the same live Mini-SWE episode — the
    # training eval reuses the rollout generate (``toolathlon_generate``), which
    # shares the same ``run_mini_swe_toolathlon_episode`` scorer. Not a slime CLI
    # flag (see ``_SLIME_SKIP``). Override with your own same-signature callable
    # (or build one via ``make_toolathlon_eval_fn``).
    eval_fn: Any = toolathlon_eval_fn


@app.local_entrypoint()
def train(experiment_name: str, eval_interval: int = 5) -> None:
    try:
        modal.Secret.from_name("huggingface-secret").hydrate()
    except modal.exception.NotFoundError as exc:
        raise RuntimeError(
            "Missing Modal Secret 'huggingface-secret' (needs HF_TOKEN)."
        ) from exc

    # In-training eval grades the eval split with the same live Mini-SWE episode
    # as the standalone `eval` entrypoint: slime reuses the rollout
    # `custom_generate_function` (toolathlon_generate → run_mini_swe_toolathlon_episode)
    # with evaluation=True. Pass --eval-interval 0 to disable.
    recipe = ToolathlonQwen3_6_35bRecipe(
        wandb=WandbConfig(project="toolathlon", group="qwen3.6-35b-a3b"),
        custom_rollout_log_function=toolathlon_rollout_log,
        eval_interval=eval_interval or None,
    )
    bad_rollout_shape = (
        recipe.global_batch_size > 64
        or recipe.n_samples_per_prompt > 8
        or recipe.rollout_batch_size > 8
    )
    if bad_rollout_shape:
        raise RuntimeError(
            "Toolathlon live rollout config is too large: "
            f"global_batch_size={recipe.global_batch_size}, "
            f"n_samples_per_prompt={recipe.n_samples_per_prompt}, "
            f"rollout_batch_size={recipe.rollout_batch_size}. "
            "Expected <=32, <=8, <=4 for live Mini-SWE rollouts."
        )
    print(
        "Toolathlon effective rollout config: "
        f"global_batch_size={recipe.global_batch_size}, "
        f"n_samples_per_prompt={recipe.n_samples_per_prompt}, "
        f"rollout_batch_size={recipe.rollout_batch_size}, "
        f"rollout_max_response_len={recipe.rollout_max_response_len}, "
        f"rollout_max_context_len={recipe.rollout_max_context_len}, "
        f"max_tokens_per_gpu={recipe.max_tokens_per_gpu}, "
        f"over_sampling_batch_size={recipe.over_sampling_batch_size}, "
        f"dynamic_sampling_filter_path={recipe.dynamic_sampling_filter_path}, "
        f"async_mode={recipe.async_mode}, "
        f"eval_interval={recipe.eval_interval}, "
        f"extra_config={recipe.extra_config}",
        flush=True,
    )

    group = TrainingGroup(
        name=experiment_name,
        base=TrainConfig(
            model=Qwen3_6_35B(),
            dataset=ToolathlonDataset(),
            recipe=recipe,
        ),
        merge_model_recipe=False,
        grid={
            "recipe.sglang_disable_custom_all_reduce": [False],
        },
    )

    print(f"Launching toolathlon trainings: {group.get_train_configs()}")
    launches = group.launch()
    print(f"Launched {len(launches)} runs")
    for launch in launches:
        print(
            f"  {launch.training_run_id}  "
            f"(app_id={launch.modal_app_id}, call_id={launch.function_call_id})"
        )
    if group.failures:
        for overrides, err in group.failures:
            print(f"  FAILED {overrides}: {_exception_summary(err)}")
    return None


@app.local_entrypoint()
def eval(
    training_run_id: str = "",
    max_concurrency: int = 4,
    step_limit: int = MINI_SWE_STEP_LIMIT,
    max_new_tokens: int = ROLLOUT_RESPONSE_LEN,
    mini_swe_command_timeout_sec: int = 120,
    thinking_budget: int = MINI_SWE_THINKING_BUDGET,
    obs_char_limit: int = MINI_SWE_OBSERVATION_CHAR_LIMIT,
    eval_fn_path: str = "",
) -> None:
    """Evaluate a deployed model on the Tier-A Toolathlon tasks with the same
    live-grading reward used in training: each task runs a full agent episode
    against a fresh sandbox, scored by ``env.evaluate()`` (1.0 pass / 0.0 fail).

    Serves the base ``Qwen3.6-35B`` by default; pass ``--training-run-id`` to
    serve and evaluate the latest checkpoint from a training run instead.

    The grader is carried by ``ToolathlonQwen3_6_35bRecipe.eval_fn`` — the same
    episode scorer the in-training slime eval uses — so it is defined once.
    Pass ``--eval-fn-path module.attr`` to plug in your own
    ``eval_fn(deployment, example)`` for this run.
    """
    from modal_training_gym import DeploymentConfig, list_checkpoints
    from modal_training_gym.common.eval import EvalConfig
    from modal_training_gym.deploy_recipes.sglang_recipe import SglangRecipe

    def _status(state: str, detail: str = "") -> None:
        print(f"\n=== [eval] {state}{f' — {detail}' if detail else ''} ===", flush=True)

    model = Qwen3_6_35B()
    recipe = SglangRecipe(
        gpu="H200", tp=1, extra_server_args={"--trust-remote-code": ""}
    )
    checkpoint = list_checkpoints(training_run_id)[-1] if training_run_id else None

    _status(
        "DEPLOYING MODEL",
        checkpoint.path if checkpoint is not None else model.model_name,
    )
    try:
        deployment = DeploymentConfig(
            model=model,
            checkpoint=checkpoint,
            recipe=recipe,
            app_name="toolathlon-serve",
            served_model_name="toolathlon",
        ).serve()
    except Exception as exc:  # noqa: BLE001
        _status("FAILED", f"deploy: {exc}")
        raise
    print(f"Deployed to {deployment.url}")

    # The grader is carried by the recipe (ToolathlonQwen3_6_35bRecipe.eval_fn),
    # so training and standalone eval share one definition. Precedence:
    # explicit --eval-fn-path > recipe's eval_fn. When the recipe still holds the
    # stock toolathlon_eval_fn, rebuild it so this run's tuning knobs (step_limit,
    # max_new_tokens, …) apply; a custom recipe grader is used verbatim.
    recipe_eval_fn = ToolathlonQwen3_6_35bRecipe.__dataclass_fields__["eval_fn"].default
    if eval_fn_path:
        eval_fn = _resolve_eval_fn(eval_fn_path)
    elif recipe_eval_fn is toolathlon_eval_fn:
        eval_fn = make_toolathlon_eval_fn(
            sampling_params={
                "temperature": TEMPERATURE,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
            },
            step_limit=step_limit,
            max_new_tokens=max_new_tokens,
            mini_swe_command_timeout_sec=mini_swe_command_timeout_sec,
            thinking_budget=thinking_budget,
            obs_char_limit=obs_char_limit,
        )
    else:
        eval_fn = recipe_eval_fn

    eval_config = EvalConfig(
        dataset=ToolathlonDataset(),
        eval_fn=eval_fn,
        prompt_column="prompt",
    )

    _status("RUNNING EVAL")
    try:
        result = eval_config.evaluate(
            deployment, debug=True, max_concurrency=max_concurrency
        )
    except Exception as exc:  # noqa: BLE001
        _status("FAILED", f"eval: {exc}")
        raise
    _status("SUCCESS", f"pass rate {result.mean:.3f} over {result.total} tasks")
