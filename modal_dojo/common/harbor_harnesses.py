"""Agent harnesses: how a model talks to a Harbor task's container."""

from __future__ import annotations

import functools
import importlib.metadata
import os
import re
import shlex
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.errors import DojoConfigError


# TODO: add to the API reference along with HarborEnvironment.
@exclude_from_api_reference
class Harness(str, Enum):
    """How the model works in a Harbor task's container."""

    BASH = "bash"
    MINI_SWE_AGENT = "mini-swe-agent"
    TERMINUS_2 = "terminus-2"


SCRATCH_DIR = "/tmp/modal-dojo-harbor"

HARBOR_VERSION = "0.8.0"
MINI_SWE_AGENT_VERSION = "2.4.6"
# Installed by the ``harbor`` extra locally and by the launchers in workers.
HARNESS_PACKAGES = (
    f"harbor=={HARBOR_VERSION}",
    f"mini-swe-agent=={MINI_SWE_AGENT_VERSION}",
)


def require_package(distribution: str, version: str) -> None:
    """Fail unless the harness package is installed at the pinned version."""
    try:
        installed = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != version:
        found = "is not installed" if installed is None else f"is {installed}"
        raise DojoConfigError(
            f"HarborEnvironment needs {distribution}=={version}, but it {found}. "
            "Install the extra: pip install 'modal-dojo[harbor]'."
        )


_FENCE_RE = re.compile(r"```([\w+-]*)[ \t]*\n(.*?)```", re.DOTALL)
_SHELL_LANGS = frozenset({"bash", "sh", "shell", "zsh"})


def env_assignments(env: dict[str, str]) -> str:
    """``K=V ...`` shell prefix, resolving ``${VAR}`` templates the way Harbor does."""
    require_package("harbor", HARBOR_VERSION)
    from harbor.utils.env import resolve_env_vars

    for key in env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise DojoConfigError(f"Invalid environment variable name {key!r}")
    try:
        resolved = resolve_env_vars(env)
    except ValueError as exc:
        raise DojoConfigError(str(exc)) from exc
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in resolved.items())


def run_shell(
    sb: Any,
    task: dict[str, Any],
    command: str,
    *,
    timeout_sec: float,
    shell: str = "bash",
    env: dict[str, str] | None = None,
) -> tuple[int, str, bool]:
    """Run ``command`` with ``shell`` in the sandbox, stdout and stderr merged."""
    sb.write(f"{SCRATCH_DIR}/step.sh", command)
    limit = max(1, int(timeout_sec))
    prefix = env_assignments({**task.get("env", {}), **(env or {})})
    result = sb.run(
        "bash",
        "-c",
        f"{prefix} timeout -k 5 {limit} {shell} {SCRATCH_DIR}/step.sh 2>&1".strip(),
        timeout=limit + 30,
    )
    timed_out = result.returncode in (124, 137)
    return result.returncode, result.stdout + result.stderr, timed_out


class Episode(ABC):
    """One harness-driven episode in one sandbox."""

    name: ClassVar[Harness]
    default_max_turns: ClassVar[int]

    def __init__(self, sb: Any, task: dict[str, Any], deadline: float) -> None:
        self.sb = sb
        self.task = task
        self.deadline = deadline
        self.metadata: dict[str, Any] = {}

    def time_left(self) -> float:
        return self.deadline - time.monotonic()

    @classmethod
    @abstractmethod
    def preview_messages(cls, instruction: str, task: dict[str, Any]) -> list[dict]:
        """The opening messages, without a sandbox (dataset rows show these)."""

    def initial_messages(self, instruction: str) -> list[dict]:
        """The opening messages for this episode's sandbox."""
        return self.preview_messages(instruction, self.task)

    @abstractmethod
    def step(self, response: str) -> str | None:
        """Act on a model reply; return what the model sees next, or ``None``."""


SINGLE_TURN_SYSTEM_PROMPT = (
    "You are solving a task inside a Linux container. Reply with a single "
    "```bash code block containing the shell script that completes the task. "
    "The script runs once, non-interactively, from the container's working "
    "directory; its output is not shown to you."
)

MULTI_TURN_SYSTEM_PROMPT = (
    "You are solving a task inside a Linux container. Work one step at a time: "
    "each reply should contain exactly one ```bash code block, which runs "
    "non-interactively in the container; the next message shows its exit code "
    "and output. The shell state does not persist between blocks, but the "
    "filesystem does. When the task is complete, reply without a code block."
)


def extract_script(response: str) -> str | None:
    """The shell script in a model response: its last ``bash``/``sh`` block."""
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[1]
    blocks = _FENCE_RE.findall(response)
    if not blocks:
        return None
    shell = [body for lang, body in blocks if lang.lower() in _SHELL_LANGS]
    return (shell or [body for _, body in blocks])[-1]


def format_observation(exit_code: int, output: str, *, max_chars: int) -> str:
    """The message showing the model a command's result."""
    if len(output) > max_chars:
        half = max_chars // 2
        omitted = len(output) - 2 * half
        output = (
            f"{output[:half]}\n[... {omitted} characters omitted ...]\n{output[-half:]}"
        )
    return f"Exit code: {exit_code}\nOutput:\n{output}"


class BashEpisode(Episode):
    name = Harness.BASH
    default_max_turns = 1

    @classmethod
    def preview_messages(cls, instruction: str, task: dict[str, Any]) -> list[dict]:
        system = task.get("system_prompt") or (
            MULTI_TURN_SYSTEM_PROMPT
            if task["max_turns"] > 1
            else SINGLE_TURN_SYSTEM_PROMPT
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]

    def step(self, response: str) -> str | None:
        script = extract_script(response)
        if script is None or self.time_left() <= 0:
            return None
        exit_code, output, _ = run_shell(
            self.sb, self.task, script, timeout_sec=self.time_left()
        )
        self.metadata.setdefault("exit_codes", []).append(exit_code)
        return format_observation(
            exit_code, output, max_chars=self.task["max_output_chars"]
        )


def _render(template: str, **variables: Any) -> str:
    """Render a mini-swe-agent template the way mini-swe-agent does."""
    from jinja2 import StrictUndefined, Template

    return Template(template, undefined=StrictUndefined).render(**variables)


@dataclass(frozen=True)
class _MiniSweAgentConfig:
    system_template: str
    instance_template: str
    observation_template: str
    format_error_template: str
    env: dict[str, str]
    action_regex: str
    command_timeout_sec: int
    max_consecutive_format_errors: int


# mini-swe-agent's local environment ends the run when a command prints this.
_SUBMIT_COMMAND_OUTPUT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


@functools.cache
def _mini_swe_agent() -> _MiniSweAgentConfig:
    """``mini_textbased.yaml`` and the defaults it relies on, from the package."""
    require_package("mini-swe-agent", MINI_SWE_AGENT_VERSION)
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")  # skip its startup banner
    import yaml
    from minisweagent.agents.default import AgentConfig
    from minisweagent.config import builtin_config_dir
    from minisweagent.environments.local import LocalEnvironmentConfig
    from minisweagent.models.litellm_textbased_model import (
        LitellmTextbasedModelConfig,
    )

    config = yaml.safe_load((builtin_config_dir / "mini_textbased.yaml").read_text())
    agent, model = config["agent"], config["model"]

    def default(cls: Any, field: str) -> Any:
        return cls.model_fields[field].default

    return _MiniSweAgentConfig(
        system_template=agent["system_template"],
        instance_template=agent["instance_template"],
        observation_template=model["observation_template"],
        format_error_template=model["format_error_template"],
        env=config["environment"]["env"],
        action_regex=default(LitellmTextbasedModelConfig, "action_regex"),
        command_timeout_sec=default(LocalEnvironmentConfig, "timeout"),
        max_consecutive_format_errors=default(
            AgentConfig, "max_consecutive_format_errors"
        ),
    )


class MiniSweAgentEpisode(Episode):
    """mini-swe-agent's ``mini_textbased.yaml`` agent, turn for turn."""

    name = Harness.MINI_SWE_AGENT
    default_max_turns = 50

    @classmethod
    def _messages(cls, instruction: str, uname: dict[str, str]) -> list[dict]:
        mini = _mini_swe_agent()

        variables = {"task": instruction, **uname}
        return [
            {"role": "system", "content": _render(mini.system_template, **variables)},
            {"role": "user", "content": _render(mini.instance_template, **variables)},
        ]

    @classmethod
    def preview_messages(cls, instruction: str, task: dict[str, Any]) -> list[dict]:
        uname = {"system": "Linux", "release": "", "version": "", "machine": ""}
        return cls._messages(instruction, uname)

    def initial_messages(self, instruction: str) -> list[dict]:
        result = self.sb.run(
            "bash",
            "-c",
            'printf "%s\\n" "$(uname -s)" "$(uname -r)" "$(uname -v)" "$(uname -m)"',
        )
        values = (result.stdout.splitlines() + ["", "", "", ""])[:4]
        uname = dict(zip(("system", "release", "version", "machine"), values))
        return self._messages(instruction, uname)

    def step(self, response: str) -> str | None:
        mini = _mini_swe_agent()

        actions = [
            a.strip() for a in re.findall(mini.action_regex, response, re.DOTALL)
        ]
        if len(actions) != 1:
            errors = self.metadata.get("consecutive_format_errors", 0) + 1
            self.metadata["consecutive_format_errors"] = errors
            if errors >= mini.max_consecutive_format_errors:
                self.metadata["exit_status"] = "RepeatedFormatError"
                return None
            return _render(
                mini.format_error_template,
                actions=actions,
                error=f"Expected exactly 1 action, found {len(actions)}.",
            )
        self.metadata["consecutive_format_errors"] = 0
        if self.time_left() <= 0:
            self.metadata["exit_status"] = "TimeExceeded"
            return None

        command = actions[0]
        timeout = min(mini.command_timeout_sec, self.time_left())
        exit_code, output, timed_out = run_shell(
            self.sb,
            self.task,
            command,
            timeout_sec=timeout,
            shell="sh",
            env=mini.env,
        )
        exception_info = ""
        if timed_out:
            exit_code = -1
            exception_info = (
                "An error occurred while executing the command: Command "
                f"'{command}' timed out after {int(timeout)} seconds"
            )
        self.metadata.setdefault("exit_codes", []).append(exit_code)

        lines = output.lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == _SUBMIT_COMMAND_OUTPUT and exit_code == 0:
            self.metadata["exit_status"] = "Submitted"
            return None
        return _render(
            mini.observation_template,
            output={
                "output": output,
                "returncode": exit_code,
                "exception_info": exception_info,
            },
        )


_TMUX_SESSION = "terminus-2"
_TMUX_SEND_KEYS_MAX_COMMAND_LENGTH = 16_000
_MAX_OUTPUT_BYTES = 10_000
_MAX_DURATION_SEC = 60

TMUX_INSTALL_COMMAND = (
    "command -v tmux"
    " || (apt-get update && apt-get install -y --no-install-recommends tmux)"
    " || apk add --no-cache tmux bash"
    " || dnf install -y tmux"
    " || yum install -y tmux"
)


def limit_output_length(output: str, max_bytes: int = _MAX_OUTPUT_BYTES) -> str:
    """Terminus 2's ``_limit_output_length``: keep the head and tail bytes."""
    output_bytes = output.encode("utf-8")
    if len(output_bytes) <= max_bytes:
        return output
    portion_size = max_bytes // 2
    first_portion = output_bytes[:portion_size].decode("utf-8", errors="ignore")
    last_portion = output_bytes[-portion_size:].decode("utf-8", errors="ignore")
    omitted_bytes = (
        len(output_bytes)
        - len(first_portion.encode("utf-8"))
        - len(last_portion.encode("utf-8"))
    )
    return (
        f"{first_portion}\n[... output limited to {max_bytes} bytes; "
        f"{omitted_bytes} interior bytes omitted ...]\n{last_portion}"
    )


def tmux_send_keys_commands(session: str, keys: str) -> list[str]:
    """Terminus 2's ``_tmux_send_keys`` for one key string, split to fit tmux."""
    prefix = f"tmux send-keys -t {shlex.quote(session)} --"
    max_escaped = _TMUX_SEND_KEYS_MAX_COMMAND_LENGTH - len(prefix.encode()) - 1
    if len(shlex.quote(keys).encode()) <= max_escaped:
        return [f"{prefix} {shlex.quote(keys)}"]
    commands, remaining = [], keys
    while remaining:
        lo, hi, best = 1, len(remaining), 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if len(shlex.quote(remaining[:mid]).encode()) <= max_escaped:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        commands.append(f"{prefix} {shlex.quote(remaining[:best])}")
        remaining = remaining[best:]
    return commands


@functools.cache
def _terminus_2() -> tuple[str, type]:
    """Terminus 2's JSON prompt template and parser class, from Harbor."""
    require_package("harbor", HARBOR_VERSION)
    from harbor.agents.terminus_2 import terminus_2
    from harbor.agents.terminus_2.terminus_json_plain_parser import (
        TerminusJSONPlainParser,
    )

    templates = Path(terminus_2.__file__).parent / "templates"
    return (templates / "terminus-json-plain.txt").read_text(), TerminusJSONPlainParser


class Terminus2Episode(Episode):
    """Harbor's Terminus 2 agent (JSON parser), turn for turn."""

    name = Harness.TERMINUS_2
    default_max_turns = 50

    def __init__(self, sb: Any, task: dict[str, Any], deadline: float) -> None:
        super().__init__(sb, task, deadline)
        _, parser_cls = _terminus_2()
        self._parser = parser_cls()
        self._previous_buffer: str | None = None
        self._pending_completion = False

    @classmethod
    def _messages(cls, instruction: str, terminal_state: str) -> list[dict]:
        prompt_template, _ = _terminus_2()
        prompt = prompt_template.format(
            instruction=instruction, terminal_state=terminal_state
        )
        return [{"role": "user", "content": prompt}]

    @classmethod
    def preview_messages(cls, instruction: str, task: dict[str, Any]) -> list[dict]:
        return cls._messages(instruction, "Current Terminal Screen:\n")

    def _tmux(self, command: str) -> Any:
        return self.sb.run("bash", "-c", command)

    def _capture(self, entire: bool) -> str:
        extra = " -S -" if entire else ""
        return self._tmux(f"tmux capture-pane -p{extra} -t {_TMUX_SESSION}").stdout

    def _incremental_output(self) -> str:
        """Terminus 2's ``get_incremental_output``."""
        current = self._capture(entire=True)
        if self._previous_buffer is None:
            self._previous_buffer = current
            return f"Current Terminal Screen:\n{self._capture(entire=False)}"
        previous = self._previous_buffer.strip()
        new_content = None
        if previous in current:
            idx = current.index(previous)
            if "\n" in previous:
                idx = previous.rfind("\n")
            new_content = current[idx:]
        self._previous_buffer = current
        if new_content is not None and new_content.strip():
            return f"New Terminal Output:\n{new_content}"
        return f"Current Terminal Screen:\n{self._capture(entire=False)}"

    def initial_messages(self, instruction: str) -> list[dict]:
        env = "".join(
            f"-e {shlex.quote(f'{k}={v}')} "
            for k, v in self.task.get("env", {}).items()
        )
        started = self._tmux(
            "export TERM=xterm-256color && export SHELL=/bin/bash && "
            f"tmux new-session {env}-x 160 -y 40 -d -s {_TMUX_SESSION} 'bash --login'"
        )
        if started.returncode != 0:
            raise RuntimeError(f"Failed to start tmux session: {started.stderr}")
        self._tmux("tmux set-option -g history-limit 10000000")
        terminal_state = limit_output_length(self._incremental_output())
        return self._messages(instruction, terminal_state)

    def _send(self, keystrokes: str, duration: float) -> None:
        start = time.monotonic()
        for command in tmux_send_keys_commands(_TMUX_SESSION, keystrokes):
            sent = self._tmux(command)
            if sent.returncode != 0:
                raise RuntimeError(f"tmux send-keys failed: {sent.stderr}")
        wait = min(duration, self.time_left()) - (time.monotonic() - start)
        if wait > 0:
            time.sleep(wait)

    def step(self, response: str) -> str | None:
        result = self._parser.parse_response(response)
        feedback = ""
        if result.error:
            feedback += f"ERROR: {result.error}"
            if result.warning:
                feedback += f"\nWARNINGS: {result.warning}"
        elif result.warning:
            feedback += f"WARNINGS: {result.warning}"

        if "ERROR:" in feedback:
            return (
                f"Previous response had parsing errors:\n{feedback}\n\n"
                "Please fix these issues and provide a proper JSON response."
            )
        if self.time_left() <= 0:
            return None

        for command in result.commands:
            self._send(command.keystrokes, min(command.duration, _MAX_DURATION_SEC))
        terminal_output = limit_output_length(self._incremental_output())

        if result.is_task_complete:
            if self._pending_completion:
                self.metadata["exit_status"] = "TaskComplete"
                return None
            self._pending_completion = True
            return (
                f"Current terminal state:\n{terminal_output}\n\n"
                "Are you sure you want to mark the task as complete? "
                "This will trigger your solution to be graded and you won't be able to "
                'make any further corrections. If so, include "task_complete": true '
                "in your JSON response again."
            )
        self._pending_completion = False
        if "WARNINGS:" in feedback:
            return (
                f"Previous response had warnings:\n{feedback}\n\n"
                f"{limit_output_length(terminal_output)}"
            )
        return limit_output_length(terminal_output)


EPISODES: dict[Harness, type[Episode]] = {
    cls.name: cls for cls in (BashEpisode, MiniSweAgentEpisode, Terminus2Episode)
}


def parse_harness(harness: Harness | str) -> Harness:
    try:
        return Harness(harness)
    except ValueError:
        raise DojoConfigError(
            f"Unknown harness {harness!r}; choose one of {[h.value for h in Harness]}"
        ) from None


def episode_class(harness: Harness | str) -> type[Episode]:
    return EPISODES[parse_harness(harness)]
