"""Agent harnesses: how a model talks to a Harbor task's container."""

from __future__ import annotations

import copy
import functools
import importlib.metadata
import os
import re
import shlex
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.errors import DojoConfigError


SCRATCH_DIR = "/tmp/modal-dojo-harbor"

HARBOR_VERSION = "0.25.0"
MINI_SWE_AGENT_VERSION = "2.4.6"
# Installed by the ``harbor`` extra locally and by the launchers in workers.
HARNESS_PACKAGES = (
    f"harbor[modal]=={HARBOR_VERSION}",
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


# TODO: add to the API reference along with HarborEnvironment.
@exclude_from_api_reference
class AgentHarness(ABC):
    """How a model works in a Harbor task's container, one episode at a time.

    Configure one and pass it as ``HarborEnvironment(..., harness=MyHarness())``.
    Subclass it to add your own. Each episode runs on a fresh copy:

    1. ``setup`` attaches the episode's started sandbox (``self.sb``, which has
       ``run``, ``write``, ``read``, ``upload_dir`` and ``download_dir``, and
       ``self.sb.environment``, Harbor's own ``ModalEnvironment``).
    2. ``initial_messages`` builds the opening chat messages.
    3. After every model reply, ``step`` acts on it and returns the next user
       message, or ``None`` to end the episode.
    4. The task's tests then grade whatever the sandbox holds.

    ```python
    class OneCommand(AgentHarness):
        name = "one-command"
        default_max_turns = 1

        def preview_messages(self, instruction, task):
            return [{"role": "user", "content": f"{instruction}\nReply with one command."}]

        def step(self, response):
            self.sb.run("bash", "-c", response)
            return None
    ```

    The harness is pickled to the rollout workers, so keep its configuration
    picklable; per-episode state belongs in ``setup``.

    Attributes:
        name: Identifies the harness in rollout metadata and dataset labels.
        default_max_turns: Model replies per episode unless ``max_turns`` is set.
    """

    name: ClassVar[str]
    default_max_turns: ClassVar[int] = 50

    def start_episode(
        self, sb: Any, task: dict[str, Any], deadline: float
    ) -> AgentHarness:
        """A fresh copy of this harness, set up for one episode."""
        episode = copy.deepcopy(self)
        episode.setup(sb, task, deadline)
        return episode

    def setup(self, sb: Any, task: dict[str, Any], deadline: float) -> None:
        """Attach the episode's sandbox; override to add per-episode state."""
        self.sb = sb
        self.task = task
        self.deadline = deadline
        self.metadata: dict[str, Any] = {}

    def time_left(self) -> float:
        """Seconds until the task's agent timeout."""
        return self.deadline - time.monotonic()

    @abstractmethod
    def preview_messages(self, instruction: str, task: dict[str, Any]) -> list[dict]:
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


# TODO: add to the API reference along with HarborEnvironment.
@exclude_from_api_reference
class BashHarness(AgentHarness):
    """One fenced bash block per reply; the model sees its exit code and output.

    With ``max_turns=1`` (the default) the model answers with one script, and
    the framework's own generation and reward function run the episode.

    Attributes:
        system_prompt: System message; defaults to one for single or multi-turn.
        max_output_chars: Command output shown per turn.
    """

    name = "bash"
    default_max_turns = 1

    def __init__(
        self, *, system_prompt: str | None = None, max_output_chars: int = 4000
    ) -> None:
        self.system_prompt = system_prompt
        self.max_output_chars = max_output_chars

    def preview_messages(self, instruction: str, task: dict[str, Any]) -> list[dict]:
        system = self.system_prompt or (
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
        return format_observation(exit_code, output, max_chars=self.max_output_chars)


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


# TODO: add to the API reference along with HarborEnvironment.
@exclude_from_api_reference
class MiniSweAgent(AgentHarness):
    """mini-swe-agent's ``mini_textbased.yaml`` agent, turn for turn.

    Its prompts, action format, command timeout and submit signal come from
    the installed ``mini-swe-agent`` package.
    """

    name = "mini-swe-agent"
    default_max_turns = 50

    @staticmethod
    def _messages(instruction: str, uname: dict[str, str]) -> list[dict]:
        mini = _mini_swe_agent()

        variables = {"task": instruction, **uname}
        return [
            {"role": "system", "content": _render(mini.system_template, **variables)},
            {"role": "user", "content": _render(mini.instance_template, **variables)},
        ]

    def preview_messages(self, instruction: str, task: dict[str, Any]) -> list[dict]:
        uname = {"system": "Linux", "release": "", "version": "", "machine": ""}
        return self._messages(instruction, uname)

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


_TMUX_PANE = (160, 40)  # Terminus 2's default width and height
_MAX_DURATION_SEC = 60  # Terminus 2 caps each command's wait at a minute


@dataclass(frozen=True)
class _Terminus2:
    prompt_template: str
    parser_cls: type
    agent_cls: type
    tmux_session_cls: type
    pane_log_path: str


@functools.cache
def _terminus_2() -> _Terminus2:
    """Terminus 2's prompt, parser, agent class and tmux session, from Harbor."""
    require_package("harbor", HARBOR_VERSION)
    from harbor.agents.terminus_2 import terminus_2
    from harbor.agents.terminus_2.terminus_json_plain_parser import (
        TerminusJSONPlainParser,
    )
    from harbor.agents.terminus_2.tmux_session import TmuxSession
    from harbor.models.trial.paths import EnvironmentPaths

    templates = Path(terminus_2.__file__).parent / "templates"
    return _Terminus2(
        prompt_template=(templates / "terminus-json-plain.txt").read_text(),
        parser_cls=TerminusJSONPlainParser,
        agent_cls=terminus_2.Terminus2,
        tmux_session_cls=TmuxSession,
        pane_log_path=str(EnvironmentPaths.agent_dir / "terminus_2.pane"),
    )


# TODO: add to the API reference along with HarborEnvironment.
@exclude_from_api_reference
class Terminus2(AgentHarness):
    """Harbor's Terminus 2 agent (JSON parser), turn for turn.

    The terminal is Harbor's own ``TmuxSession`` on the episode's Harbor
    environment; as in Harbor's trials, it installs tmux when the image lacks
    it. The agent's loop is async and talks to an LLM itself, so its per-turn
    steps are reproduced here; the output limit and completion prompt are its
    own methods.
    """

    name = "terminus-2"
    default_max_turns = 50

    def setup(self, sb: Any, task: dict[str, Any], deadline: float) -> None:
        super().setup(sb, task, deadline)
        self._t2 = _terminus_2()
        self._parser = self._t2.parser_cls()
        self._session: Any = None
        self._pending_completion = False

    @staticmethod
    def _messages(instruction: str, terminal_state: str) -> list[dict]:
        prompt = _terminus_2().prompt_template.format(
            instruction=instruction, terminal_state=terminal_state
        )
        return [{"role": "user", "content": prompt}]

    def preview_messages(self, instruction: str, task: dict[str, Any]) -> list[dict]:
        return self._messages(instruction, "Current Terminal Screen:\n")

    def _limit(self, output: str) -> str:
        # Terminus 2's _limit_output_length doesn't use its instance.
        return self._t2.agent_cls._limit_output_length(None, output)

    def _confirm_completion(self, terminal_output: str) -> str:
        json_agent = SimpleNamespace(options=SimpleNamespace(parser_name="json"))
        return self._t2.agent_cls._get_completion_confirmation_message(
            json_agent, terminal_output
        )

    def _output(self) -> str:
        return self._limit(self.sb.call(self._session.get_incremental_output()))

    def initial_messages(self, instruction: str) -> list[dict]:
        pane_log = self._t2.pane_log_path
        self.sb.run("mkdir", "-p", str(Path(pane_log).parent))
        width, height = _TMUX_PANE
        self._session = self._t2.tmux_session_cls(
            session_name=self.name,
            environment=self.sb.environment,
            logging_path=pane_log,
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
            pane_width=width,
            pane_height=height,
        )
        self.sb.call(self._session.start())
        return self._messages(instruction, self._output())

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
            wait = max(0.0, min(command.duration, _MAX_DURATION_SEC, self.time_left()))
            self.sb.call(
                self._session.send_keys(
                    command.keystrokes, block=False, min_timeout_sec=wait
                )
            )
        terminal_output = self._output()

        if result.is_task_complete:
            if self._pending_completion:
                self.metadata["exit_status"] = "TaskComplete"
                return None
            self._pending_completion = True
            return self._confirm_completion(terminal_output)
        self._pending_completion = False
        if "WARNINGS:" in feedback:
            return (
                f"Previous response had warnings:\n{feedback}\n\n"
                f"{self._limit(terminal_output)}"
            )
        return self._limit(terminal_output)
