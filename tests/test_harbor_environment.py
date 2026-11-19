import asyncio
import base64
import io
import json
import shlex
import tarfile
import threading
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from modal_dojo import Environment, HarborEnvironment
from modal_dojo.common import harbor_environment as he
from modal_dojo.common import harbor_harnesses as hh
from modal_dojo.common import harbor_sandbox as hs
from modal_dojo.common.harbor_harnesses import BashHarness, MiniSweAgent, Terminus2
from modal_dojo.common.errors import DojoConfigError


@pytest.fixture(autouse=True)
def local_task_dirs(monkeypatch):
    """Let tests point HarborEnvironment at a temporary directory of tasks."""
    parse, fetch = he._parse_source, he._fetch_source
    monkeypatch.setattr(
        he,
        "_parse_source",
        lambda s: {"kind": "local", "path": s} if s and Path(s).is_dir() else parse(s),
    )
    monkeypatch.setattr(
        he,
        "_fetch_source",
        lambda spec: Path(spec["path"]) if spec["kind"] == "local" else fetch(spec),
    )


def _make_task(
    root: Path, name: str, *, toml: str = "", dockerfile: bool = True
) -> Path:
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "instruction.md").write_text(f"Create /app/{name}.txt\n")
    (task / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    if dockerfile:
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "task.toml").write_text(toml)
    return task


def _label(env: HarborEnvironment, index: int = 0) -> dict:
    return json.loads(list(env.dataset.rows())[index]["label"])


def test_dataset_rows_prompt_with_instruction_and_pack_task(tmp_path):
    _make_task(
        tmp_path,
        "b-task",
        toml=(
            "[verifier]\ntimeout_sec = 30.0\nenv = { JUDGE = 'x' }\n"
            "[agent]\ntimeout_sec = 45.0\n"
            "[environment]\ncpus = 2\nmemory_mb = 4096\n"
        ),
    )
    _make_task(tmp_path, "a-task")
    env = HarborEnvironment(str(tmp_path), harness=BashHarness(system_prompt="sys"))

    rows = list(env.dataset.rows())
    assert [r["messages"][1]["content"] for r in rows] == [
        "Create /app/a-task.txt",
        "Create /app/b-task.txt",
    ]
    assert rows[0]["messages"][0] == {"role": "system", "content": "sys"}

    label = _label(env, 1)
    assert label["task_name"] == "b-task"
    assert label["verifier_timeout_sec"] == 30.0
    assert label["agent_timeout_sec"] == 45.0
    with tarfile.open(
        fileobj=io.BytesIO(base64.b64decode(label["task_files"])), mode="r:gz"
    ) as tar:
        assert sorted(tar.getnames()) == [
            "environment",
            "environment/Dockerfile",
            "instruction.md",
            "task.toml",
            "tests",
            "tests/test.sh",
        ]


def test_dataset_is_stable_and_validates_columns(tmp_path):
    _make_task(tmp_path, "t")
    env = HarborEnvironment(str(tmp_path), harness=BashHarness())
    out = tmp_path / "out.jsonl"
    env.dataset.write(str(out))
    env.dataset.validate_written(str(out))
    assert (
        env.dataset.cache_key()
        == HarborEnvironment(str(tmp_path), harness=BashHarness()).dataset.cache_key()
    )


def test_task_selection(tmp_path):
    for name in ("a", "b", "c"):
        _make_task(tmp_path, name)
    assert [
        d.name
        for d in HarborEnvironment(
            str(tmp_path), max_tasks=2, harness=BashHarness()
        ).task_dirs()
    ] == ["a", "b"]
    assert [
        d.name
        for d in HarborEnvironment(
            str(tmp_path), task_names=["c"], harness=BashHarness()
        ).task_dirs()
    ] == ["c"]
    with pytest.raises(DojoConfigError, match="not found"):
        HarborEnvironment(
            str(tmp_path), task_names=["zzz"], harness=BashHarness()
        ).task_dirs()


@pytest.mark.parametrize(
    ("toml", "dockerfile", "match"),
    [
        ("[[steps]]\nname = 's1'\n", True, "multi-step"),
        ("[environment]\nos = 'windows'\n", True, "only linux"),
        ("", False, "docker-compose"),
    ],
)
def test_unsupported_tasks_are_rejected(tmp_path, toml, dockerfile, match):
    _make_task(tmp_path, "t", toml=toml, dockerfile=dockerfile)
    with pytest.raises(DojoConfigError, match=match):
        _ = HarborEnvironment(str(tmp_path), harness=BashHarness()).dataset


@pytest.mark.parametrize(
    ("source", "spec"),
    [
        ("hello-world", {"kind": "hub", "ref": "hello-world"}),
        ("terminal-bench@2.0", {"kind": "hub", "ref": "terminal-bench@2.0"}),
        (
            "scale-ai/swe-atlas-qna@1.0",
            {"kind": "hub", "ref": "scale-ai/swe-atlas-qna@1.0"},
        ),
        (
            "https://github.com/laude-institute/harbor",
            {"kind": "github", "owner": "laude-institute", "repo": "harbor"},
        ),
        (
            "github.com/o/r.git",
            {"kind": "github", "owner": "o", "repo": "r"},
        ),
        (
            "https://github.com/laude-institute/harbor/tree/main/examples/tasks/hello-world/",
            {
                "kind": "github",
                "owner": "laude-institute",
                "repo": "harbor",
                "ref": "main",
                "subdir": "examples/tasks/hello-world",
            },
        ),
    ],
)
def test_parse_source(source, spec):
    assert he._parse_source(source) == spec


@pytest.mark.parametrize(
    "source",
    [
        "/abs/tasks",
        "./tasks",
        "~/tasks",
        "../tasks",
        "file:///tmp/tasks",
        "https://gitlab.com/o/r",
        "https://github.com/o",
        "",
    ],
)
def test_local_and_unknown_sources_are_rejected(source):
    with pytest.raises(DojoConfigError, match="Local paths are not supported"):
        HarborEnvironment(source, harness=BashHarness())


def test_pinned_github_commit_is_cached(tmp_path, monkeypatch):
    sha = "a" * 40
    cached = tmp_path / "github" / "o" / "r" / sha / "tasks"
    cached.mkdir(parents=True)
    monkeypatch.setattr(he, "_CACHE_ROOT", tmp_path)

    def no_network(*args, **kwargs):
        raise AssertionError("pinned commit should come from the cache")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    spec = he._parse_source(f"https://github.com/o/r/tree/{sha}/tasks")
    assert he._fetch_source(spec) == cached


def test_reward_fn_ships_as_import_path():
    fn = HarborEnvironment("hello-world", harness=BashHarness()).reward_fn
    assert f"{fn.__module__}.{fn.__name__}" == (
        "modal_dojo.common.harbor_environment.harbor_reward"
    )


@pytest.mark.parametrize(
    ("response", "script"),
    [
        ("```bash\necho hi\n```", "echo hi\n"),
        ("```python\nx\n```\n```sh\nls\n```\n```python\ny\n```", "ls\n"),
        ("```\ntouch a\n```", "touch a\n"),
        ("<think>```bash\nbad\n```</think>```bash\ngood\n```", "good\n"),
        ("no code here", None),
    ],
)
def test_extract_script(response, script):
    assert hh.extract_script(response) == script


class _FakeEnvironment(BaseEnvironment):
    """Harbor's environment interface, running nothing.

    Records calls and serves canned verifier output.
    """

    instances: list["_FakeEnvironment"] = []
    reward_files: dict[str, str] = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[tuple] = []
        self.files: dict[str, str] = {}
        super().__init__(
            environment_dir=kwargs["environment_dir"],
            environment_name=kwargs["environment_name"],
            session_id=kwargs["session_id"],
            trial_paths=kwargs["trial_paths"],
            task_env_config=kwargs["task_env_config"],
        )
        type(self).instances.append(self)

    @staticmethod
    def type():
        return "fake"

    def _validate_definition(self):
        pass

    async def start(self, force_build):
        self.calls.append(("start",))

    async def stop(self, delete):
        self.calls.append(("stop",))

    async def upload_file(self, source_path, target_path):
        self.calls.append(("write", target_path))
        self.files[target_path] = Path(source_path).read_text()  # noqa: ASYNC240

    async def upload_dir(self, source_dir, target_dir):
        self.calls.append(("upload_dir", target_dir))

    async def download_file(self, source_path, target_path):
        raise FileNotFoundError(source_path)

    async def download_dir(self, source_dir, target_dir):
        self.calls.append(("download_dir", source_dir))
        Path(target_dir).mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
        for name, text in self.reward_files.items():
            (Path(target_dir) / name).write_text(text)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.calls.append(("run", command, env))
        return ExecResult(stdout="", stderr="", return_code=0)


@pytest.fixture
def fake_environment(monkeypatch):
    """Harbor's environment factory, returning a ``_FakeEnvironment``."""
    _FakeEnvironment.instances = []
    _FakeEnvironment.reward_files = {}
    monkeypatch.setattr(
        hs.EnvironmentFactory,
        "create_environment_from_config",
        staticmethod(lambda **kwargs: _FakeEnvironment(**kwargs)),
    )
    return _FakeEnvironment


def _sample(env: HarborEnvironment, response: str):
    return SimpleNamespace(
        label=list(env.dataset.rows())[0]["label"], response=response
    )


def test_reward_runs_answer_before_tests_and_reads_reward(
    tmp_path, fake_environment, monkeypatch
):
    monkeypatch.setenv("HARBOR_TEST_TOKEN", "tok")
    _make_task(
        tmp_path,
        "t",
        toml="[verifier]\nenv = { TOKEN = '${HARBOR_TEST_TOKEN}' }\n",
    )
    env = HarborEnvironment(str(tmp_path), harness=BashHarness())
    fake_environment.reward_files = {"reward.txt": "0.75\n"}

    reward = asyncio.run(
        env.reward_fn(None, _sample(env, "```bash\ntouch /app/t.txt\n```"))
    )

    assert reward == 0.75
    (environment,) = fake_environment.instances
    assert environment.kwargs["config"].type == "modal"  # Harbor's ModalEnvironment
    assert environment.files[f"{hh.SCRATCH_DIR}/step.sh"] == "touch /app/t.txt\n"
    solve_at = next(
        i
        for i, c in enumerate(environment.calls)
        if c[0] == "run" and "step.sh" in c[1]
    )
    assert solve_at < environment.calls.index(("upload_dir", "/tests"))
    _, test_cmd, test_env = next(
        c for c in environment.calls if c[0] == "run" and "(/tests/test.sh)" in c[1]
    )
    assert test_cmd == "(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1"
    assert test_env == {"TOKEN": "tok"}
    assert environment.calls[-1] == ("stop",)


def test_reward_is_zero_without_script_or_reward(tmp_path, fake_environment):
    _make_task(tmp_path, "t")
    env = HarborEnvironment(str(tmp_path), harness=BashHarness())

    assert asyncio.run(env.reward_fn(None, _sample(env, "I can't"))) == 0.0
    assert fake_environment.instances == []

    assert asyncio.run(env.reward_fn(None, _sample(env, "```bash\ntrue\n```"))) == 0.0
    assert len(fake_environment.instances) == 1


# ── Multi-turn ────────────────────────────────────────────────────────────

_EOT = "¶"  # the fake tokenizer's end-of-turn special token


class _FakeTokenizer:
    """One token per character; chat turns render as ``<role>content¶\\n``."""

    all_special_ids = [ord(_EOT)]

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, enable_thinking=True
    ):
        assert tokenize is False
        text = "".join(f"<{m['role']}>{m['content']}{_EOT}\n" for m in messages)
        if add_generation_prompt:
            text += "<assistant>" + ("" if enable_thinking else "<nothink>")
        return text

    def encode(self, text, add_special_tokens):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class _FakeSample:
    """The ``Sample`` fields Miles and slime share (no slime-only helpers)."""

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    def __init__(self, prompt, label):
        self.prompt = prompt
        self.label = label
        self.tokens = []
        self.response = ""
        self.response_length = 0
        self.loss_mask = None
        self.rollout_log_probs = None
        self.reward = None
        self.status = self.Status.ABORTED  # a retried sample
        self.metadata = {}


def _generation(text, finish="stop"):
    return {
        "text": text,
        "meta_info": {
            "output_token_logprobs": [[-0.5, ord(c)] for c in text],
            "finish_reason": {"type": finish},
        },
    }


@pytest.fixture
def multi_turn(tmp_path, monkeypatch, fake_environment):
    _make_task(tmp_path, "t")
    env = HarborEnvironment(
        str(tmp_path), max_turns=3, harness=BashHarness(max_output_chars=20)
    )
    row = list(env.dataset.rows())[0]
    posts: list[dict] = []
    replies: list[dict] = []

    async def fake_post(url, payload):
        posts.append(payload)
        return replies.pop(0)

    monkeypatch.setattr(he, "_post", fake_post)
    monkeypatch.setattr(he, "_tokenizer", lambda path: _FakeTokenizer())
    args = SimpleNamespace(
        hf_checkpoint="/m",
        sglang_router_ip="r",
        sglang_router_port=1,
        rollout_max_response_len=1000,
        apply_chat_template_kwargs='{"enable_thinking": false}',
    )
    return SimpleNamespace(
        env=env,
        row=row,
        posts=posts,
        replies=replies,
        args=args,
        sandbox=fake_environment,
    )


def _run(mt, sample=None):
    sample = sample or _FakeSample(mt.row["messages"], mt.row["label"])
    params = {"temperature": 1.0, "max_new_tokens": 500}
    return asyncio.run(mt.env.generate_fn(mt.args, sample, params))


def test_generate_fn_only_for_multi_turn(tmp_path):
    assert HarborEnvironment("x", harness=BashHarness()).generate_fn is None
    for harness, max_turns in [
        (BashHarness(), 2),
        (MiniSweAgent(), None),
        (Terminus2(), 1),
    ]:
        fn = HarborEnvironment("x", harness=harness, max_turns=max_turns).generate_fn
        assert fn.func is he.harbor_generate
        assert fn.keywords == {"harness": harness}


def test_multi_turn_episode_masks_observations_and_scores_in_sandbox(multi_turn):
    mt = multi_turn
    mt.sandbox.reward_files = {"reward.txt": "1"}
    turn1 = "```bash\nls\n```" + _EOT
    turn2 = "Done." + _EOT
    mt.replies.extend([_generation(turn1), _generation(turn2)])

    sample = _run(mt)

    assert sample.reward == 1.0
    assert sample.status is _FakeSample.Status.COMPLETED
    prompt = mt.row["messages"]
    prompt_text = _FakeTokenizer().apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    observation = "\n<user>Exit code: 0\nOutput:\n" + _EOT + "\n<assistant><nothink>"
    assert sample.response == turn1 + observation + turn2
    assert _FakeTokenizer().decode(sample.tokens) == prompt_text + sample.response
    assert sample.loss_mask == [1] * len(turn1) + [0] * len(observation) + [1] * len(
        turn2
    )
    assert sample.rollout_log_probs[len(turn1)] == 0.0
    assert len(sample.rollout_log_probs) == sample.response_length
    # The second request continues from exactly the tokens built so far.
    assert mt.posts[1]["input_ids"] == sample.tokens[: -len(turn2)]
    assert mt.posts[1]["sampling_params"]["max_new_tokens"] == 500
    # One sandbox for the whole episode: command, then tests.
    (sb,) = mt.sandbox.instances
    assert sample.metadata["harbor_exit_codes"] == [0]
    assert sb.calls.index(("write", f"{hh.SCRATCH_DIR}/step.sh")) < sb.calls.index(
        ("upload_dir", "/tests")
    )


def test_multi_turn_stops_at_max_turns_without_trailing_observation(multi_turn):
    mt = multi_turn
    mt.replies.extend([_generation("```bash\nls\n```" + _EOT) for _ in range(3)])

    sample = _run(mt)

    assert len(mt.posts) == 3
    assert sample.metadata["harbor_exit_codes"] == [0, 0, 0]
    assert sample.loss_mask[-1] == 1  # ends on the model's turn, not output
    assert sample.reward == 0.0  # the fake verifier wrote nothing


def test_multi_turn_truncates_long_output(multi_turn, monkeypatch):
    mt = multi_turn
    monkeypatch.setattr(
        hh, "run_shell", lambda sb, task, command, **kw: (3, "x" * 100, False)
    )
    mt.replies.extend(
        [_generation("```bash\nls\n```" + _EOT), _generation("ok" + _EOT)]
    )

    sample = _run(mt)

    assert "Exit code: 3\nOutput:\nxxxxxxxxxx\n[... 80 characters omitted ...]\n" in (
        sample.response
    )


def test_multi_turn_length_and_abort(multi_turn):
    mt = multi_turn
    mt.replies.append(_generation("```bash\nls", finish="length"))
    sample = _run(mt)
    assert sample.status is _FakeSample.Status.TRUNCATED
    assert sample.reward == 0.0  # still verified: the tests ran
    assert len(mt.sandbox.instances) == 1

    mt.replies.append(_generation("partial", finish="abort"))
    sample = _run(mt)
    assert sample.status is _FakeSample.Status.ABORTED
    assert sample.reward is None  # the framework retries it


def test_multi_turn_respects_response_token_budget(multi_turn):
    mt = multi_turn
    mt.args.rollout_max_response_len = 30
    turn1 = "```bash\nls\n```" + _EOT  # 15 tokens, then a 29-token observation
    mt.replies.append(_generation(turn1))

    sample = _run(mt)

    assert len(mt.posts) == 1
    assert sample.status is _FakeSample.Status.TRUNCATED


def test_multi_turn_sandbox_failure_scores_zero(multi_turn, monkeypatch):
    mt = multi_turn

    async def boom(self, force_build):
        raise RuntimeError("no capacity")

    monkeypatch.setattr(_FakeEnvironment, "start", boom)
    sample = _run(mt)
    assert sample.reward == 0.0
    assert sample.status is _FakeSample.Status.COMPLETED
    assert mt.posts == []


def test_reward_fn_rejects_multi_turn_rows(multi_turn):
    sample = SimpleNamespace(label=multi_turn.row["label"], response="```bash\nls\n```")
    with pytest.raises(DojoConfigError, match="generate_fn"):
        asyncio.run(multi_turn.env.reward_fn(None, sample))


class _AlternatingTokenizer(_FakeTokenizer):
    """Rejects two user turns in a row, like strict-alternation templates."""

    def apply_chat_template(self, messages, **kwargs):
        roles = [m["role"] for m in messages]
        if any(a == b for a, b in zip(roles, roles[1:])):
            raise ValueError("roles must alternate")
        return super().apply_chat_template(messages, **kwargs)


class _RerenderingTokenizer(_FakeTokenizer):
    """Renders the final assistant turn differently, like Qwen3.5's template."""

    def apply_chat_template(self, messages, **kwargs):
        text = super().apply_chat_template(messages, **kwargs)
        if messages[-1]["role"] == "assistant":
            text = text.replace("<assistant>", "<assistant><think></think>")
        return text


@pytest.mark.parametrize(
    "tokenizer", [_FakeTokenizer(), _AlternatingTokenizer(), _RerenderingTokenizer()]
)
def test_observation_tokens_close_turn_and_open_next(tokenizer):
    tokens = he.observation_tokens(
        tokenizer, "out", last_token=ord(_EOT), template_kwargs={}
    )
    assert tokenizer.decode(tokens) == "\n<user>out¶\n<assistant>"
    no_eot = he.observation_tokens(
        tokenizer, "out", last_token=ord("x"), template_kwargs={}
    )
    assert tokenizer.decode(no_eot) == "<user>out¶\n<assistant>"


# ── ManagedTrainConfig ────────────────────────────────────────────────────


def test_managed_config_trains_on_a_harbor_environment(tmp_path):
    from modal_dojo import Budget, ManagedTrainConfig, Qwen3_5_4B

    _make_task(tmp_path, "t")
    env = HarborEnvironment(str(tmp_path), harness=MiniSweAgent())
    assert isinstance(env, Environment)

    recipe = ManagedTrainConfig(Qwen3_5_4B(), env, Budget(max_rollouts=7)).recipe

    assert recipe.num_rollout == 7
    assert recipe.custom_rm_function is env.reward_fn
    assert recipe.custom_generate_function.func is he.harbor_generate
    assert recipe.custom_generate_function.keywords == {"harness": env.harness}
    assert recipe.rollout_batch_size == 1  # one task


# ── Harnesses ─────────────────────────────────────────────────────────────


def test_harness_is_validated():
    with pytest.raises(DojoConfigError, match="AgentHarness instance"):
        HarborEnvironment("x", harness="claude-code")
    with pytest.raises(DojoConfigError, match="AgentHarness instance"):
        HarborEnvironment("x", harness=MiniSweAgent)
    assert HarborEnvironment("x", harness=BashHarness()).max_turns == 1
    assert HarborEnvironment("x", harness=MiniSweAgent()).max_turns == 50
    assert HarborEnvironment("x", harness=Terminus2()).max_turns == 50


def test_dataset_rows_preview_each_harness(tmp_path):
    _make_task(tmp_path, "t")
    multi = list(
        HarborEnvironment(
            str(tmp_path), harness=BashHarness(), max_turns=2
        ).dataset.rows()
    )[0]["messages"]
    assert "one step at a time" in multi[0]["content"]

    mini = list(
        HarborEnvironment(str(tmp_path), harness=MiniSweAgent()).dataset.rows()
    )[0]["messages"]
    assert [m["role"] for m in mini] == ["system", "user"]
    assert "```mswea_bash_command" in mini[0]["content"]
    assert mini[1]["content"].startswith("Please solve this issue: Create /app/t.txt")

    (terminus,) = list(
        HarborEnvironment(str(tmp_path), harness=Terminus2()).dataset.rows()
    )[0]["messages"]
    assert terminus["role"] == "user"
    assert "Task Description:\nCreate /app/t.txt" in terminus["content"]
    assert '"keystrokes": "ls -la\\n"' in terminus["content"]  # {{ }} unescaped


def test_reward_fn_rejects_agent_harness_rows(tmp_path):
    _make_task(tmp_path, "t")
    env = HarborEnvironment(str(tmp_path), harness=MiniSweAgent(), max_turns=1)
    row = list(env.dataset.rows())[0]
    sample = SimpleNamespace(label=row["label"], response="```bash\nls\n```")
    with pytest.raises(DojoConfigError, match="mini-swe-agent"):
        asyncio.run(env.reward_fn(None, sample))


def _task(harness, **extra):
    return {
        "task_name": "t",
        "harness": harness.name,
        "max_turns": 50,
        "env": {},
        **extra,
    }


class _ShellRecorder:
    """Stands in for run_shell; returns canned results in order."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, sb, task, command, **kwargs):
        self.calls.append((command, kwargs))
        return self.results.pop(0)


def _mini_episode(monkeypatch, *results):
    shell = _ShellRecorder(*results)
    monkeypatch.setattr(hh, "run_shell", shell)
    harness = MiniSweAgent()
    episode = harness.start_episode(None, _task(harness), deadline=1e18)
    return episode, shell


def _mini_reply(command):
    return f"THOUGHT: go.\n\n```mswea_bash_command\n{command}\n```"


def test_mini_swe_agent_runs_one_action_and_formats_output(monkeypatch):
    episode, shell = _mini_episode(monkeypatch, (0, "a\nb\n", False))
    observation = episode.step(_mini_reply("ls"))
    assert observation == "<returncode>0</returncode>\n<output>\na\nb\n</output>"
    ((command, kwargs),) = shell.calls
    assert command == "ls"
    assert kwargs["shell"] == "sh"
    assert kwargs["timeout_sec"] == 30
    assert kwargs["env"]["PAGER"] == "cat"


def test_mini_swe_agent_long_output_and_timeout(monkeypatch):
    episode, _ = _mini_episode(monkeypatch, (0, "x" * 12000, False), (124, "", True))
    long = episode.step(_mini_reply("cat big"))
    assert "<elided_chars>\n2000 characters elided\n</elided_chars>" in long
    assert "<warning>\nThe output of your last command was too long." in long
    timed_out = episode.step(_mini_reply("sleep 99"))
    assert timed_out.startswith(
        "<exception>An error occurred while executing the command: Command "
        "'sleep 99' timed out after 30 seconds</exception>\n<returncode>-1</returncode>"
    )


def test_mini_swe_agent_format_errors_and_submit(monkeypatch):
    episode, shell = _mini_episode(
        monkeypatch, (0, "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", False)
    )
    error = episode.step("no action here")
    assert error.startswith(
        "Format error:\n\n<error>\nExpected exactly 1 action, found 0."
    )
    assert "found 0 actions" in error
    assert shell.calls == []
    assert (
        episode.step(_mini_reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")) is None
    )
    assert episode.metadata["exit_status"] == "Submitted"

    episode, _ = _mini_episode(monkeypatch)
    two = _mini_reply("ls") + _mini_reply("pwd")
    assert episode.step(two) is not None
    assert episode.step(two) is not None
    assert episode.step(two) is None  # third format error in a row
    assert episode.metadata["exit_status"] == "RepeatedFormatError"


class _FakeTmuxEnvironment(_FakeEnvironment):
    """A tmux session whose pane echoes whatever keys are sent."""

    buffer = "root@box:/app# "

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.calls.append(("run", command, env))
        if "capture-pane" in command:
            return ExecResult(stdout=self.buffer, stderr="", return_code=0)
        if "send-keys" in command:
            keys = shlex.split(command)[-1]
            self.buffer += keys.replace("\n", "\nroot@box:/app# ")
        return ExecResult(stdout="", stderr="", return_code=0)


@pytest.fixture
def tmux_sandbox(tmp_path):
    """An ``EnvironmentSandbox`` over a fake tmux pane, on a background loop."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    environment = _FakeTmuxEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="t",
        session_id="t",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
    )
    yield hs.EnvironmentSandbox(environment, loop)
    loop.call_soon_threadsafe(loop.stop)
    thread.join()


def _terminus(commands, complete=False):
    return json.dumps(
        {
            "analysis": "a",
            "plan": "p",
            "commands": [{"keystrokes": k, "duration": 0.1} for k in commands],
            "task_complete": complete,
        }
    )


def test_terminus_2_drives_tmux_and_shows_new_output(tmux_sandbox):
    harness = Terminus2()
    episode = harness.start_episode(tmux_sandbox, _task(harness), 1e18)

    (message,) = episode.initial_messages("Do it.")
    assert (
        "Task Description:\nDo it.\n\nCurrent terminal state:\n" in message["content"]
    )
    assert message["content"].endswith("Current Terminal Screen:\nroot@box:/app# \n")
    commands = [c[1] for c in tmux_sandbox.environment.calls if c[0] == "run"]
    start = next(c for c in commands if "tmux new-session" in c)
    assert "tmux new-session -x 160 -y 40 -d -s terminus-2 'bash --login'" in start

    observation = episode.step(_terminus(["ls\n"]))
    assert observation.startswith("New Terminal Output:\n")
    assert "ls\nroot@box:/app# " in observation
    commands = [c[1] for c in tmux_sandbox.environment.calls if c[0] == "run"]
    assert any("send-keys -t terminus-2 -- 'ls\n'" in c for c in commands)


def test_terminus_2_parse_errors_warnings_and_double_completion(tmux_sandbox):
    harness = Terminus2()
    episode = harness.start_episode(tmux_sandbox, _task(harness), 1e18)
    episode.initial_messages("Do it.")

    error = episode.step("not json")
    assert error == (
        "Previous response had parsing errors:\nERROR: No valid JSON found in "
        "response\nWARNINGS: - No valid JSON object found\n\n"
        "Please fix these issues and provide a proper JSON response."
    )
    warned = episode.step('Sure! {"analysis": "", "plan": "", "commands": []}')
    assert warned.startswith(
        "Previous response had warnings:\nWARNINGS: - Extra text detected before "
        "JSON object\n\n"
    )
    confirm = episode.step(_terminus([], complete=True))
    assert "Are you sure you want to mark the task as complete?" in confirm
    assert episode.step(_terminus([], complete=True)) is None
    assert episode.metadata["exit_status"] == "TaskComplete"


def test_context_length_passes_through_and_caps_rollouts(tmp_path, multi_turn):
    from modal_dojo import Budget, ManagedTrainConfig, Qwen3_5_4B

    env = HarborEnvironment(
        str(tmp_path), harness=BashHarness(), max_turns=2, max_context_len=8192
    )
    assert env.max_context_len == 8192
    recipe = ManagedTrainConfig(Qwen3_5_4B(), env, Budget(max_rollouts=1)).recipe
    assert recipe.rollout_max_response_len == 8191

    mt = multi_turn
    prompt = _FakeTokenizer().apply_chat_template(
        mt.row["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    mt.args.rollout_max_context_len = len(prompt) + 20
    mt.replies.append(_generation("Done." + _EOT))
    _run(mt)
    assert mt.posts[0]["sampling_params"]["max_new_tokens"] == 20
