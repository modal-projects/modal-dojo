"""Harbor tasks as a training environment: a dataset plus a verifier-backed reward."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.dataset import DatasetConfig, DatasetRow
from modal_dojo.common.environment import Environment
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.harbor_harnesses import (
    HARBOR_VERSION,
    SCRATCH_DIR,
    AgentHarness,
    BashHarness,
    extract_script,
    require_package,
    run_shell,
)
from modal_dojo.common.sandbox import extract_tar_gz, tar_gz

logger = logging.getLogger(__name__)

# Harbor leaves ``agent.timeout_sec`` unset unless a task sets it.
_DEFAULT_AGENT_TIMEOUT_SEC = 600.0


def _tar_b64(directory: Path, *, exclude: tuple[str, ...] = ()) -> str:
    return base64.b64encode(tar_gz(directory, exclude=exclude)).decode("ascii")


def _find_task_dirs(root: Path) -> list[Path]:
    """Directories laid out as Harbor tasks; ``_build_row`` validates each one."""
    require_package("harbor", HARBOR_VERSION)
    from harbor.models.task.paths import TaskPaths

    def is_task(path: Path) -> bool:
        paths = TaskPaths(path)
        return paths.config_path.is_file() and paths.environment_dir.is_dir()

    if is_task(root):
        return [root]
    return sorted(
        p.parent for p in root.rglob(TaskPaths.CONFIG_FILENAME) if is_task(p.parent)
    )


_CACHE_ROOT = Path.home() / ".cache" / "modal-dojo" / "harbor"
_GITHUB_RE = re.compile(
    r"^(?:https://)?github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?"
    r"(?:/tree/(?P<ref>[^/]+)(?:/(?P<subdir>.+?))?)?/?$"
)
_HUB_RE = re.compile(r"^[A-Za-z0-9][\w.-]*(?:/[A-Za-z0-9][\w.-]*)?(?:@[\w.+-]+)?$")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _parse_source(source: str) -> dict[str, str]:
    """Where a Harbor dataset lives: the Harbor hub or a GitHub directory."""
    source = source.strip()
    if match := _GITHUB_RE.match(source):
        spec = {k: v for k, v in match.groupdict().items() if v}
        return {"kind": "github", **spec}
    if "://" not in source and _HUB_RE.match(source):
        return {"kind": "hub", "ref": source}
    raise DojoConfigError(
        f"Unsupported Harbor source {source!r}. Use a Harbor hub dataset "
        "('name' or 'name@version') or a GitHub URL "
        "('https://github.com/owner/repo/tree/<ref>/<dir>'). Local paths are "
        "not supported: publish the tasks to GitHub or the Harbor hub."
    )


def _fetch_source(spec: dict[str, str]) -> Path:
    """Download the tasks; only pinned versions are cached between runs."""
    if spec["kind"] == "github":
        return _fetch_github(spec)
    return _fetch_hub(spec["ref"])


def _fetch_hub(ref: str) -> Path:
    version = ref.partition("@")[2]
    if version and version not in ("latest", "head"):
        dest = _CACHE_ROOT / "hub" / re.sub(r"[^\w.-]+", "--", ref)
        if dest.is_dir():
            return dest
    else:  # a moving version is fetched fresh every time
        dest = Path(tempfile.mkdtemp(prefix="harbor-hub-")) / "tasks"
    require_package("harbor", HARBOR_VERSION)
    tmp = dest.with_name(dest.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "harbor.cli.main",
            "datasets",
            "download",
            ref,
            "--export",
            "--overwrite",
            "--output-dir",
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(dest)
    return dest


def _fetch_github(spec: dict[str, str]) -> Path:
    import urllib.error
    import urllib.request

    owner, repo = spec["owner"], spec["repo"]
    ref = spec.get("ref", "HEAD")
    if _COMMIT_RE.fullmatch(ref):
        dest = _CACHE_ROOT / "github" / owner / repo / ref
    else:  # a branch or tag can move, so it is fetched fresh every time
        dest = Path(tempfile.mkdtemp(prefix="harbor-github-")) / "repo"
    if not dest.is_dir():
        url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/{ref}"
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                archive = response.read()
        except urllib.error.HTTPError as exc:
            raise DojoConfigError(
                f"Could not download github.com/{owner}/{repo} at {ref!r} "
                f"(HTTP {exc.code}). The repository must be public."
            ) from exc
        tmp = Path(tempfile.mkdtemp(prefix="harbor-github-", dir=dest.parent.parent))
        extract_tar_gz(archive, tmp)
        (top,) = tmp.iterdir()  # GitHub archives hold one <repo>-<ref>/ folder
        dest.parent.mkdir(parents=True, exist_ok=True)
        top.replace(dest)
        shutil.rmtree(tmp, ignore_errors=True)
    root = dest / spec["subdir"] if "subdir" in spec else dest
    if not root.is_dir():
        raise DojoConfigError(
            f"{spec['subdir']!r} is not a directory in github.com/{owner}/{repo} "
            f"at {ref!r}"
        )
    return root


class _HarborTaskDataset(DatasetConfig):
    """Pre-built rows for a set of Harbor tasks. Built by ``HarborEnvironment``."""

    def __init__(self, rows: list[DatasetRow]) -> None:
        self._rows = rows
        digest = hashlib.sha256()
        for row in rows:
            digest.update(json.dumps(row, sort_keys=True).encode())
        self._cache_key = digest.hexdigest()

    def cache_key(self) -> str:
        return self._cache_key

    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> str:
        return "label"

    def rows(self) -> Iterable[DatasetRow]:
        return self._rows


# TODO: add to the API reference once the API shape is final.
@exclude_from_api_reference
class HarborEnvironment(Environment):
    """An ``Environment`` built from Harbor tasks, run through an agent harness.

    ```python
    run = ManagedTrainConfig(
        Qwen3_5_4B(),
        HarborEnvironment("swesmith@1.0", harness=MiniSweAgent()),
        Budget(max_cost_usd=100),
    ).train()
    ```

    Each episode runs in a sandbox built from the task's ``environment/``; the
    task's tests then set the reward, graded by Harbor's own ``Verifier``.

    Attributes:
        source: A Harbor hub dataset (``name[@version]``) or a public GitHub URL.
        harness: How the model works in the container: ``BashHarness()``,
            ``MiniSweAgent()``, ``Terminus2()``, or your own ``AgentHarness``.
        task_names: Only use these tasks.
        max_tasks: Use at most this many tasks.
        max_turns: Model replies per episode (default: the harness's
            ``default_max_turns``: 1 for bash, 50 for the others).
        reward_key: Key to read from ``reward.json``.
        max_context_len: Token limit for one episode; tasks don't set one.
    """

    def __init__(
        self,
        source: str,
        *,
        harness: AgentHarness,
        task_names: list[str] | None = None,
        max_tasks: int | None = None,
        max_turns: int | None = None,
        reward_key: str = "reward",
        max_context_len: int | None = None,
    ) -> None:
        self._source_spec = _parse_source(source)
        if not isinstance(harness, AgentHarness):
            raise DojoConfigError(
                f"harness must be an AgentHarness instance such as MiniSweAgent(), "
                f"got {harness!r}"
            )
        if max_turns is None:
            max_turns = harness.default_max_turns
        if max_turns < 1:
            raise DojoConfigError(f"max_turns must be at least 1, got {max_turns}")
        self.source = source
        self.harness = harness
        self.task_names = task_names
        self.max_tasks = max_tasks
        self.max_turns = max_turns
        self.reward_key = reward_key
        super().__init__(None, harbor_reward, max_context_len=max_context_len)
        self._dataset: _HarborTaskDataset | None = None

    @property
    def single_turn(self) -> bool:
        """Whether the framework's own generation and ``reward_fn`` suffice."""
        return type(self.harness) is BashHarness and self.max_turns == 1

    def task_dirs(self) -> list[Path]:
        """The Harbor task directories this environment trains on."""
        root = _fetch_source(self._source_spec)
        dirs = _find_task_dirs(root)
        if self.task_names is not None:
            wanted = set(self.task_names)
            dirs = [d for d in dirs if d.name in wanted]
            missing = wanted - {d.name for d in dirs}
            if missing:
                raise DojoConfigError(
                    f"Harbor tasks not found under {root}: {sorted(missing)}"
                )
        if self.max_tasks is not None:
            dirs = dirs[: self.max_tasks]
        if not dirs:
            raise DojoConfigError(f"No Harbor tasks found under {root}")
        return dirs

    def _build_row(self, task_dir: Path) -> DatasetRow:
        """The task's prompt plus everything a worker needs to run and verify it."""
        from harbor.models.task.config import TaskConfig, TaskOS
        from harbor.models.task.task import Task

        name = task_dir.name
        # Check what HarborEnvironment supports before Task() validates the rest.
        try:
            config = TaskConfig.model_validate_toml(
                (task_dir / "task.toml").read_text()
            )
        except ValueError as exc:
            raise DojoConfigError(f"Invalid task.toml in {name!r}: {exc}") from exc
        if config.steps:
            raise DojoConfigError(
                f"Harbor task {name!r} is multi-step; HarborEnvironment supports "
                "single-step tasks only."
            )
        if config.environment.os != TaskOS.LINUX:
            raise DojoConfigError(
                f"Harbor task {name!r} targets os={config.environment.os.value!r}; "
                "only linux tasks are supported."
            )
        try:
            task = Task(task_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise DojoConfigError(f"Invalid Harbor task {name!r}: {exc}") from exc
        env_cfg = task.config.environment
        if task.paths.discovered_test_path_for(env_cfg.os) is None:
            raise DojoConfigError(
                f"Harbor task {name!r} has no test script in tests/ (separate "
                "verifier environments are not supported)."
            )
        if not (
            env_cfg.docker_image
            or (task.paths.environment_dir / "Dockerfile").is_file()
        ):
            raise DojoConfigError(
                f"Harbor task {name!r} has neither environment.docker_image nor "
                "environment/Dockerfile (docker-compose environments are not "
                "supported)."
            )

        instruction = task.instruction.strip()
        label = {
            "task_name": name,
            "env": env_cfg.env,
            # The worker rebuilds Harbor's Task from these: Harbor builds the
            # image from environment/ and grades with tests/. The solution
            # never ships.
            "task_files": _tar_b64(task_dir, exclude=("solution",)),
            "verifier_timeout_sec": task.config.verifier.timeout_sec,
            "agent_timeout_sec": task.config.agent.timeout_sec
            or _DEFAULT_AGENT_TIMEOUT_SEC,
            "instruction": instruction,
            "harness": self.harness.name,
            "reward_key": self.reward_key,
            "max_turns": self.max_turns,
        }
        messages = self.harness.preview_messages(instruction, label)
        return {"messages": messages, "label": json.dumps(label)}

    @property
    def dataset(self) -> DatasetConfig:
        """One prompt per task; pass as ``TrainConfig.dataset``."""
        if self._dataset is None:
            self._dataset = _HarborTaskDataset(
                [self._build_row(d) for d in self.task_dirs()]
            )
        return self._dataset

    @property
    def reward_fn(self):
        """Verifier-backed reward; pass as the recipe's ``custom_rm_function``."""
        return harbor_reward

    @property
    def generate_fn(self):
        """Harness-driven rollout; pass as the recipe's ``custom_generate_function``."""
        if self.single_turn:
            return None
        # The harness travels with the function: launchers pickle functions
        # defined outside modal_dojo by value. A custom harness's module may
        # not exist in the image, so pickle its class by value too.
        module = sys.modules.get(type(self.harness).__module__)
        if module is not None and not module.__name__.startswith(
            ("__main__", "modal_dojo")
        ):
            import cloudpickle

            cloudpickle.register_pickle_by_value(module)
        return functools.partial(harbor_generate, harness=self.harness)


def _load_task(sample: Any) -> dict[str, Any]:
    label = sample.label
    return json.loads(label) if isinstance(label, str) else label


def _episode_environment(task: dict[str, Any]):
    """The task's started Harbor ``ModalEnvironment``, as an async context."""
    require_package("harbor", HARBOR_VERSION)
    from modal_dojo.common.harbor_sandbox import episode_environment

    return episode_environment(task, scratch_dir=SCRATCH_DIR)


async def _verify(task: dict[str, Any], script: str) -> float:
    """Run ``script`` then the task's tests in a fresh environment."""
    async with _episode_environment(task) as episode:
        await asyncio.to_thread(
            run_shell,
            episode.sandbox,
            task,
            script,
            timeout_sec=task["agent_timeout_sec"],
        )
        return await episode.verify(task["reward_key"])


async def harbor_reward(args: Any, sample: Any, **kwargs: Any) -> float:
    """Slime/Miles ``custom_rm`` for rows built by ``HarborEnvironment``."""
    task = _load_task(sample)
    if task["harness"] != BashHarness.name or task["max_turns"] > 1:
        raise DojoConfigError(
            f"Harbor task {task['task_name']!r} uses the {task['harness']!r} "
            f"harness for {task['max_turns']} turn(s); pass "
            "custom_generate_function=env.generate_fn to the recipe as well as "
            "custom_rm_function=env.reward_fn."
        )
    script = extract_script(sample.response or "")
    if script is None:
        return 0.0
    try:
        return await _verify(task, script)
    except Exception:
        logger.exception(
            "Harbor task %s: sandbox verification failed; scoring 0.0",
            task.get("task_name"),
        )
        return 0.0


@functools.cache
def _tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _template_kwargs(args: Any) -> dict[str, Any]:
    kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
    return json.loads(kwargs) if isinstance(kwargs, str) else dict(kwargs)


def _prompt_ids(tokenizer: Any, prompt: Any, template_kwargs: dict) -> list[int]:
    if not isinstance(prompt, str):
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
    return tokenizer.encode(prompt, add_special_tokens=False)


def observation_tokens(
    tokenizer: Any, text: str, *, last_token: int | None, template_kwargs: dict
) -> list[int]:
    """Tokens that append a user message after the model's last turn."""
    message = {"role": "user", "content": text}
    for base in (
        [{"role": "user", "content": "."}],
        [{"role": "user", "content": "."}, {"role": "assistant", "content": "."}],
    ):
        try:
            without = tokenizer.apply_chat_template(
                base, tokenize=False, add_generation_prompt=False, **template_kwargs
            )
            with_message = tokenizer.apply_chat_template(
                [*base, message],
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except Exception:  # e.g. a template that enforces role alternation
            continue
        if with_message.startswith(without):
            break
    else:
        raise RuntimeError(
            "chat template is not append-only, so a user message cannot be "
            "added after a generated turn"
        )
    delta = with_message[len(without) :]
    if last_token is not None and last_token in tokenizer.all_special_ids:
        end = tokenizer.decode([last_token])
        if end and end in without:
            delta = without[without.rindex(end) + len(end) :] + delta
    return tokenizer.encode(delta, add_special_tokens=False)


async def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST through the framework's own client (retries, Ray fan-out)."""
    try:
        from slime.utils.http_utils import post
    except ImportError:
        from miles.utils.http_utils import post
    return await post(url, payload)


def _append(
    sample: Any,
    args: Any,
    *,
    tokens: list[int],
    log_probs: list[float] | None,
    meta_info: dict | None,
    text: str,
) -> None:
    """Append response tokens; ``log_probs=None`` marks them untrained."""
    trainable = log_probs is not None
    if hasattr(sample, "append_response_tokens"):  # slime
        sample.append_response_tokens(
            args,
            tokens=tokens,
            log_probs=log_probs,
            trainable=trainable,
            meta_info=meta_info,
            text=text,
        )
        return
    sample.tokens = list(sample.tokens) + tokens  # miles
    sample.response += text
    sample.response_length += len(tokens)
    sample.loss_mask = (sample.loss_mask or []) + [int(trainable)] * len(tokens)
    sample.rollout_log_probs = (sample.rollout_log_probs or []) + (
        log_probs if trainable else [0.0] * len(tokens)
    )
    if meta_info is not None and hasattr(sample, "update_from_meta_info"):
        sample.update_from_meta_info(args, meta_info)


async def harbor_generate(
    args: Any,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
    *,
    harness: AgentHarness,
) -> Any:
    """Slime/Miles ``custom_generate_function`` for ``HarborEnvironment`` rows.

    ``HarborEnvironment.generate_fn`` binds ``harness``.
    """
    Status = type(sample).Status
    task = _load_task(sample)
    tokenizer = _tokenizer(args.hf_checkpoint)
    template_kwargs = _template_kwargs(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    budget = int(
        getattr(args, "rollout_max_response_len", None)
        or sampling_params["max_new_tokens"]
    )

    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = None
    sample.rollout_log_probs = None
    sample.reward = None
    sample.status = Status.PENDING

    environments = contextlib.AsyncExitStack()
    episode = None
    try:
        harbor_episode = await environments.enter_async_context(
            _episode_environment(task)
        )
        deadline = time.monotonic() + task["agent_timeout_sec"]
        episode = harness.start_episode(harbor_episode.sandbox, task, deadline)
        messages = await asyncio.to_thread(
            episode.initial_messages, task["instruction"]
        )
        sample.tokens = _prompt_ids(tokenizer, messages, template_kwargs)
        for turn in range(1, task["max_turns"] + 1):
            remaining = budget - sample.response_length
            context = getattr(args, "rollout_max_context_len", None)
            if context:
                remaining = min(remaining, context - len(sample.tokens))
            if remaining <= 0:
                sample.status = Status.TRUNCATED
                break
            output = await _post(
                url,
                {
                    "input_ids": sample.tokens,
                    "sampling_params": {
                        **sampling_params,
                        "max_new_tokens": min(
                            sampling_params["max_new_tokens"], remaining
                        ),
                    },
                    "return_logprob": True,
                },
            )
            meta_info = output["meta_info"]
            pairs = meta_info.get("output_token_logprobs") or []
            new_tokens = [pair[1] for pair in pairs]
            _append(
                sample,
                args,
                tokens=new_tokens,
                log_probs=[pair[0] for pair in pairs],
                meta_info=meta_info,
                text=output["text"],
            )
            finish = (meta_info.get("finish_reason") or {}).get("type")
            if finish == "abort":
                sample.status = Status.ABORTED
                return sample
            if finish == "length":
                sample.status = Status.TRUNCATED
                break

            observation = await asyncio.to_thread(episode.step, output["text"])
            if observation is None or turn == task["max_turns"]:
                break  # the harness ended the episode, or no turn is left
            obs_tokens = observation_tokens(
                tokenizer,
                observation,
                last_token=new_tokens[-1] if new_tokens else None,
                template_kwargs=template_kwargs,
            )
            _append(
                sample,
                args,
                tokens=obs_tokens,
                log_probs=None,
                meta_info=None,
                text=tokenizer.decode(obs_tokens),
            )

        if sample.status == Status.PENDING:
            sample.status = Status.COMPLETED
        sample.reward = await harbor_episode.verify(task["reward_key"])
    except Exception:
        logger.exception(
            "Harbor task %s: episode failed; scoring 0.0", task.get("task_name")
        )
        if sample.status == Status.PENDING:
            sample.status = Status.COMPLETED
        sample.reward = 0.0
    finally:
        if episode is not None:
            sample.metadata = {
                **(sample.metadata or {}),
                **{f"harbor_{k}": v for k, v in episode.metadata.items()},
            }
        await environments.aclose()
    return sample
