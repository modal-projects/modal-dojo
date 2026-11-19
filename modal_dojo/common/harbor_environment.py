"""Harbor tasks as a training environment: a dataset plus a verifier-backed reward."""

from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import io
import json
import logging
import re
import shutil
import subprocess
import sys
import tarfile
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
    TMUX_INSTALL_COMMAND,
    Harness,
    env_assignments,
    episode_class,
    extract_script,
    parse_harness,
    require_package,
    run_shell,
)
from modal_dojo.common.sandbox import open_sandbox

logger = logging.getLogger(__name__)

# Harbor leaves ``agent.timeout_sec`` unset unless a task sets it.
_DEFAULT_AGENT_TIMEOUT_SEC = 600.0


def _tar_b64(directory: Path) -> str:
    """Gzipped tarball of ``directory``'s contents, base64-encoded."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for child in sorted(directory.iterdir()):
            tar.add(child, arcname=child.name)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _untar_b64(payload: str, dest: Path) -> None:
    with tarfile.open(
        fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz"
    ) as tar:
        tar.extractall(dest, filter="data")


def _find_task_dirs(root: Path) -> list[Path]:
    require_package("harbor", HARBOR_VERSION)
    from harbor.models.task.task import Task

    if Task.is_valid_dir(root):
        return [root]
    return sorted(
        p.parent for p in root.rglob("task.toml") if Task.is_valid_dir(p.parent)
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
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            tar.extractall(tmp, filter="data")
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
        HarborEnvironment("swesmith@1.0", harness=Harness.MINI_SWE_AGENT),
        Budget(max_cost_usd=100),
    ).train()
    ```

    Each episode runs in a sandbox built from the task's ``environment/``; the
    task's ``tests/test.sh`` then sets the reward.

    Attributes:
        source: A Harbor hub dataset (``name[@version]``) or a public GitHub URL.
        harness: A ``Harness``: ``BASH``, ``MINI_SWE_AGENT``, or ``TERMINUS_2``.
        task_names: Only use these tasks.
        max_tasks: Use at most this many tasks.
        max_turns: Model replies per episode (default: 1 for bash, else 50).
        max_output_chars: Command output shown per turn (bash only).
        system_prompt: System message (bash only).
        reward_key: Key to read from ``reward.json``.
        max_context_len: Token limit for one episode; tasks don't set one.
    """

    def __init__(
        self,
        source: str,
        *,
        harness: Harness,
        task_names: list[str] | None = None,
        max_tasks: int | None = None,
        max_turns: int | None = None,
        max_output_chars: int = 4000,
        system_prompt: str | None = None,
        reward_key: str = "reward",
        max_context_len: int | None = None,
    ) -> None:
        self._source_spec = _parse_source(source)
        harness = parse_harness(harness)
        episode = episode_class(harness)
        if max_turns is None:
            max_turns = episode.default_max_turns
        if max_turns < 1:
            raise DojoConfigError(f"max_turns must be at least 1, got {max_turns}")
        if system_prompt is not None and harness != Harness.BASH:
            raise DojoConfigError(
                f"system_prompt applies to the bash harness only; {harness.value!r} "
                "uses its own prompts"
            )
        self.source = source
        self.harness = harness
        self.task_names = task_names
        self.max_tasks = max_tasks
        self.max_turns = max_turns
        self.max_output_chars = max_output_chars
        self.system_prompt = system_prompt
        self.reward_key = reward_key
        super().__init__(None, harbor_reward, max_context_len=max_context_len)
        self._dataset: _HarborTaskDataset | None = None

    @property
    def single_turn(self) -> bool:
        """Whether the framework's own generation and ``reward_fn`` suffice."""
        return self.harness == Harness.BASH and self.max_turns == 1

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
        from harbor.models.task.config import TaskOS
        from harbor.models.task.task import Task

        name = task_dir.name
        try:
            task = Task(task_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise DojoConfigError(f"Invalid Harbor task {name!r}: {exc}") from exc
        config, env_cfg = task.config, task.config.environment
        if config.steps:
            raise DojoConfigError(
                f"Harbor task {name!r} is multi-step; HarborEnvironment supports "
                "single-step tasks only."
            )
        if env_cfg.os != TaskOS.LINUX:
            raise DojoConfigError(
                f"Harbor task {name!r} targets os={env_cfg.os.value!r}; only linux "
                "tasks are supported."
            )
        test_script = task.paths.discovered_test_path_for(env_cfg.os)
        if test_script is None:
            raise DojoConfigError(
                f"Harbor task {name!r} has no test script in tests/ (separate "
                "verifier environments are not supported)."
            )
        if env_cfg.docker_image:
            image: dict[str, str] = {"docker_image": env_cfg.docker_image}
        elif (task.paths.environment_dir / "Dockerfile").is_file():
            image = {"dockerfile_context": _tar_b64(task.paths.environment_dir)}
        else:
            raise DojoConfigError(
                f"Harbor task {name!r} has neither environment.docker_image nor "
                "environment/Dockerfile (docker-compose environments are not "
                "supported)."
            )

        instruction = task.instruction.strip()
        label = {
            "task_name": name,
            "image": image,
            "workdir": env_cfg.workdir,
            "cpu": env_cfg.cpus,
            "memory_mb": env_cfg.memory_mb,
            "env": env_cfg.env,
            "tests": _tar_b64(task.paths.tests_dir),
            "test_script": test_script.relative_to(task.paths.tests_dir).as_posix(),
            "verifier_env": config.verifier.env,
            "verifier_timeout_sec": config.verifier.timeout_sec,
            "agent_timeout_sec": config.agent.timeout_sec or _DEFAULT_AGENT_TIMEOUT_SEC,
            "instruction": instruction,
            "harness": self.harness,
            "system_prompt": self.system_prompt,
            "reward_key": self.reward_key,
            "max_turns": self.max_turns,
            "max_output_chars": self.max_output_chars,
        }
        messages = episode_class(self.harness).preview_messages(instruction, label)
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
        return None if self.single_turn else harbor_generate


def parse_reward(
    reward_json: str | None, reward_txt: str | None, *, reward_key: str
) -> float:
    """Harbor's reward from the verifier's output files.

    Harbor reads ``reward.json`` as a dict of rewards and ``reward.txt`` as
    ``{"reward": value}``, but only inside private ``Verifier`` methods that
    read local files, so this mirrors them and then picks ``reward_key``.
    """
    if reward_json and reward_json.strip():
        data = json.loads(reward_json)
        if isinstance(data, (int, float)):
            return float(data)
        if reward_key in data:
            return float(data[reward_key])
        numeric = [v for v in data.values() if isinstance(v, (int, float))]
        if len(numeric) == 1:
            return float(numeric[0])
        raise ValueError(
            f"reward.json has no {reward_key!r} key and several values: {sorted(data)}"
        )
    if reward_txt and reward_txt.strip():
        return float(reward_txt.strip())
    raise ValueError("verifier wrote neither reward.json nor reward.txt")


def _load_task(sample: Any) -> dict[str, Any]:
    label = sample.label
    return json.loads(label) if isinstance(label, str) else label


_image_cache: dict[str, Any] = {}


def _task_image(task: dict[str, Any]):
    import modal

    image_spec = task["image"]
    needs_tmux = task.get("harness") == Harness.TERMINUS_2
    key = hashlib.sha256(
        json.dumps([image_spec, needs_tmux], sort_keys=True).encode()
    ).hexdigest()
    if key in _image_cache:
        return _image_cache[key]
    if "docker_image" in image_spec:
        image = modal.Image.from_registry(image_spec["docker_image"])
    else:
        context = Path(tempfile.mkdtemp(prefix=f"harbor-{task['task_name']}-"))
        _untar_b64(image_spec["dockerfile_context"], context)
        image = modal.Image.from_dockerfile(context / "Dockerfile", context_dir=context)
    if needs_tmux:  # a layer on the task image, built once per task
        image = image.run_commands(TMUX_INSTALL_COMMAND)
    _image_cache[key] = image
    return image


def _open_sandbox(task: dict[str, Any]):
    """A started sandbox in the task's container; close with ``__exit__``."""
    return open_sandbox(
        app_name="modal-dojo-harbor",
        image=_task_image(task),
        timeout=int(task["agent_timeout_sec"] + task["verifier_timeout_sec"] + 3600),
        workdir=task.get("workdir"),
        cpu=float(task["cpu"]) if task.get("cpu") else None,
        memory=task.get("memory_mb"),
        setup=[("mkdir", "-p", SCRATCH_DIR)],
    )


def _run_tests(sb: Any, task: dict[str, Any]) -> float:
    """Upload and run the task's tests in ``sb`` as Harbor does; return the reward."""
    from harbor.models.task.config import TaskOS
    from harbor.models.trial.paths import EnvironmentPaths
    from harbor.utils.scripts import build_execution_command, needs_chmod

    paths = EnvironmentPaths()
    tests_dir, verifier_dir = paths.tests_dir, paths.verifier_dir
    sb.write(f"{SCRATCH_DIR}/tests.tar.gz", base64.b64decode(task["tests"]))
    sb.run(
        "bash",
        "-c",
        f"rm -rf {tests_dir} {verifier_dir} && "
        f"mkdir -p {tests_dir} {verifier_dir} && "
        f"tar -xzf {SCRATCH_DIR}/tests.tar.gz -C {tests_dir}",
    )
    script = tests_dir / task["test_script"]
    if needs_chmod(script):
        sb.run("chmod", "+x", str(script))
    command = build_execution_command(
        script, verifier_dir / "test-stdout.txt", task_os=TaskOS.LINUX
    )
    env = env_assignments({**task.get("env", {}), **task.get("verifier_env", {})})
    tested = sb.run(
        "bash",
        "-c",
        f"{env} {command}".strip(),
        timeout=int(task["verifier_timeout_sec"]),
    )
    reward_json = sb.run("cat", str(paths.reward_json_path))
    reward_txt = sb.run("cat", str(paths.reward_text_path))
    try:
        return parse_reward(
            reward_json.stdout if reward_json.returncode == 0 else None,
            reward_txt.stdout if reward_txt.returncode == 0 else None,
            reward_key=task["reward_key"],
        )
    except ValueError:
        logger.warning(
            "Harbor task %s: no usable reward (test exit %s); scoring 0.0",
            task["task_name"],
            tested.returncode,
        )
        return 0.0


def _verify(task: dict[str, Any], script: str) -> float:
    """Run ``script`` then the task's tests in a fresh sandbox; return the reward."""
    sb = _open_sandbox(task)
    try:
        run_shell(sb, task, script, timeout_sec=task["agent_timeout_sec"])
        return _run_tests(sb, task)
    finally:
        sb.__exit__(None, None, None)


async def harbor_reward(args: Any, sample: Any, **kwargs: Any) -> float:
    """Slime/Miles ``custom_rm`` for rows built by ``HarborEnvironment``."""
    task = _load_task(sample)
    if task["harness"] != Harness.BASH or task["max_turns"] > 1:
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
        return await asyncio.to_thread(_verify, task, script)
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
    args: Any, sample: Any, sampling_params: dict[str, Any], evaluation: bool = False
) -> Any:
    """Slime/Miles ``custom_generate_function`` for ``HarborEnvironment`` rows."""
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

    sb = None
    episode = None
    try:
        sb = await asyncio.to_thread(_open_sandbox, task)
        deadline = time.monotonic() + task["agent_timeout_sec"]
        episode = episode_class(task["harness"])(sb, task, deadline)
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
        sample.reward = await asyncio.to_thread(_run_tests, sb, task)
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
        if sb is not None:
            await asyncio.to_thread(sb.__exit__, None, None, None)
    return sample
