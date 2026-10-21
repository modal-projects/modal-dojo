"""Optional integration contracts: run with uv run --extra spindle pytest."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("spindle")
pytest.importorskip("tinker_cookbook")

import chz
import tinker
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl import rollouts, train as rl_train

from modal_dojo import (
    DatasetConfig,
    DojoDatasetBuilder,
    HuggingFaceDataset,
    SpindleRecipe,
)
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.frameworks.spindle import dataset_builders, launcher
from modal_dojo.frameworks.spindle.trajectories import capture_trajectories
from tutorials.spindle_rl import build_recipe


class Tokenizer:
    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class Renderer:
    def build_generation_prompt(self, messages):
        return tinker.ModelInput.from_ints(list(map(ord, messages[-1]["content"])))

    def get_stop_sequences(self):
        return []

    def parse_response(self, tokens):
        return {"role": "assistant", "content": Tokenizer().decode(tokens)}, None


class CustomDataset(DatasetConfig):
    def input_key(self):
        return "prompt"

    def label_key(self):
        return "answer"

    def rows(self):
        return [{"prompt": "2+2", "answer": "4"}]


def test_unsupported_dataset_fails_before_loading():
    source = CustomDataset()
    source.rows = Mock(side_effect=AssertionError("must not load data"))
    with pytest.raises(DojoConfigError, match="to_dataset_builder"):
        DojoDatasetBuilder.from_dataset(source)


def test_custom_dataset_can_opt_in():
    class Supported(CustomDataset):
        def to_dataset_builder(self, **kwargs):
            return kwargs["builder"]

    sentinel = object()
    assert DojoDatasetBuilder.from_dataset(Supported(), builder=sentinel) is sentinel


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"model_name": "wrong/model"}, "match"),
        ({"lora_rank": 64}, "lora_rank"),
        ({"max_steps": 0}, "max_steps"),
        ({"base_url": "https://example.com"}, "base_url"),
        ({"dataset_builder": CustomDataset()}, "dataset_builder"),
        ({"max_tokens": 32768}, "context window"),
        (
            {
                "async_config": rl_train.AsyncConfig(
                    max_steps_off_policy=1, groups_per_batch=4
                )
            },
            "synchronous",
        ),
    ],
)
def test_invalid_recipe_fails_during_construction(changes, match):
    recipe = build_recipe()
    with pytest.raises(DojoConfigError, match=match):
        SpindleRecipe(
            chz.replace(recipe.rl_config, **changes), deploy_config=recipe.deploy_config
        )


def test_deploy_config_class_and_rank_limit():
    from spindle.configs.qwen35_2b_lora_32k import Config

    recipe = build_recipe()
    assert (
        SpindleRecipe(recipe.rl_config, deploy_config=Config).deploy_config.model
        == recipe.rl_config.model_name
    )
    config = Config(overrides={"miles_cfg.max_lora_rank": 8})
    with pytest.raises(DojoConfigError, match="lora_rank"):
        SpindleRecipe(recipe.rl_config, deploy_config=config)


@pytest.fixture
def hf_builder(monkeypatch):
    source = HuggingFaceDataset(
        hf_repo="test/data", input_column="q", output_column="a"
    )
    monkeypatch.setattr(
        source,
        "rows",
        lambda: [
            {
                "messages": [
                    {"role": "system", "content": "Calculate"},
                    {"role": "user", "content": "2+2"},
                ],
                "label": "4",
            },
            {"messages": [{"role": "user", "content": "2+3"}], "label": "5"},
            {"messages": [{"role": "user", "content": "3+3"}], "label": "6"},
        ],
    )
    monkeypatch.setattr(dataset_builders, "get_tokenizer", lambda _: Tokenizer())
    monkeypatch.setattr(
        dataset_builders.renderers, "get_renderer", lambda *a, **k: Renderer()
    )
    return DojoDatasetBuilder.from_dataset(
        source,
        model_name="test/model",
        renderer_name="test",
        batch_size=2,
        group_size=2,
        reward_fn=lambda response, label: float(response.strip() == label),
    )


def test_hf_adapter_uses_messages_labels_and_partial_batch(hf_builder):
    async def check():
        dataset, evaluation = await hf_builder()
        assert evaluation is None
        assert len(dataset) == 2
        assert len(dataset.get_batch(1)) == 1
        envs = await dataset.get_batch(0)[0].make_envs()
        assert len(envs) == 2
        assert envs[0].messages[0]["role"] == "system"
        assert (
            Tokenizer().decode((await envs[0].initial_observation())[0].to_ints())
            == "2+2"
        )
        for response, expected in [("4", 1), ("5", 0), ("", 0), ("bad", 0)]:
            assert (await envs[0].step(list(map(ord, response)))).reward == expected
        with pytest.raises(IndexError):
            dataset.get_batch(2)

    asyncio.run(check())


@pytest.mark.parametrize("kwargs", [{"input_format": "raw"}, {"output_column": None}])
def test_hf_unsupported_shape_fails_before_loading(kwargs):
    source = HuggingFaceDataset(
        **{"hf_repo": "unused", "input_column": "q", "output_column": "a", **kwargs}
    )
    with pytest.raises(DojoConfigError):
        source.to_dataset_builder(
            model_name="x",
            renderer_name="x",
            batch_size=1,
            group_size=1,
            reward_fn=lambda x, y: 0,
        )


def test_actual_rollout_is_captured_with_rewards_tokens_and_text(
    hf_builder, tmp_path, capsys
):
    original = rollouts.do_group_rollout

    async def policy(*args, **kwargs):
        return TokensWithLogprobs(tokens=[ord("4")], maybe_logprobs=[-0.3])

    async def check():
        dataset, _ = await hf_builder()
        async with capture_trajectories(str(tmp_path), Tokenizer()):
            group = await rollouts.do_group_rollout(dataset.get_batch(0)[0], policy)
        assert group.get_total_rewards() == [1, 1]
        assert rollouts.do_group_rollout is original

    asyncio.run(check())
    record = json.loads((tmp_path / "dojo-trajectories.jsonl").read_text())
    assert record == json.loads(capsys.readouterr().out)
    assert record["decoded"][0][0] == {"prompt": "2+2", "response": "4"}
    raw = record["trajectory_group"]["trajectories_G"][0]["transitions"][0]
    assert raw["reward"] == 1
    assert raw["ac"]["maybe_logprobs"] == [-0.3]
    assert raw["ob"]["chunks"][0]["tokens"] == list(map(ord, "2+2"))


def test_capture_restored_after_failure(tmp_path):
    original = rollouts.do_group_rollout

    async def check():
        with pytest.raises(RuntimeError):
            async with capture_trajectories(str(tmp_path), Tokenizer()):
                raise RuntimeError("rollout failed")

    asyncio.run(check())
    assert rollouts.do_group_rollout is original


def test_capture_cancels_sibling_rollouts_before_closing(monkeypatch, tmp_path):
    async def check():
        entered = asyncio.Event()

        async def blocked(*args):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(rollouts, "do_group_rollout", blocked)
        with pytest.raises(RuntimeError):
            async with capture_trajectories(str(tmp_path), Tokenizer()):
                sibling = asyncio.create_task(rollouts.do_group_rollout(None))
                await entered.wait()
                raise RuntimeError("another group failed")
        assert sibling.cancelled()
        assert rollouts.do_group_rollout is blocked

    asyncio.run(check())


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("finish_supported", [False, True])
def test_train_routes_to_owned_server_and_restores_hooks(
    monkeypatch, tmp_path, fail, finish_supported
):
    recipe = build_recipe(log_path=str(tmp_path))
    lifecycle = []

    @asynccontextmanager
    async def deployment(config, **kwargs):
        lifecycle.append("deploy")
        try:
            yield "https://owned.modal.run", "secret"
        finally:
            lifecycle.append("cleanup")

    monkeypatch.setattr(launcher, "deployment", deployment)
    monkeypatch.setattr(tokenizer_utils, "get_tokenizer", lambda _: Tokenizer())
    import httpx

    close = AsyncMock()
    if not finish_supported:
        close.side_effect = tinker.NotFoundError(
            "unsupported finish route",
            response=httpx.Response(
                404, request=httpx.Request("POST", "https://owned.modal.run/finish")
            ),
            body=None,
        )
    service = Mock(return_value=SimpleNamespace(close=close))
    monkeypatch.setattr(tinker, "ServiceClient", service)
    original_tinker = rl_train.tinker
    original_rollout = rollouts.do_group_rollout

    async def main(config):
        assert config.base_url == "https://owned.modal.run"
        assert config.dataset_builder is recipe.rl_config.dataset_builder
        assert rollouts.do_group_rollout is not original_rollout
        rl_train.tinker.ServiceClient(base_url=config.base_url, user_metadata={})
        if fail:
            raise RuntimeError("training failed")

    monkeypatch.setattr(rl_train, "main", main)
    if fail:
        with pytest.raises(RuntimeError, match="training failed"):
            asyncio.run(recipe.train())
    else:
        asyncio.run(recipe.train())
    assert lifecycle == ["deploy", "cleanup"]
    assert recipe.rl_config.base_url is None
    service.assert_called_once_with(
        base_url="https://owned.modal.run", api_key="secret", user_metadata={}
    )
    assert rl_train.tinker is original_tinker
    assert rollouts.do_group_rollout is original_rollout
    assert not launcher._active_run.locked()


@pytest.mark.parametrize("missing_hf", [False, True])
def test_deployment_cleans_partial_failure(monkeypatch, missing_hf):
    import modal
    import spindle.deployment_cli as deploy_cli

    recipe = build_recipe().deploy_config
    recipe.platform["modal"]["environment"] = "test-env"
    hydrate = AsyncMock(
        side_effect=[
            None,
            modal.exception.NotFoundError("missing") if missing_hf else None,
        ]
    )
    monkeypatch.setattr(
        modal.Secret,
        "from_name",
        Mock(return_value=SimpleNamespace(hydrate=SimpleNamespace(aio=hydrate))),
    )
    create, delete = AsyncMock(), AsyncMock()
    monkeypatch.setattr(
        modal.Secret,
        "objects",
        SimpleNamespace(
            create=SimpleNamespace(aio=create), delete=SimpleNamespace(aio=delete)
        ),
    )
    monkeypatch.setattr(
        deploy_cli, "deploy", Mock(side_effect=RuntimeError("build failed"))
    )
    stopped = []
    monkeypatch.setattr(
        launcher, "_stop_app", lambda name, env: stopped.append((name, env))
    )

    async def check():
        async with launcher.deployment(recipe):
            pytest.fail("must not yield on failed deploy")

    with pytest.raises(RuntimeError, match="build failed"):
        asyncio.run(check())
    assert len(stopped) == 4
    assert all(env == "test-env" for _, env in stopped)
    assert all("dojo-spindle-" in name for name, _ in stopped)
    assert recipe.name == "qwen35-2b-lora-32k"
    create.assert_awaited_once()
    delete.assert_awaited_once()
    deployed = deploy_cli.deploy.call_args.args[0][0]
    assert deployed.recipe.platform["secrets"]["huggingface"] == (
        None if missing_hf else "huggingface-secret"
    )


def test_harbor_preserves_selected_split_and_repeats(tmp_path):
    from modal_dojo import HarborDataset

    for name in ["a", "b", "c"]:
        task = tmp_path / name
        task.mkdir()
        (task / "instruction.md").write_text(f"Solve {name}")
        (task / "task.toml").write_text("[environment]\ncpus=1\n")
    source = HarborDataset(
        task_root=str(tmp_path), train_size=2, eval_size=1, train_repeats=2
    )
    builder = source.to_dataset_builder(
        model_name="test", renderer_name="test", batch_size=3, group_size=2
    )

    async def check():
        dataset, evaluation = await builder()
        assert evaluation is None
        assert len(dataset) == 2
        assert [g.task.task_name for g in dataset.groups] == ["a", "a", "b", "b"]
        assert dataset.groups[0].task.instruction == "Solve a"

    asyncio.run(check())
    with pytest.raises(DojoConfigError, match="Unknown"):
        source.to_dataset_builder(
            model_name="test",
            renderer_name="test",
            batch_size=1,
            group_size=1,
            typo=True,
        )


def test_cancellation_waits_for_deployment_before_cleanup(monkeypatch):
    import threading
    import modal
    import spindle.deployment_cli as deploy_cli

    started, release = threading.Event(), threading.Event()
    events = []

    def deploy(configs):
        started.set()
        assert release.wait(timeout=5)
        events.append("deployed")

    monkeypatch.setattr(deploy_cli, "deploy", deploy)
    monkeypatch.setattr(
        modal.Secret,
        "from_name",
        Mock(return_value=SimpleNamespace(hydrate=SimpleNamespace(aio=AsyncMock()))),
    )
    monkeypatch.setattr(
        modal.Secret,
        "objects",
        SimpleNamespace(
            create=SimpleNamespace(aio=AsyncMock()),
            delete=SimpleNamespace(aio=AsyncMock()),
        ),
    )
    monkeypatch.setattr(launcher, "_stop_app", lambda *args: events.append("stopped"))

    async def check():
        async def run():
            async with launcher.deployment(build_recipe().deploy_config):
                pytest.fail("cancelled deployment must not yield")

        task = asyncio.create_task(run())
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not events
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(check())
    assert events == ["deployed", "stopped", "stopped", "stopped", "stopped"]


def test_missing_secret_fails_without_creating_resources(monkeypatch):
    import modal
    import spindle.deployment_cli as deploy_cli

    hydrate = AsyncMock(side_effect=modal.exception.NotFoundError("missing"))
    monkeypatch.setattr(
        modal.Secret,
        "from_name",
        Mock(return_value=SimpleNamespace(hydrate=SimpleNamespace(aio=hydrate))),
    )
    create = AsyncMock()
    monkeypatch.setattr(
        modal.Secret, "objects", SimpleNamespace(create=SimpleNamespace(aio=create))
    )
    deploy = Mock()
    monkeypatch.setattr(deploy_cli, "deploy", deploy)

    async def check():
        async with launcher.deployment(build_recipe().deploy_config):
            pytest.fail("missing credentials must not deploy")

    with pytest.raises(DojoConfigError, match="spindle-proxy"):
        asyncio.run(check())
    deploy.assert_not_called()
    create.assert_not_awaited()


def test_resume_rejected_before_deployment(tmp_path):
    recipe = build_recipe(log_path=str(tmp_path))
    (tmp_path / "checkpoints.jsonl").write_text(
        '{"state_path": "tinker://old/state"}\n'
    )
    with pytest.raises(DojoConfigError, match="fresh log_path"):
        SpindleRecipe(recipe.rl_config, deploy_config=recipe.deploy_config)


@pytest.mark.parametrize("already_stopped", [False, True])
def test_app_stop_is_noninteractive_and_idempotent(monkeypatch, already_stopped):
    import modal

    monkeypatch.setattr(
        modal.App, "lookup", Mock(return_value=SimpleNamespace(app_id="ap-owned"))
    )

    def command(args, **kwargs):
        assert "--yes" in args
        assert args[-2:] == ["--env", "helena-dev"]
        return SimpleNamespace(
            returncode=int(already_stopped),
            stdout="",
            stderr="App is already stopped." if already_stopped else "",
        )

    monkeypatch.setattr(launcher.subprocess, "run", command)
    launcher._stop_app("owned", "helena-dev")
