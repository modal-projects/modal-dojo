from __future__ import annotations

from modal_dojo.common import ids
from modal_dojo.common.dataset import DatasetConfig, HuggingFaceDataset
from modal_dojo.common.models import ModelConfig, Qwen3_4B
from modal_dojo.common.train import TrainConfig
from modal_dojo.train_recipes.slime_recipe import SlimeRecipe
from modal_dojo.train_recipes.slime_recipe.qwen3_4b import Qwen3_4B_Recipe


def test_create_hash_has_word_word_hash_shape() -> None:
    value = ids.create_hash("model", "ckpt", "recipe", "app", "path")

    assert value.count("-") >= 2


def test_create_hash_suffix_is_stable_for_same_parts(monkeypatch) -> None:
    import randomname

    monkeypatch.setattr(randomname, "get_name", lambda *, sep: "brisk-river")
    first = ids.create_hash("model", "ckpt", "recipe", "app", "path")
    second = ids.create_hash("model", "ckpt", "recipe", "app", "path")

    assert first.rsplit("-", 1)[-1] == second.rsplit("-", 1)[-1]


def test_create_hash_suffix_differs_for_different_parts(monkeypatch) -> None:
    import randomname

    monkeypatch.setattr(randomname, "get_name", lambda *, sep: "brisk-river")
    first = ids.create_hash("model-a", "ckpt", "recipe", "app", "path")
    second = ids.create_hash("model-b", "ckpt", "recipe", "app", "path")

    assert first.rsplit("-", 1)[-1] != second.rsplit("-", 1)[-1]


def test_train_config_generates_fresh_run_id_per_call(monkeypatch) -> None:
    class DummyDataset(DatasetConfig):
        def cache_key(self) -> str | None:
            raise AssertionError("run ID generation must not resolve dataset paths")

        def input_key(self) -> str:
            return "prompt"

        def label_key(self) -> str:
            return "label"

        def rows(self):
            yield {"prompt": "p", "label": "l"}

    calls: list[tuple[str, ...]] = []

    def fake_create_hash(*parts: str) -> str:
        calls.append(parts)
        return f"brisk-river-{len(calls):08x}"

    monkeypatch.setattr(
        "modal_dojo.common.train.create_hash",
        fake_create_hash,
    )

    config = TrainConfig(
        dataset=DummyDataset(),
        model=ModelConfig(model_name="Qwen/Qwen3-4B"),
        recipe=Qwen3_4B_Recipe(),
    )

    first = config._generate_training_run_id()
    second = config._generate_training_run_id()

    assert first != second
    assert len(calls) == 2
    assert [parts[2] for parts in calls] == ["Qwen3_4B_Recipe:slime"] * 2
    assert all(parts[3] == "" for parts in calls)


def test_the_wandb_run_id_is_the_whole_training_run_id() -> None:
    """The same id is exported as WANDB_RUN_ID and recorded for the dashboard's
    deep link, so the producer and the record have to agree on it, and it has to
    stay distinguishing: WANDB_RESUME=allow turns a repeat into a resume of the
    earlier run rather than a new one."""
    from modal_dojo.common.run import metric_run_id_for_attempt
    from modal_dojo.common.wandb import WandbConfig

    run_id = "electric-batter-6362579afd91"
    assert metric_run_id_for_attempt(run_id, 1) == run_id
    assert metric_run_id_for_attempt(run_id, 3) == f"{run_id}-a3"

    summary = TrainConfig(
        dataset=HuggingFaceDataset(
            hf_repo="some/dataset",
            input_column="prompt",
            output_column="answer",
            input_format="text",
        ),
        model=Qwen3_4B(),
        recipe=SlimeRecipe(
            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,
            num_rollout=1,
            rollout_batch_size=16,
            rollout_max_response_len=4096,
            rollout_temperature=1.0,
            save_interval=10,
            metrics=WandbConfig(project="p", entity="e"),
        ),
    )._build_config_summary(run_id)

    assert summary["metrics"]["run_id"] == run_id
