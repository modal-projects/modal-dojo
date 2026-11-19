import pytest

from modal_dojo import (
    Budget,
    DatasetConfig,
    Environment,
    Gemma4_26B_A4B,
    RoutedTrainConfig,
    ModelConfig,
    Qwen3_5_4B,
    Qwen3_5_4B_Recipe,
)
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe


class _Rows(DatasetConfig):
    def __init__(self, n: int) -> None:
        self.n = n

    def input_key(self) -> str:
        return "prompt"

    def label_key(self) -> str:
        return "label"

    def rows(self):
        return [{"prompt": f"q{i}", "label": "a"} for i in range(self.n)]


async def exact_match(args, sample, **kwargs) -> float:
    return float(sample.response.strip() == sample.label)


async def multi_turn(args, sample, sampling_params):
    return sample


def test_environment_holds_dataset_and_functions():
    dataset = _Rows(1)
    env = Environment(dataset=dataset, reward_fn=exact_match)
    assert (env.dataset, env.reward_fn, env.generate_fn) == (dataset, exact_match, None)


def test_uses_preset_and_wires_environment():
    env = Environment(_Rows(1), exact_match, generate_fn=multi_turn)

    routed = RoutedTrainConfig(Qwen3_5_4B(), env, Budget(max_rollouts=7))

    recipe = routed.recipe
    assert type(recipe) is Qwen3_5_4B_Recipe
    assert recipe.num_rollout == 7
    assert recipe.budget == Budget(max_rollouts=7)
    assert recipe.custom_rm_function is exact_match
    assert recipe.extra_config["custom_generate_function_path"].endswith("multi_turn")
    # One task: the batch shrinks to it, one update per rollout.
    assert recipe.rollout_batch_size == 1
    assert recipe.global_batch_size == recipe.n_samples_per_prompt
    assert routed.train_config.dataset is env.dataset
    assert recipe.lr == Qwen3_5_4B_Recipe().lr  # preset values kept


def test_overrides_win_and_are_checked():
    env = Environment(_Rows(1), exact_match)
    recipe = RoutedTrainConfig(
        Qwen3_5_4B(),
        env,
        Budget(max_rollouts=3),
        recipe_overrides={"n_samples_per_prompt": 8, "lr": 1e-6},
    ).recipe
    assert (recipe.n_samples_per_prompt, recipe.global_batch_size) == (8, 8)
    assert recipe.lr == 1e-6
    assert recipe.custom_generate_function is None
    assert "custom_generate_function_path" not in (recipe.extra_config or {})

    with pytest.raises(DojoConfigError, match="not_a_field"):
        RoutedTrainConfig(
            Qwen3_5_4B(),
            env,
            Budget(max_rollouts=3),
            recipe_overrides={"not_a_field": 1},
        )
    with pytest.raises(DojoConfigError, match="through budget"):
        RoutedTrainConfig(
            Qwen3_5_4B(), env, Budget(max_hours=1), recipe_overrides={"num_rollout": 5}
        )
    with pytest.raises(DojoConfigError, match="must be a Budget"):
        RoutedTrainConfig(Qwen3_5_4B(), env, 10)


def test_keeps_preset_batch_when_tasks_suffice():
    recipe = RoutedTrainConfig(
        Qwen3_5_4B(), Environment(_Rows(3), exact_match), Budget(max_rollouts=1)
    ).recipe
    preset = Qwen3_5_4B_Recipe()
    assert (recipe.rollout_batch_size, recipe.global_batch_size) == (
        preset.rollout_batch_size,
        preset.global_batch_size,
    )


def test_falls_back_to_miles_and_rejects_unknown_models():
    env = Environment(_Rows(1), exact_match)
    recipe = RoutedTrainConfig(Gemma4_26B_A4B(), env, Budget(max_rollouts=1)).recipe
    assert isinstance(recipe, MilesRecipe)

    class Unknown(ModelConfig):
        model_name = "org/unknown"

    with pytest.raises(DojoConfigError, match="No recipe preset"):
        RoutedTrainConfig(Unknown(), env, Budget(max_rollouts=1))


def test_cost_and_time_budgets_run_until_spent():
    budget = Budget(max_cost_usd=25, max_hours=2)
    recipe = RoutedTrainConfig(
        Qwen3_5_4B(), Environment(_Rows(4), exact_match), budget
    ).recipe
    assert recipe.budget is budget
    assert recipe.num_rollout == budget.num_rollout  # stops on cost or time


def test_context_length_sets_rollout_and_batch_limits():
    env = Environment(_Rows(1), exact_match, max_context_len=32_768)
    recipe = RoutedTrainConfig(Qwen3_5_4B(), env, Budget(max_rollouts=1)).recipe
    assert recipe.rollout_max_response_len == recipe.eval_max_response_len == 32_767
    assert recipe.max_tokens_per_gpu == 32_768
    assert recipe.extra_config["rollout_max_context_len"] == 32_768

    small = Environment(_Rows(1), exact_match, max_context_len=1024)
    recipe = RoutedTrainConfig(Qwen3_5_4B(), small, Budget(max_rollouts=1)).recipe
    assert recipe.max_tokens_per_gpu == Qwen3_5_4B_Recipe().max_tokens_per_gpu

    miles = RoutedTrainConfig(Gemma4_26B_A4B(), env, Budget(max_rollouts=1)).recipe
    assert miles.rollout_max_response_len == 32_767

    overridden = RoutedTrainConfig(
        Qwen3_5_4B(),
        env,
        Budget(max_rollouts=1),
        recipe_overrides={"rollout_max_response_len": 4096},
    ).recipe
    assert overridden.rollout_max_response_len == 4096

    with pytest.raises(DojoConfigError, match="max_context_len"):
        Environment(_Rows(1), exact_match, max_context_len=1)
