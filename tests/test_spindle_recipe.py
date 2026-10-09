import pytest
from pydantic import ValidationError

from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.framework import Framework
from modal_dojo.common.models import Qwen3_5_4B, Qwen3_8_27B
from modal_dojo.common.status import SpindleStatus, resolve_framework_status
from modal_dojo.frameworks.spindle.deployment import spindle_config_values
from modal_dojo.frameworks.spindle.grpo import (
    GrpoSettings,
    make_sample,
    prompt_messages,
)
from modal_dojo.train_recipes.miles_recipe import MilesRecipe
from modal_dojo.train_recipes.spindle_recipe import (
    Qwen3_8_27B_Spindle_Recipe,
    SpindleRecipe,
)


def test_spindle_recipe_is_a_miles_recipe():
    recipe = Qwen3_8_27B_Spindle_Recipe(num_rollout=3, lr=1e-5)
    assert isinstance(recipe, MilesRecipe)
    assert recipe.lr == 1e-5
    assert recipe.model_config_class is Qwen3_8_27B
    assert recipe.max_context_length == 32768


def test_spindle_fields_not_emitted_as_cli_flags():
    fields = Qwen3_8_27B_Spindle_Recipe()._fields()
    for name in ("max_context_length", "spindle_git_ref", "grpo_config_overrides"):
        assert name not in fields


def test_only_32k_context_supported():
    with pytest.raises(ValidationError, match="max_context_length"):
        Qwen3_8_27B_Spindle_Recipe(max_context_length=65536)


def test_only_qwen3_8_27b_supported():
    assert isinstance(
        SpindleRecipe.get_base_recipe(Qwen3_8_27B()), Qwen3_8_27B_Spindle_Recipe
    )
    with pytest.raises(DojoConfigError, match="Qwen/Qwen3.8-27B"):
        SpindleRecipe.get_base_recipe(Qwen3_5_4B())
    with pytest.raises(DojoConfigError, match="Qwen/Qwen3.8-27B"):
        Qwen3_8_27B_Spindle_Recipe().validate_model_parallelism(Qwen3_5_4B())


def test_unsupported_miles_fields_raise():
    with pytest.raises(ValidationError, match="use_critic"):
        Qwen3_8_27B_Spindle_Recipe(use_critic=True)
    with pytest.raises(ValidationError, match="policy_loss"):
        Qwen3_8_27B_Spindle_Recipe(loss_type="sft_loss")
    with pytest.raises(ValidationError, match="rm_type"):
        Qwen3_8_27B_Spindle_Recipe(rm_type="gsm8k")
    assert Qwen3_8_27B_Spindle_Recipe(rm_type="deepscaler").rm_type == "deepscaler"
    with pytest.raises(ValidationError, match="lora_rank"):
        Qwen3_8_27B_Spindle_Recipe(lora_rank=None)
    with pytest.raises(ValidationError, match="multiple of global_batch_size"):
        Qwen3_8_27B_Spindle_Recipe(
            rollout_batch_size=3, n_samples_per_prompt=3, global_batch_size=4
        )


def test_grpo_settings_derived_from_miles_fields():
    recipe = Qwen3_8_27B_Spindle_Recipe(
        rollout_batch_size=16,
        n_samples_per_prompt=8,
        global_batch_size=32,
        kl_coef=0.05,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=2,
    )
    assert recipe.grpo_num_substeps == 4
    assert recipe.grpo_kl_penalty_coef == 0.05
    assert recipe.inference_replicas == 4
    assert recipe.spindle_frontend_name("run1") == "spindle-dojo-run1"
    assert SpindleRecipe(name="my-run").spindle_frontend_name("x") == "spindle-my-run"


def test_spindle_config_values_mirror_spindle_preset():
    recipe = Qwen3_8_27B_Spindle_Recipe(
        spindle_config_overrides={"sglang_cfg.schedule_policy": "lpm"},
        spindle_platform_overrides={"storage.checkpoints": "my-ckpts"},
        region="us-east",
    )
    values = spindle_config_values(
        recipe, model_name="Qwen/Qwen3.8-27B", training_run_id="run1"
    )
    assert values["name"] == "dojo-run1"
    assert values["model"] == "Qwen/Qwen3.8-27B"
    assert values["max_context_length"] == 32768
    assert values["parameterization"] == "lora"
    assert values["trainer_gpu"] == "H200"
    assert values["trainer_gpus_per_node"] == 8
    assert values["inference_max_replicas"] == 8
    miles_cfg = values["miles_cfg"]
    assert miles_cfg["model_type"] == "qwen3.8-27B"
    assert miles_cfg["tensor_model_parallel_size"] == 4
    assert miles_cfg["max_tokens_per_gpu"] == 32768
    assert miles_cfg["max_lora_rank"] == 32
    assert miles_cfg["target_modules"][0] == "q_proj"
    assert miles_cfg["cli_options"]["recompute_granularity"] == "full"
    assert values["sglang_cfg"]["schedule_policy"] == "lpm"
    assert values["sglang_cfg"]["mem_fraction_static"] == 0.8
    assert values["platform"]["storage"]["checkpoints"] == "my-ckpts"
    assert values["platform"]["modal"]["region"] == "us-east"
    assert values["platform"]["frontend"] == "spindle-dojo-run1"
    assert values["trainer_env"]["TORCHINDUCTOR_COMPILE_THREADS"] == "1"


def test_spindle_config_values_rejects_unknown_model():
    with pytest.raises(ValueError, match="model_type"):
        spindle_config_values(Qwen3_8_27B_Spindle_Recipe(), model_name="other/model")


def test_train_config_dispatches_spindle():
    from modal_dojo.common.train import TrainConfig
    from modal_dojo.common.dataset import HuggingFaceDataset

    config = TrainConfig(
        model=Qwen3_8_27B(),
        dataset=HuggingFaceDataset(
            hf_repo="statworx/haiku",
            input_column="keywords",
            output_column="text",
            input_format="text",
        ),
        recipe=Qwen3_8_27B_Spindle_Recipe(),
    )
    assert config.framework is Framework.SPINDLE
    assert config._initializing_status() is SpindleStatus.INITIALIZING
    assert resolve_framework_status("training", "spindle") is SpindleStatus.TRAINING


def test_grpo_settings_roundtrip_and_sample():
    settings = GrpoSettings(
        model_name="Qwen/Qwen3.8-27B",
        base_url="https://example.modal.run",
        log_path="/tmp/log",
        dataset_path="/data/x.jsonl",
        input_key="prompt",
        label_key="answer",
        apply_chat_template=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert GrpoSettings.from_json(settings.to_json()) == settings
    sample = make_sample({"prompt": "hi", "answer": "4", "extra": 1}, 3, settings)
    assert sample.prompt == "hi" and sample.label == "4" and sample.index == 3
    assert sample.metadata == {"answer": "4", "extra": 1}
    assert prompt_messages("hi") == [{"role": "user", "content": "hi"}]
    assert prompt_messages([{"role": "system", "content": "s"}])[0]["role"] == "system"


def test_default_boxed_reward():
    from types import SimpleNamespace

    from modal_dojo.frameworks.spindle.rewards import boxed_match_reward

    good = SimpleNamespace(label="42", response="<think>x</think> so \\boxed{42}.")
    bad = SimpleNamespace(label="42", response="The answer is 41")
    assert boxed_match_reward(None, good) == 1.0
    assert boxed_match_reward(None, bad) == 0.0
    assert (
        boxed_match_reward(None, SimpleNamespace(label="1,000", response="1000")) == 1.0
    )


def test_spindle_validation_backend():
    import sys

    sys.path.insert(0, "scripts")
    from validation_backends import build_recipe_and_dataset

    recipe, dataset = build_recipe_and_dataset(Framework.SPINDLE, Qwen3_8_27B(), 2)
    assert isinstance(recipe, Qwen3_8_27B_Spindle_Recipe)
    assert dataset.hf_split == f"train[:{recipe.rollout_batch_size * 2}]"
