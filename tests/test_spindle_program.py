"""Custom Tinker programs through SpindleRecipe."""

import pytest

from modal_dojo.common.errors import DojoConfigError

MODEL = "Qwen/Qwen3.5-2B"


def test_custom_program_uses_backend_requirements_and_no_implicit_retries(monkeypatch):
    pytest.importorskip("tinker_cookbook")
    from modal_dojo import SpindleRecipe
    from modal_dojo.frameworks.spindle.app import _dataset_name
    from modal_dojo.frameworks.spindle.reporting import new_training_run

    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    class Program:
        model_name = MODEL

        async def run(self, client, ctx):
            pass

    program = Program()
    recipe = SpindleRecipe(program, context_length=32768, trainer_gpu="H100")
    assert recipe.program is program
    assert recipe.max_retries == 0
    assert recipe.requirements.model_name == MODEL
    assert recipe.requirements.context_length == 32768
    assert recipe.requirements.overrides == {"trainer_gpu": "H100"}
    assert recipe.requirements.max_tokens == 0
    assert _dataset_name(program) == "Program"
    run = new_training_run(program, "run-test", "/checkpoints/run-test")
    assert run.source_model["model_name"] == MODEL
    assert "program" in run.config["recipe"]
    assert SpindleRecipe(program, max_retries=2).max_retries == 2

    class SynchronousProgram:
        model_name = MODEL

        def run(self, client, ctx):
            pass

    with pytest.raises(DojoConfigError, match="async method"):
        SpindleRecipe(SynchronousProgram())
