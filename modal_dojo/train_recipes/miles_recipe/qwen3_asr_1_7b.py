from collections.abc import Callable
from dataclasses import field
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import ModelConfig, Qwen3_ASR_1_7B
from modal_dojo.common.patches import encode_patch
from modal_dojo.frameworks.miles.audio_transcription_rollout import (
    transcription_rollout,
)
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

_ASR_PATCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "frameworks"
    / "miles"
    / "modal_helpers"
    / "patches"
)
_ASR_PATCHES = (
    "patch_qwen3_asr_bridge_config",
    "patch_qwen3_asr_processor",
    "patch_qwen3_asr_pg_collection",
    "patch_qwen3_asr_packed_seq",
    "patch_qwen3_asr_export",
)


def _asr_image_run_commands() -> list[str]:
    cmds = [
        "pip install --no-cache-dir jiwer==4.0.0 librosa==0.11.0 soundfile==0.14.0 "
        '"numpy<2"'
    ]
    cmds += [
        f"echo {encode_patch(name, _ASR_PATCH_DIR)} | base64 -d | python3"
        for name in _ASR_PATCHES
    ]
    return cmds


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_ASR_1_7B_Recipe(MilesRecipe):
    """Qwen3-ASR-1.7B recipe."""

    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_ASR_1_7B
    sft_supported: ClassVar[bool] = False

    recompute_granularity: str | None = "full"

    custom_generate_function: Callable | None = transcription_rollout
    sglang_mem_fraction_static: float = 0.45
    lr_decay_style: str = "cosine"

    use_dynamic_batch_size: bool = False
    extra_config: dict | None = field(
        default_factory=lambda: {"qkv_format": "bshd", "micro_batch_size": 1}
    )

    image_run_commands: list[str] = field(default_factory=_asr_image_run_commands)
