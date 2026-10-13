"""GLM-5.3-Flash bridge LoRA with a 256k context window."""

from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.miles_recipe.glm_5_3_flash_lora import (
    _PATCH_DIR,
    GLM_5_3_Flash_LoRA_Recipe,
    _image_commands,
)


def _image_commands_256k() -> list[str]:
    return [
        *_image_commands(),
        *[
            f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
            for name in (
                "patch_glm_5_3_flash_kpool_row_chunk",
                "patch_glm_5_3_flash_moe_row_chunk",
                "patch_glm_5_3_flash_moe_seq_chunk",
                "patch_glm_5_3_flash_checkpoint_cpu",
                "patch_glm_5_3_flash_mhc_release",
            )
        ],
    ]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class GLM_5_3_Flash_LoRA_256k_Recipe(GLM_5_3_Flash_LoRA_Recipe):
    """GLM-5.3-Flash rank-16 bridge LoRA for 256k context on 8 H200:8 nodes."""

    # The GLM-5.3 kpool indexer does not support context parallelism, so each
    # TP8 group holds a full sequence; extra nodes add data parallelism.
    image_run_commands: list[str] = field(default_factory=_image_commands_256k)

    actor_num_nodes: int = 8
    expert_model_parallel_size: int = 32

    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    global_batch_size: int = 8
    rollout_max_context_len: int = 262144

    seq_length: int = 262144
    max_tokens_per_gpu: int = 262144

    sglang_context_length: int = 262144
    sglang_mem_fraction_static: float = 0.5
