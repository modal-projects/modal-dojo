"""Kimi-K3 LoRA GRPO recipe with a 256k context window on H200."""

from dataclasses import field

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.miles_recipe.kimi_k3 import (
    _PATCH_DIR,
    _PATCHES as _BASE_PATCHES,
)
from modal_dojo.train_recipes.miles_recipe.kimi_k3_64k import (
    Kimi_K3_LoRA_64k_Recipe,
    _PATCHES as _64K_PATCHES,
    _compressed_patch,
)

# Bound the PP adapter broadcast by update_weight_buffer_size.
_BASE_PATCH_OVERRIDES = {
    "patch_lora_sync_stream_pp": "patch_lora_sync_stream_pp_bounded"
}

_PATCHES = (
    *_64K_PATCHES,
    "patch_k3_marlin_padding",
)


def _image_patches() -> list[str]:
    return [
        *[
            _compressed_patch(_BASE_PATCH_OVERRIDES[name])
            if name in _BASE_PATCH_OVERRIDES
            else f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
            for name in _BASE_PATCHES
        ],
        *[_compressed_patch(name) for name in _PATCHES],
    ]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Kimi_K3_LoRA_256k_H200_Recipe(Kimi_K3_LoRA_64k_Recipe):
    """Kimi-K3 rank-32 LoRA recipe for 256k context on 16 H200:8 nodes."""

    image_run_commands: list[str] = field(default_factory=_image_patches)
    gpu_type: str = "H200"
    # H200 hosts have 2 TiB: inference backups plus a 1/16 trainer shard.
    memory: tuple[int, int] = (1792 * 1024, 1920 * 1024)

    # ── Cluster ──────────────────────────────────────────────────────────────
    actor_num_nodes: int = 16

    # ── Parallelism: TP8 x CP4 x PP4 = 128 (DP1), EP32 inside each stage ─────
    tensor_model_parallel_size: int = 8
    pipeline_model_parallel_size: int = 4
    # 93 = 23 + 24 x 2 + 22 across 4 stages.
    decoder_first_pipeline_num_layers: int | None = 23
    decoder_last_pipeline_num_layers: int | None = 22
    # TP8 x CP4 keeps 8k tokens per rank, matching the 64k recipe.
    context_parallel_size: int = 4
    expert_model_parallel_size: int = 32

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer_offload_fraction: float = 1.0

    # ── Rollout and sampling ─────────────────────────────────────────────────
    rollout_max_response_len: int = 262144
    eval_max_response_len: int = 262144

    # ── Training ─────────────────────────────────────────────────────────────
    seq_length: int = 262144
    max_position_embeddings: int = 262144
    # CP4 splits each 256k sequence across four ranks.
    max_tokens_per_gpu: int = 65536
    log_probs_max_tokens_per_gpu: int = 65536

    # ── SGLang: one TP16 engine per two nodes ────────────────────────────────
    rollout_num_gpus_per_engine: int = 16
    sglang_mem_fraction_static: float = 0.95
    sglang_context_length: int = 262144
    sglang_max_total_tokens: int = 524288

    @model_validator(mode="after")
    def _keep_image_patches(self) -> "Kimi_K3_LoRA_256k_H200_Recipe":
        patches = _image_patches()
        current = list(self.image_run_commands or [])
        if current[: len(patches)] != patches:
            object.__setattr__(
                self,
                "image_run_commands",
                [*patches, *(c for c in current if c not in patches)],
            )
        return self
