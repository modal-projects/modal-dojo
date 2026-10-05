"""Kimi-K3 LoRA GRPO recipe with a 64k context window."""

from dataclasses import field

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.miles_recipe.kimi_k3 import (
    Kimi_K3_LoRA_Recipe,
    _PATCH_DIR,
    _image_patches as _base_image_patches,
)

_PATCHES = (
    "patch_sglang_offload_timing",
    "patch_k3_marlin_padding",
    "patch_k3_lora_health",
)


def _image_patches() -> list[str]:
    return [
        *_base_image_patches(),
        *[
            f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
            for name in _PATCHES
        ],
        # Retain pinned CPU allocations using the image's allocator revision.
        "git clone https://github.com/fzyzcjy/torch_memory_saver.git /tmp/dojo-tms "
        "&& git -C /tmp/dojo-tms checkout b5588e83de86412a48689a6583a4b567e75f7acc",
        f"echo {encode_patch('patch_tms_retain_backup', _PATCH_DIR)} | base64 -d | python3",
        "TMS_CUDA_MAJOR=13 uv pip install --python /opt/sglang/bin/python "
        "--no-deps --no-build-isolation --reinstall /tmp/dojo-tms",
    ]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Kimi_K3_Long_Context_Recipe(Kimi_K3_LoRA_Recipe):
    """Kimi-K3 rank-32 LoRA recipe for 64k context on 8 B300:8 nodes."""

    image_run_commands: list[str] = field(default_factory=_image_patches)
    environment: dict[str, str] = field(
        default_factory=lambda: {
            **Kimi_K3_LoRA_Recipe().environment,
            # Cache autotuning decisions as well as compiled kernels.
            "TRITON_CACHE_AUTOTUNING": "1",
            "TRITON_PRINT_AUTOTUNING": "1",
            # Reuse host allocations while copying fresh weights on every pause.
            "DOJO_TMS_RETAIN_BACKUP_TAG": "weights",
        }
    )

    # ── Colocation and weight sync ───────────────────────────────────────────
    # Offload the active model before restoring the other to avoid GPU overlap.
    colocate_memory_peak_device: str = "cpu"
    memory: tuple[int, int] = (2560 * 1024, 3072 * 1024)

    # ── Rollout and sampling ─────────────────────────────────────────────────
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    global_batch_size: int = 8
    rollout_max_response_len: int = 65536
    eval_max_response_len: int = 65536

    # ── Training ─────────────────────────────────────────────────────────────
    seq_length: int = 65536
    max_position_embeddings: int = 65536
    # CP2 splits each 64k sequence across two ranks.
    max_tokens_per_gpu: int = 32768
    log_probs_max_tokens_per_gpu: int = 32768

    # ── SGLang: one TP16 engine spans two nodes ──────────────────────────────
    sglang_context_length: int = 65536
    sglang_chunked_prefill_size: int = 4096
    sglang_server_concurrency: int | None = 2
    sglang_max_running_requests: int | None = 2
    sglang_max_mamba_cache_size: int = 10
    # Two active sequences plus one context window of cache headroom.
    sglang_max_total_tokens: int = 196608
    sglang_cuda_graph_bs_decode: list[int] | None = field(
        default_factory=lambda: [1, 2]
    )

    # ── Config overrides ─────────────────────────────────────────────────────
    # Disable the initial prompt-length filter.
    extra_config: dict | None = field(
        default_factory=lambda: {"rollout_max_prompt_len": None}
    )

    @model_validator(mode="after")
    def _keep_image_patches(self) -> "Kimi_K3_Long_Context_Recipe":
        patches = _image_patches()
        current = list(self.image_run_commands or [])
        if current[: len(patches)] != patches:
            object.__setattr__(
                self,
                "image_run_commands",
                [*patches, *(c for c in current if c not in patches)],
            )
        return self
