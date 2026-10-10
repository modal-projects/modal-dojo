"""Kimi-K3 LoRA GRPO recipe with a 64k context window."""

import base64
import zlib
from dataclasses import field

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.train_recipes.miles_recipe.kimi_k3 import (
    Kimi_K3_LoRA_Recipe,
    _PATCH_DIR,
    _image_patches as _base_image_patches,
)

_PATCHES = (
    "patch_sglang_lora_cpu_stash",
    "patch_k3_weight_sync_timing",
    "patch_sglang_freeze_gc",
    "patch_k3_ddp_stream",
    "patch_k3_kernel_warmup",
    "patch_checkpoint_tensor_reads",
    "patch_sglang_offload_timing",
    "patch_k3_marlin_padding",
    "patch_k3_lora_health",
)


def _compressed_patch(name: str) -> str:
    # Keep the recipe below Modal's serialized-function size limit.
    payload = base64.b64encode(
        zlib.compress((_PATCH_DIR / f"{name}.py").read_bytes(), level=9)
    ).decode()
    return (
        f"echo {payload} | base64 -d | python3 -c "
        "'import sys,zlib; exec(zlib.decompress(sys.stdin.buffer.read()))'"
    )


def _image_patches() -> list[str]:
    return [
        *_base_image_patches(),
        *[_compressed_patch(name) for name in _PATCHES],
    ]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Kimi_K3_LoRA_Long_Context_Recipe(Kimi_K3_LoRA_Recipe):
    """Kimi-K3 rank-32 LoRA recipe for 64k context on 8 B300:8 nodes."""

    image_run_commands: list[str] = field(default_factory=_image_patches)
    environment: dict[str, str] = field(
        default_factory=lambda: {
            **Kimi_K3_LoRA_Recipe().environment,
            # Cache autotuning decisions as well as compiled kernels.
            "TRITON_CACHE_AUTOTUNING": "1",
            "TRITON_PRINT_AUTOTUNING": "1",
            # Warm each pipeline stage before the first training microbatch.
            "DOJO_K3_KERNEL_WARMUP": "1",
            # A full unsharded adapter does not fit alongside TP8 inference weights.
            "DOJO_SGLANG_LORA_CPU_STASH": "1",
        }
    )

    # ── Colocation and weight sync ───────────────────────────────────────────
    # Offload the active model before restoring the other to avoid GPU overlap.
    # Release restored inference CPU backups before staging the next adapter.
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

    # ── SGLang: one TP8 engine per node ──────────────────────────────────────
    rollout_num_gpus_per_engine: int = 8
    sglang_mem_fraction_static: float = 0.94
    sglang_context_length: int = 65536
    sglang_chunked_prefill_size: int = 4096
    sglang_server_concurrency: int | None = 1
    sglang_max_running_requests: int | None = 1
    sglang_max_mamba_cache_size: int = 5
    sglang_max_total_tokens: int = 131072
    sglang_decode_attention_backend: str | None = "triton"
    sglang_page_size: int = 1
    sglang_cuda_graph_bs_decode: list[int] | None = field(default_factory=lambda: [1])

    # ── Config overrides ─────────────────────────────────────────────────────
    # Disable the initial prompt-length filter.
    extra_config: dict | None = field(
        default_factory=lambda: {"rollout_max_prompt_len": None}
    )

    @model_validator(mode="after")
    def _keep_image_patches(self) -> "Kimi_K3_LoRA_Long_Context_Recipe":
        patches = _image_patches()
        current = list(self.image_run_commands or [])
        if current[: len(patches)] != patches:
            object.__setattr__(
                self,
                "image_run_commands",
                [*patches, *(c for c in current if c not in patches)],
            )
        return self
