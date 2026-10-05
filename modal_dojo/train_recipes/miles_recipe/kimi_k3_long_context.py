"""Kimi-K3 LoRA GRPO recipe with a 64k context window."""

from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.train_recipes.miles_recipe.kimi_k3 import Kimi_K3_LoRA_Recipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Kimi_K3_LoRA_Long_Context_Recipe(Kimi_K3_LoRA_Recipe):
    """Kimi-K3 rank-32 LoRA recipe for 64k context on 8 B300:8 nodes."""

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
    sglang_server_concurrency: int | None = 1
    sglang_max_running_requests: int | None = 1
    sglang_max_mamba_cache_size: int = 5
    # One active sequence plus one context window of cache headroom.
    sglang_max_total_tokens: int = 131072
    sglang_cuda_graph_bs_decode: list[int] | None = field(default_factory=lambda: [1])

    # ── Config overrides ─────────────────────────────────────────────────────
    # Disable the initial prompt-length filter.
    extra_config: dict | None = field(
        default_factory=lambda: {"rollout_max_prompt_len": None}
    )
