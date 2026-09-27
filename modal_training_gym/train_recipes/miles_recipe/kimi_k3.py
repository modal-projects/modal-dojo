"""Kimi-K3 LoRA GRPO recipe, ported from upstream's ``run_kimi_k3.py``.

Upstream validates the 93-layer release on 16 nodes x 4 GB300 (radixark/miles#1825,
``docs/models/kimi/kimi-k3.md``): TP4 / SP / PP8 / CP2 / EP8 / ETP1 over 64 GPUs,
rollout TP16, rank-32 LoRA at lr 1e-5. Modal has no 4-GPU Blackwell nodes, so
the same 64-GPU layout runs here as 8 nodes x 8 B300; the per-rank shapes are
unchanged and only the per-node host footprint doubles.

Full-parameter training is not offered: the 2.8T-parameter base only fits with
frozen weights, and upstream itself validates the release in LoRA mode only.
"""

from dataclasses import field
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.common.models import Kimi_K3, ModelConfig
from modal_training_gym.common.patches import encode_patch
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe

_PATCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "frameworks"
    / "miles"
    / "modal_helpers"
    / "patches"
)

# Build-time patches; see each script's docstring.
_PATCHES = ("patch_cell_tick_timeout",)


def _image_patches() -> list[str]:
    return [
        f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
        for name in _PATCHES
    ]


# Upstream's validated rollout concurrency per engine. The KDA radix cache needs
# five cache slots per running request under the extra-buffer strategy, and the
# decode graphs are captured only for the batch sizes that concurrency reaches.
_ROLLOUT_MAX_CONCURRENCY = 8


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Kimi_K3_LoRA_Recipe(MilesRecipe):
    """Kimi-K3 rank-32 LoRA recipe for 8 nodes with 8 B300 GPUs each."""

    model_config_class: ClassVar[type[ModelConfig]] = Kimi_K3

    # First multi-arch nightly carrying radixark/miles#1825 together with its
    # Megatron (radixark/Megatron-LM#94) and sglang (sgl-project/sglang#37704)
    # halves; the base pin predates all three.
    docker_image: str = "radixark/miles:dev-202609251434"
    image_run_commands: list[str] = field(default_factory=_image_patches)
    gpu_type: str = "B300"
    # ``lora_base_cpu_backup`` mirrors each rank's frozen base weights into host
    # RAM (~90 GB a rank, 8 ranks a node) so the GPU copy can be released for
    # rollout without re-shipping it every step.
    memory: tuple[int, int] = (1792 * 1024, 2048 * 1024)

    # KDA layers and the attention-residual bank are not representable as a
    # ModelArchitecture, so the launcher renders upstream's
    # scripts/models/kimi-k3.py through model_args_utils.py and passes
    # ${MODEL_ARGS[@]} verbatim — including its --spec for the K3 layer spec.
    miles_model_name: str = "kimi-k3"
    # Selects miles' megatron→HF weight mapping
    # (miles/backends/megatron_utils/megatron_to_hf/kimi_k3.py).
    model_name: str = "kimi_k3"

    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            # transformers copies the tokenizer's remote code into this cache on
            # first import; the default lives on the shared HF Volume, where 32
            # conversion ranks racing to write it read each other's partial
            # files. Container-local, and warmed once per node by the launcher.
            "HF_MODULES_CACHE": "/tmp/hf_modules",
            # Multi-node B300 on Modal has no MNNVL fabric.
            "NCCL_MNNVL_ENABLE": "0",
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_RAS_ENABLE": "0",
            # The TP16 engines span two nodes. sglang's GroupCoordinator
            # broadcasts the NCCL unique id over a gloo CPU group, and gloo
            # picks its address from the hostname, which resolves to 127.0.0.1
            # in Modal containers: every rank waits on the peer's loopback and
            # ncclCommInitRank is never reached. Pin gloo and the TCPStore to
            # eth1, the interface NCCL already bootstraps on (Modal sets
            # NCCL_SOCKET_IFNAME=eth1); eth0 is a per-host network.
            "GLOO_SOCKET_IFNAME": "eth1",
            "TP_SOCKET_IFNAME": "eth1",
            "NCCL_TIMEOUT": "3600",
            # The release packs its routed experts as MXFP4 compressed-tensors
            # and ships no bf16 export; the converter dequantizes them as
            # mbridge reads them instead of staging a ~5.6 TB bf16 checkpoint.
            "CONVERT_DEQUANT_MXFP4": "1",
            "SGLANG_JIT_ROUTE_RADIX": "1",
            # sglang's membind pins the whole host backup to one NUMA node,
            # which cannot hold it.
            "SGLANG_NUMA_BIND_V2": "0",
            # The LoRA wrapper bypasses the o_proj.forward patch the K3
            # all-reduce fusion relies on.
            "SGLANG_K3_AR_FUSION": "0",
        }
    )

    # ── Checkpoints ──────────────────────────────────────────────────────────
    megatron_to_hf_mode: str = "raw"
    ref_load: str = "/checkpoints/Kimi-K3_torch_dist"
    # Upstream converts on 32 ranks at TP32/EP32 (4 GB300 hosts, 8 ranks each).
    # The torch_dist save stages each rank's bf16 shard through host RAM, and
    # at 32 ranks that is ~1.4 TB a node: the node died at 1068 GiB RSS. EP64
    # spreads the 896 experts over all 8 nodes (~0.7 TB a node) and the DP
    # replicas the wider expert world needs. torch_dist re-shards at load, so
    # training at TP4/PP8/EP8 reads it fine.
    conversion_tensor_model_parallel_size: int = 32
    conversion_pipeline_model_parallel_size: int = 1
    conversion_expert_model_parallel_size: int = 64
    conversion_expert_tensor_parallel_size: int = 1
    # Each conversion node stages its eighth of the ~5.6 TB bf16 torch_dist
    # checkpoint on local disk before the Volume commits it.
    convert_ephemeral_disk_mb: int | None = 2 * 1024 * 1024
    # 1.5 TB in, ~5.6 TB out: neither fits the launcher's 4-hour stage default.
    download_timeout_seconds: int | None = 8 * 60 * 60
    convert_timeout_seconds: int | None = 12 * 60 * 60

    # ── Cluster ──────────────────────────────────────────────────────────────
    actor_num_nodes: int = 8
    actor_num_gpus_per_node: int = 8

    # ── Parallelism: TP4 x CP2 x PP8 = 64 (DP1), EP8 inside each stage ───────
    tensor_model_parallel_size: int = 4
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 8
    # 93 = 12 x 7 + 9 across 8 stages.
    decoder_first_pipeline_num_layers: int | None = 12
    decoder_last_pipeline_num_layers: int | None = 9
    context_parallel_size: int = 2
    expert_model_parallel_size: int = 8

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_rank: int | None = 32
    lora_alpha: int | None = 64
    # Resolves to Kimi-K3's HF defaults: attention output, both MLA
    # down-projections, the dense MLP and both routed-expert projections.
    # The pinned miles validates targets against HF names, so upstream's
    # older Megatron-style module list is rejected at parse time.
    target_modules: str | None = "all-linear"
    # One A factor shared across the 896 routed experts, per-expert B factors.
    experts_shared_outer_loras: bool = True
    lora_base_cpu_backup: bool = True
    no_gradient_accumulation_fusion: bool = True

    # ── Rollout and sampling ─────────────────────────────────────────────────
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    use_dynamic_global_batch_size: bool = True
    balance_data: bool = True
    use_miles_router: bool = True
    skip_eval_before_train: bool = True

    # ── Training ─────────────────────────────────────────────────────────────
    recompute_granularity: str | None = "full"
    recompute_method: str | None = "uniform"
    recompute_num_layers: int | None = 1
    max_tokens_per_gpu: int = 8192
    log_probs_chunk_size: int = 512
    distributed_timeout_minutes: int = 60

    # ── Optimizer ────────────────────────────────────────────────────────────
    # Upstream's validated LoRA learning rate; eval/aime rose 0.367 -> 0.467
    # over the first ten evals with it.
    lr: float = 1e-5
    use_distributed_optimizer: bool = True
    optimizer_cpu_offload: bool = True
    optimizer_offload_fraction: float = 0.8
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # ── Colocation and weight sync ───────────────────────────────────────────
    offload_train: bool = True
    # Upstream's resolved config for this recipe shows ``gpu``; the miles
    # default is ``cpu``. ``gpu`` onloads the other side first so the trainer
    # and the engine briefly coexist in GPU memory rather than host memory,
    # which is the tighter budget here (8 ranks' backups a node).
    colocate_memory_peak_device: str = "gpu"
    # Only the adapter crosses to the engines each step, so the sync buffer is
    # small; the base stays resident in the engines.
    update_weight_buffer_size: int | None = 256 * 1024**2
    train_memory_margin_bytes: int = 4 * 1024**3
    # The trainer has no vision tower, and the MXFP4 experts round-trip
    # through bf16 on every sync.
    check_weight_update_skip_list: list[str] = field(
        default_factory=lambda: ["vision_tower.", "mm_projector."]
    )
    check_weight_update_allow_quant_error: bool = True

    # ── SGLang: one TP16 engine spans two nodes, experts replicated over TP ──
    rollout_num_gpus_per_engine: int = 16
    sglang_server_concurrency: int | None = 16
    sglang_max_running_requests: int | None = _ROLLOUT_MAX_CONCURRENCY
    sglang_max_mamba_cache_size: int = 5 * _ROLLOUT_MAX_CONCURRENCY
    sglang_max_total_tokens: int = 65536
    # Marlin is the one MXFP4 MoE runner with a LoRA path; the skill's
    # ``triton`` guidance is for INT4 checkpoints, where marlin's LoRA path
    # faults capturing decode graphs.
    sglang_moe_runner_backend: str | None = "marlin"
    sglang_lora_backend: str | None = "triton"
    sglang_lora_strict_loading: bool = True
    sglang_max_lora_rank: int = 32
    sglang_max_loras_per_batch: int = 1
    # Above TP8 the Marlin MoE intermediate is tile-padded, which the
    # virtual-experts LoRA kernel rejects.
    no_sglang_lora_use_virtual_experts: bool = True
    # The adapter is re-streamed every step; a host copy per TP rank (~45 GB)
    # would never be read.
    sglang_lora_no_cpu_backup: bool = True
    sglang_decode_attention_backend: str | None = "trtllm_mla"
    sglang_mamba_radix_cache_strategy: str | None = "extra_buffer"
    sglang_cuda_graph_bs_decode: list[int] | None = field(
        default_factory=lambda: [1, 2, 4, 8]
    )
    sglang_cuda_graph_backend_prefill: str | None = "disabled"
    # Each of the four engines reads the whole 1.5 TB MXFP4 release off the
    # Volume and repacks it for Marlin before it answers a health check; the
    # 14 GB prune loaded at ~1 GB/s per engine.
    rollout_health_check_first_wait: int = 7200

    @model_validator(mode="after")
    def _keep_image_patches(self) -> "Kimi_K3_LoRA_Recipe":
        """Keep the build-time patches at the head of ``image_run_commands``.

        The field is replaced wholesale, so a caller adding their own command
        would otherwise drop the tick-timeout patch and the engines' first
        memory release would time out again.
        """
        patches = _image_patches()
        current = list(self.image_run_commands or [])
        if current[: len(patches)] != patches:
            object.__setattr__(
                self,
                "image_run_commands",
                [*patches, *(c for c in current if c not in patches)],
            )
        return self
