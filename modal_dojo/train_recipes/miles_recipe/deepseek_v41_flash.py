"""DeepSeek-V4.1-Flash GRPO recipe, ported from upstream's ``run_deepseek_v41.py``."""

from dataclasses import field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import DeepSeek_V4_1_Flash, ModelConfig
from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

# radixark/miles#3179 (DeepSeek-V4.1 RL support) is unmerged, and the only image
# built from it, `radixark/miles:deepseek-v41`, is arm64/GB300-only — Modal's
# builder resolves amd64 and rejects it. That image is the last amd64+arm64
# nightly built on `lmsysorg/sglang:v0.5.18` with two source trees copied over
# it: miles at the PR head, and an sglang tree that merges sgl-project/sglang#38798
# (V4.1 engine support) into `sglang-miles` (the weight-update session API miles
# needs) and exists nowhere public but inside that image. So this recipe starts
# from the same v0.5.18 nightly for amd64, checks out the miles PR, and unpacks
# the sglang tree from the published image's layer
# (patch_deepseek_v41_sglang_tree). Replace all of it with a single published tag
# once the PR lands in a nightly.
_MILES_PR = "pull/3179/head"  # 6a54b4629c4259f4733990a7fcd6c77c3c56296e

# Local write buffer for the params-only save (~1.1 TB bf16 over 8 nodes, so
# ~140 GB a node) on top of the CPU-offloaded optimizer's own host usage.
_TRAIN_EPHEMERAL_DISK_MIB = 768 * 1024

_PATCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "frameworks"
    / "miles"
    / "modal_helpers"
    / "patches"
)

# Build-time source overlay and shims for gaps in it; see each script's
# docstring. They run after ``apply_source_overlays`` so the sglang tree lands
# over the nightly's checkout and the edits land on that tree.
_PATCHES = (
    "patch_deepseek_v41_sglang_tree",
    "patch_deepseek_v41_processor_tokenizer",
    "patch_deepseek_v41_fp8_hopper_gemm",
    "patch_deepseek_v41_chat_template",
    "patch_deepseek_v41_fp4_dequant_block",
    "patch_deepseek_v41_vision_topk_capture",
)


def _image_patches() -> list[str]:
    return [
        f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
        for name in _PATCHES
    ]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class DeepSeek_V4_1_Flash_Recipe(MilesRecipe):
    """DeepSeek-V4.1-Flash GRPO recipe for 8 nodes with 8 H200 GPUs each."""

    model_config_class: ClassVar[type[ModelConfig]] = DeepSeek_V4_1_Flash

    docker_image: str = "radixark/miles:dev-202609050049"
    miles_git_ref: str | None = _MILES_PR
    image_run_commands: list[str] = field(default_factory=_image_patches)
    gpu_type: str = "H200"
    # The fp32 optimizer state is offloaded to host RAM: ~8.5B params per GPU at
    # 12 bytes each is ~100 GiB per rank.
    memory: tuple[int, int] = (1024, int(2 * 1024 * 1024))

    # V4.1's DSA indexer, CSA compression and Engram memory are not
    # representable as a ModelArchitecture, so the launcher renders upstream's
    # scripts/models/deepseek-v4.1.py through model_args_utils.py and passes
    # ${MODEL_ARGS[@]} verbatim — including its --spec for the V4.1 layer spec.
    miles_model_name: str = "deepseek-v4.1"
    # Selects miles' megatron→HF weight mapping (miles/backends/megatron_utils/
    # megatron_to_hf/deepseekv41.py).
    model_name: str = "deepseekv41"
    # Selects the miles (not sglang) V4.1 training implementation.
    dsv4_impl: str = "miles"

    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            # Pin the conversion to the pipeline layout below; the converter
            # otherwise inflates PP to the rank count.
            "CONVERT_KEEP_PP1": "1",
            # The HF release is fp8 dense / packed-fp4 experts with e8m0 block
            # scales and no bf16 export; mbridge expects bf16, so the conversion
            # wrapper dequantizes weights as it reads them.
            "CONVERT_DEQUANT_HF_WEIGHTS": "1",
            "NCCL_CUMEM_ENABLE": "1",
            "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
            # Upstream serves an FP4→FP8 pre-converted checkpoint
            # (SGLANG_DSV4_FP4_EXPERTS=0). We load the public release, whose
            # routed experts are packed mxfp4, so the engine must expect FP4
            # and dequantize them to FP8 at load: the mxfp4 MoE kernels are
            # Blackwell-only, the dequantized path runs on H200.
            "SGLANG_DSV4_FP4_EXPERTS": "1",
            "SGLANG_DSV4_FP4_DEQUANT": "1",
            # The rest guard against slow/absent kernels and a load that
            # outlives the default health-check window.
            "SGLANG_HEALTH_CHECK_TIMEOUT": "900",
            "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
            "SGLANG_OPT_FP8_WO_A_GEMM": "0",
            "SGLANG_OPT_FUSE_WQA_WKV": "0",
            "SGLANG_DISABLE_MULTIMEM_AG": "1",
            "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "0",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "PYTHONFAULTHANDLER": "1",
            # Paired with --deterministic-mode below.
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )

    # ── Checkpoints ──────────────────────────────────────────────────────────
    # "raw" turns on the HF → torch_dist conversion that upstream runs as
    # `run_deepseek_v41.py prepare-spmd`; miles then loads it as ref_load.
    megatron_to_hf_mode: str = "raw"
    ref_load: str = "/checkpoints/DeepSeek-V4.1-Flash_torch_dist"

    # Upstream's conversion layout (TP4, PP1) over two nodes instead of one: the
    # converter builds the model on GPU and the bf16 weights are ~970 GB, which
    # is ~121 GiB a rank at EP8 — fine on their 288 GiB GB300s, an OOM on an
    # H200. EP16 halves it, served by data-parallel replicas; TP stays at 4
    # since the 129280 vocab only splits 8 ways with a smaller
    # make-vocab-size-divisible-by. torch_dist reshards on load, so this layout
    # is independent of the training one below.
    conversion_tensor_model_parallel_size: int = 4
    conversion_pipeline_model_parallel_size: int = 1
    conversion_expert_model_parallel_size: int = 16
    conversion_expert_tensor_parallel_size: int = 1
    # The bf16 torch_dist checkpoint is ~1.1 TB, and a Volume buffers writes on
    # container-local disk before committing them.
    convert_ephemeral_disk_mb: int | None = 2048 * 1024

    # ── Cluster + parallelism ────────────────────────────────────────────────
    # ~560B total parameters: bf16 weights and grads alone are ~2.2 TB, so the
    # expert dimension is sharded across every rank (upstream's default
    # ep_size = actor_num_nodes * gpus_per_node / pp_size).
    actor_num_nodes: int = 8
    actor_num_gpus_per_node: int = 8
    tensor_model_parallel_size: int = 4
    sequence_parallel: bool = True
    expert_model_parallel_size: int = 64

    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    transformer_impl: str = "transformer_engine"
    bf16: bool = True
    qkv_format: str = "bshd"
    # V4.1's router gate and its expert-bias correction term are frozen during RL.
    moe_router_freeze_gate: bool = True
    freeze_e_score_correction_bias: bool = True
    # Bit-exact recomputation, so a rollout's log probs match the training pass.
    deterministic_mode: bool = True
    train_memory_margin_bytes: int = 3 * 1024**3
    colocate_memory_peak_device: str = "cpu"
    update_weight_buffer_size: int = 1024**3

    # ── Rollout + reward ─────────────────────────────────────────────────────
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    # One train step per rollout, so the global batch is the rollout itself.
    global_batch_size: int = 128
    # Dynamic packing conflicts with --qkv-format bshd upstream.
    use_dynamic_batch_size: bool = False
    rollout_temperature: float = 0.8
    max_tokens_per_gpu: int = 2048
    micro_batch_size: int = 1
    skip_eval_before_train: bool = True
    # R3: replay the rollout's routed expert ids during training so train and
    # inference routing agree.
    use_rollout_routing_replay: bool = True

    # ── Optimizer + GRPO ─────────────────────────────────────────────────────
    use_distributed_optimizer: bool = True
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True
    # A Volume buffers writes on container-local disk before committing them, and
    # the fp32 master weights plus Adam moments for 560B params are ~7 TB on top
    # of the ~1.1 TB bf16 params — over what a training node can stage, and the
    # writer dies with the zip writer's "unexpected pos" rather than ENOSPC.
    # Save params only; a resumed run restarts the Adam moments. no_load_optim
    # must be paired with it or Megatron's load dies with KeyError: 'optimizer'.
    no_save_optim: bool = True
    no_load_optim: bool = True
    train_function_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"ephemeral_disk": _TRAIN_EPHEMERAL_DISK_MIB}
    )

    # ── SGLang ───────────────────────────────────────────────────────────────
    # One engine per node, expert-parallel across it.
    rollout_num_gpus_per_engine: int = 8
    sglang_ep_size: int = 8
    sglang_dp_size: int = 1
    sglang_attention_backend: str = "dsv4"
    sglang_moe_runner_backend: str = "auto"
    # The loaded engine weights (dequantized FP8 experts, sharded engram tables)
    # occupy ~112 GB of each H200; upstream's 0.6 leaves no room for a KV pool
    # here, and the KV pool a smoke step needs is small.
    sglang_mem_fraction_static: float = 0.9
    sglang_max_running_requests: int = 128
    sglang_disable_cuda_graph: bool = True
    sglang_disable_radix_cache: bool = True

    # miles always renders ``device`` on the engine command line, and sglang
    # builds from 0.5.19 on no longer fill in an unset one, so name it rather
    # than depend on the nightly's auto-detect.
    #
    # Each TP rank reads all 48 ~10 GB HF shards while the co-located Megatron
    # ranks hold their offloaded weights in host RAM. sglang's default loader
    # keeps 9 mmap'd shards in flight per rank, and the sandbox counts those
    # pages against the container, so 8 ranks can exhaust the 2 TB node and get
    # a scheduler SIGKILLed mid-load. Narrow the window to 3 shards and drop
    # each from the page cache once it is copied to the GPU. (mmap stays on:
    # the eager `safetensors.torch.load` path cannot decode V4.1's F8_E8M0
    # scale tensors.)
    extra_config: dict | None = field(
        default_factory=lambda: {
            "sglang_device": "cuda",
            "sglang_weight_loader_drop_cache_after_load": True,
            "sglang_model_loader_extra_config": '{"num_threads": 2}',
        }
    )

    rollout_health_check_interval: int = 300
    rollout_health_check_timeout: int = 300
    # The first engine start compiles deepgemm kernels for a 560B MoE; without a
    # long grace period the health checker kills the engines mid-warmup.
    rollout_health_check_first_wait: int = 3600

    @model_validator(mode="after")
    def _keep_disk_reservation(self) -> "DeepSeek_V4_1_Flash_Recipe":
        """Keep the reservation when a caller supplies their own kwargs.

        Passing ``{"secrets": [...]}`` would otherwise drop it and the save would
        exhaust local disk. A caller who names ``ephemeral_disk`` wins.
        """
        kwargs = self.train_function_kwargs or {}
        if "ephemeral_disk" not in kwargs:
            object.__setattr__(
                self,
                "train_function_kwargs",
                {"ephemeral_disk": _TRAIN_EPHEMERAL_DISK_MIB, **kwargs},
            )
        return self

    @model_validator(mode="after")
    def _keep_image_patches(self) -> "DeepSeek_V4_1_Flash_Recipe":
        """Keep the build-time patches at the head of ``image_run_commands``.

        The field is replaced wholesale, so a caller adding their own command
        would otherwise drop the chat-template shim and every prompt would fail
        to render.
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
