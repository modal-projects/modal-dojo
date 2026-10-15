from __future__ import annotations

import json
import warnings

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HFValidationError, HfHubHTTPError
from huggingface_hub.utils import validate_repo_id

from modal_dojo.common.models.base import ModelArchitecture, ModelConfig
from modal_dojo.train_recipes.base import BaseTrainRecipe
from modal_dojo.train_recipes.gpu_allocation import gpu_memory_gib

GIB = 1024**3

_UNMODELED_CFG_HINTS = ("engram", "index_", "dspark_", "nextn", "vision")


def _hf_cfg(model_name: str) -> dict | None:
    try:
        validate_repo_id(model_name)
        with open(hf_hub_download(repo_id=model_name, filename="config.json")) as f:
            cfg = json.load(f)
    except (
        HFValidationError,
        HfHubHTTPError,
        EntryNotFoundError,
        json.JSONDecodeError,
    ):
        return None
    if isinstance(cfg.get("text_config"), dict):
        cfg = {**cfg, **cfg["text_config"]}
    return cfg


def _arch_from_hf(model_name: str, cfg: dict | None = None) -> ModelArchitecture | None:
    cfg = cfg if cfg is not None else _hf_cfg(model_name)
    if cfg is None:
        return None
    layers, hidden, heads = (
        int(cfg.get(k) or 0)
        for k in ("num_hidden_layers", "hidden_size", "num_attention_heads")
    )
    if not (layers and hidden and heads):
        return None
    ffn = int(cfg.get("intermediate_size") or 0)
    experts = int(
        cfg.get("n_routed_experts")
        or cfg.get("num_experts")
        or cfg.get("num_local_experts")
        or 0
    )
    moe_ffn = int(cfg.get("moe_intermediate_size") or ffn)
    shared = int(cfg.get("shared_expert_intermediate_size") or 0) or (
        int(cfg.get("n_shared_experts") or 0) * moe_ffn
    )
    dense_first = int(cfg.get("first_k_dense_replace") or 0)
    freq = (
        f"[0]*{dense_first}+[1]*{layers - dense_first}"
        if experts and dense_first
        else ""
    )
    return ModelArchitecture(
        num_layers=layers,
        hidden_size=hidden,
        ffn_hidden_size=ffn,
        num_attention_heads=heads,
        num_query_groups=int(cfg.get("num_key_value_heads") or heads),
        kv_channels=int(cfg.get("head_dim") or hidden // heads),
        vocab_size=int(cfg.get("vocab_size") or 0),
        untie_embeddings_and_output_weights=not cfg.get("tie_word_embeddings", True),
        num_experts=experts,
        moe_ffn_hidden_size=moe_ffn,
        moe_shared_expert_intermediate_size=shared,
        moe_layer_freq=freq,
        multi_latent_attention=bool(
            cfg.get("kv_lora_rank") and cfg.get("qk_rope_head_dim")
        ),
        kv_lora_rank=int(cfg.get("kv_lora_rank") or 0),
        q_lora_rank=int(cfg.get("q_lora_rank") or 0),
        qk_head_dim=int(cfg.get("qk_nope_head_dim") or 0),
        qk_pos_emb_head_dim=int(cfg.get("qk_rope_head_dim") or 0),
        v_head_dim=int(cfg.get("v_head_dim") or 0),
    )


def _param_counts(arch: ModelArchitecture) -> tuple[int, int]:
    """Scalar parameter counts as (dense, experts) element totals."""
    h, heads = arch.hidden_size, max(1, arch.num_attention_heads)
    kv, nqg = arch.kv_channels or h // heads, arch.num_query_groups or heads
    if arch.multi_latent_attention:
        qk = arch.qk_head_dim or kv
        pos = arch.qk_pos_emb_head_dim
        v = arch.v_head_dim or qk
        q = (
            h * arch.q_lora_rank + arch.q_lora_rank * heads * (qk + pos)
            if arch.q_lora_rank
            else h * heads * (qk + pos)
        )
        attn = (
            q
            + h * (arch.kv_lora_rank + pos)
            + arch.kv_lora_rank * heads * (qk + v)
            + heads * v * h
        )
    else:
        attn = h * heads * kv * 2 + h * nqg * kv * 2
    gates = 3 if arch.swiglu else 2
    if not arch.num_experts:
        moe_n = 0
    elif not arch.moe_layer_freq:
        moe_n = arch.num_layers
    else:
        freq = eval(arch.moe_layer_freq, {"__builtins__": {}})
        if isinstance(freq, int):
            # Megatron: layer i is MoE iff i % freq == 0.
            moe_n = len(range(0, arch.num_layers, freq)) if freq else 0
        else:
            moe_n = sum(1 for x in freq if x)
    untie = 2 if arch.untie_embeddings_and_output_weights else 1
    dense = arch.vocab_size * h * untie
    dense += (arch.num_layers - moe_n) * (attn + gates * h * arch.ffn_hidden_size)
    dense += moe_n * (
        attn + gates * h * (arch.moe_shared_expert_intermediate_size or 0)
    )
    experts = moe_n * arch.num_experts * gates * h * (arch.moe_ffn_hidden_size or 0)
    return dense, experts


def _peak_gib(
    arch: ModelArchitecture, knobs: dict[str, object], gpu_gib: float
) -> tuple[float, dict[str, object]]:
    get = knobs.get
    raised: dict[str, object] = {}

    def raise_(name: str, value: object, when: bool = True) -> None:
        if when:
            raised[name] = value

    def parallel(name: str, *, applicable: bool = True) -> int:
        value = int(get(name) or 1)
        raise_(name, value, applicable and value == 1)
        return value

    dense, experts = _param_counts(arch)
    h = arch.hidden_size

    gpus_per_node = int(get("actor_num_gpus_per_node") or 1)
    world = int(get("actor_num_nodes") or 1) * gpus_per_node
    tp = parallel("tensor_model_parallel_size", applicable=gpus_per_node > 1)
    pp = parallel("pipeline_model_parallel_size", applicable=world > 1)
    cp = parallel("context_parallel_size", applicable=world > 1)
    ep = parallel(
        "expert_model_parallel_size",
        applicable=bool(arch.num_experts) and world > 1,
    )
    etp = int(get("expert_tensor_parallel_size") or 1)
    dense_p, expert_p = dense / (tp * pp), experts / (ep * etp * pp)

    if get("lora_rank"):
        # Frozen bf16 base; adapter weights/grads/optimizer state are negligible.
        states = (dense_p + expert_p) * 2 / GIB
    else:
        offload = get("optimizer_cpu_offload")
        raise_("optimizer_cpu_offload", offload, not offload)
        fraction = get("optimizer_offload_fraction")
        fraction = 1.0 if fraction is None else min(1.0, max(0.0, float(fraction)))
        raise_("optimizer_offload_fraction", fraction, offload and fraction < 1.0)
        opt = 12.0 * (1.0 - fraction) if offload else 12.0
        d_div = e_div = 1.0
        if get("use_distributed_optimizer"):
            d_div = max(1, world // (tp * pp))
            e_div = max(1, world // (ep * etp * pp))
        states = (
            (dense_p + expert_p) * 6 + dense_p * opt / d_div + expert_p * opt / e_div
        ) / GIB

    ctx = get("rollout_max_context_len")
    prompt = get("rollout_max_prompt_len")
    response = get("rollout_max_response_len")
    sample = int(ctx or 0) or (int(prompt or 0) + int(response or 0))

    def raise_sample_len() -> None:
        if ctx:
            raise_("rollout_max_context_len", ctx)
        else:
            raise_("rollout_max_prompt_len", prompt, bool(prompt))
            raise_("rollout_max_response_len", response, bool(response))

    if get("use_dynamic_batch_size"):
        mtp = int(get("max_tokens_per_gpu") or 0)
        tokens = max(mtp, sample / cp)
        if mtp >= sample / cp:
            raise_("max_tokens_per_gpu", get("max_tokens_per_gpu"))
        else:
            raise_sample_len()
    else:
        mbs = int(get("micro_batch_size") or 1)
        tokens = mbs * sample / cp
        raise_("micro_batch_size", get("micro_batch_size"), mbs > 1)
        raise_sample_len()

    recompute = get("recompute_granularity")
    raise_("recompute_granularity", recompute, recompute != "full")
    layer = tokens * h * 34 / tp  # 34 = activation bytes per token per layer
    acts = (
        tokens * h * 2 * arch.num_layers / tp + layer
        if recompute == "full"
        else layer * arch.num_layers
    ) / GIB
    # bf16 logits + fp32 copy saved by the fused cross-entropy + bf16 logits grad.
    logits = tokens * arch.vocab_size / tp * 8 / GIB

    colocate = get("colocate")
    no_rollout = get("no_offload_rollout")
    no_train = get("no_offload_train")
    sglang = 0.0
    if colocate and (no_rollout or no_train):
        raise_("colocate", colocate)
        raise_("no_offload_rollout", no_rollout, bool(no_rollout))
        raise_("no_offload_train", no_train, bool(no_train))
        sglang = float(get("sglang_mem_fraction_static") or 0) * gpu_gib

    return states + acts + logits + sglang, raised


def maybe_warn_gpu_oom(recipe: BaseTrainRecipe, model: ModelConfig) -> None:
    if hasattr(recipe, "train_backend") and recipe.train_backend != "megatron":
        return
    gpu_gib = gpu_memory_gib(recipe.gpu_type)
    if gpu_gib is None:
        return
    arch = model.architecture
    unmodeled: list[str] = []
    if arch is None:
        cfg = _hf_cfg(model.model_name)
        if cfg is None:
            return
        arch = _arch_from_hf(model.model_name, cfg)
        if arch is None:
            return
        unmodeled = sorted(k for k in cfg if any(h in k for h in _UNMODELED_CFG_HINTS))
    if unmodeled:
        warnings.warn(
            "HF config declares parameters the GPU OOM estimate can't model "
            f"({', '.join(unmodeled[:8])}); the estimate is a lower bound.",
            UserWarning,
            stacklevel=3,
        )
    knobs = recipe._field_values() | recipe._escape_hatch_values()
    peak, raised = _peak_gib(arch, knobs, gpu_gib)
    multi_turn_hooks = ["custom_generate_function", "custom_generate_function_path"]
    if knobs.get("loss_type") != "sft_loss":
        multi_turn_hooks += ["rollout_function", "rollout_function_path"]
    if any(knobs.get(k) for k in multi_turn_hooks) and not knobs.get(
        "rollout_max_context_len"
    ):
        warnings.warn(
            "Multi-turn rollouts (custom_generate_function / rollout_function) can grow a sample past "
            "rollout_max_prompt_len + rollout_max_response_len, so the GPU OOM "
            "estimate is a lower bound. Set rollout_max_context_len to cap sample length.",
            UserWarning,
            stacklevel=3,
        )
    if peak <= gpu_gib:
        return
    shape = ", ".join(
        f"{name}={getattr(recipe, name)}"
        for name in ("gpu_type", "actor_num_nodes", "actor_num_gpus_per_node")
    )
    settings = ", ".join(f"{name}={value}" for name, value in raised.items())
    warnings.warn(
        f"Estimated peak ~{peak:.1f} GiB per GPU exceeds "
        f"{recipe.gpu_type} capacity ({gpu_gib:g} GiB). Some relevant settings: {shape}. {settings}. "
        f"Set gpu_type on the recipe to use a larger GPU; see https://modal.com/docs/guide/gpu.",
        UserWarning,
        stacklevel=3,
    )


# Usable host RAM per GPU type (GiB), just under what the instance families
# admit; Ray kills workers once node usage crosses ~95% of it. Mirrors the
# per-type container caps in the deepseek_v41 recipe.
_HOST_RAM_GIB_BY_GPU = {
    "H100": 1900.0,
    "H200": 1900.0,
    "B300": 2800.0,
}

# Measured on a 3x8xH200 GLM-5.3 colocated run that host-OOMed: each sglang
# scheduler rank held ~150 GiB RSS (weight staging, engine structures, host
# KV offload). Used as the per-engine-GPU host estimate.
_SGLANG_HOST_GIB_PER_ENGINE_GPU = 150.0

# Ray object store, driver, and per-node overhead seen on the same run.
_HOST_RUNTIME_GIB = 100.0


def _host_gib(arch: ModelArchitecture, knobs: dict[str, object]) -> float:
    """Per-node host-RAM demand in GiB; every term is a lower bound."""
    get = knobs.get
    dense, experts = _param_counts(arch)
    params_gib = (dense + experts) * 2 / GIB  # one bf16 replica per node

    need = _HOST_RUNTIME_GIB
    colocate = bool(get("colocate"))
    if not colocate:
        return need

    # Training state spilled to host RAM while the rollout engine holds GPUs.
    # Recipes spell the toggle two ways: `offload_train` (glm_5_3) or the
    # inverted `no_offload_train` base flag; offload is on by default.
    target = str(get("offload_train_target") or "cpu").lower()
    train_offloaded = bool(get("offload_train")) or not get("no_offload_train")
    if train_offloaded and target != "disk":
        # bf16 weights (+ grads/optimizer for full FT); LoRA spills the base.
        need += params_gib if get("lora_rank") else params_gib * 8
    if get("optimizer_cpu_offload"):
        fraction = get("optimizer_offload_fraction")
        fraction = 1.0 if fraction is None else min(1.0, max(0.0, float(fraction)))
        need += (dense + experts) * 12 * fraction / GIB
    if get("lora_base_cpu_backup"):
        need += params_gib
    # Engine weights offloaded to host during the training step.
    rollout_offloaded = bool(get("offload_rollout")) or not get("no_offload_rollout")
    if rollout_offloaded:
        need += params_gib

    # Co-located engines: host RSS scales with the ranks resident on the node.
    engine_gpus = int(get("rollout_num_gpus_per_engine") or 1)
    gpus_per_node = int(get("actor_num_gpus_per_node") or engine_gpus)
    engines = max(1, gpus_per_node // max(1, engine_gpus))
    need += engines * engine_gpus * _SGLANG_HOST_GIB_PER_ENGINE_GPU
    return need


def maybe_warn_host_oom(recipe: BaseTrainRecipe, model: ModelConfig) -> None:
    """Warn when a colocated run's host-RAM estimate exceeds the node's RAM.

    GPU-side checks miss this: Ray kills workers once *host* usage crosses
    ~95% (OutOfMemoryError: "node running low on memory"), and the top
    consumers are host-side — sglang scheduler RSS, train/rollout offload,
    and the object store.
    """
    if hasattr(recipe, "train_backend") and recipe.train_backend != "megatron":
        return
    gpu = recipe.gpu_type.split(":")[0].rstrip("!").upper()
    host_gib = _HOST_RAM_GIB_BY_GPU.get(gpu)
    if host_gib is None:
        return
    arch = model.architecture
    if arch is None:
        cfg = _hf_cfg(model.model_name)
        if cfg is None:
            return
        arch = _arch_from_hf(model.model_name, cfg)
        if arch is None:
            return
    knobs = recipe._field_values() | recipe._escape_hatch_values()
    need = _host_gib(arch, knobs)
    if need <= host_gib:
        return
    shape = ", ".join(
        f"{name}={getattr(recipe, name)}"
        for name in ("gpu_type", "actor_num_nodes", "actor_num_gpus_per_node")
    )
    warnings.warn(
        f"Estimated host RAM ~{need:.0f} GiB per node exceeds the ~{host_gib:.0f} GiB "
        f"usable on {recipe.gpu_type} hosts; Ray kills workers at ~95% node usage. "
        f"Some relevant settings: {shape}. "
        "Set offload_train_target='disk' (as the deepseek_v41 recipe does) or use "
        "more nodes; the estimate is a lower bound.",
        UserWarning,
        stacklevel=3,
    )
