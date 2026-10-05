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


def _arch_from_hf(model_name: str) -> ModelArchitecture | None:
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
    )


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

    h, heads = arch.hidden_size, max(1, arch.num_attention_heads)
    kv, nqg = arch.kv_channels or h // heads, arch.num_query_groups or heads
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
        opt = 0.0 if offload else 12.0
        d_div = e_div = 1.0
        if get("use_distributed_optimizer"):
            d_div = max(1, world // (tp * pp))
            e_div = max(1, world // (ep * etp * pp))
        states = (
            (dense_p + expert_p) * 6 + dense_p * opt / d_div + expert_p * opt / e_div
        ) / GIB

    # max_seq_len caps a whole session sample, honored by agentic generators.
    # rollout_max_context_len bounds each generation request, so it only bounds
    # a sample for single-turn generation. Default: prompt + one response.
    path = str(get("custom_generate_function_path") or "")
    agentic = "agentic" in path
    cap_name = "max_seq_len" if agentic else "rollout_max_context_len"
    cap = int(get(cap_name) or 0)
    prompt = get("rollout_max_prompt_len")
    response = get("rollout_max_response_len")
    sample = cap or (int(prompt or 0) + int(response or 0))

    def raise_sample_len() -> None:
        if cap:
            raise_(cap_name, cap)
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
    knobs = recipe._field_values() | recipe._escape_hatch_values()
    # Multi-turn generators assemble a session of turns into one sample; a
    # callable can't be identified, so warn only on known path names.
    path = str(knobs.get("custom_generate_function_path") or "")
    multi_turn = "agentic" in path or "multi_turn" in path
    # Only agentic generators honor max_seq_len as a session cap.
    if multi_turn and not ("agentic" in path and knobs.get("max_seq_len")):
        warnings.warn(
            "Multi-turn rollouts can exceed "
            "rollout_max_prompt_len + rollout_max_response_len, so the OOM "
            "estimate is a lower bound. Set max_seq_len to cap sample length.",
            UserWarning,
            stacklevel=3,
        )
    gpu_gib = gpu_memory_gib(recipe.gpu_type)
    if gpu_gib is None:
        return
    arch = model.architecture or _arch_from_hf(model.model_name)
    if arch is None:
        return
    peak, raised = _peak_gib(arch, knobs, gpu_gib)
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
