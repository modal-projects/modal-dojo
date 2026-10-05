"""GPT-OSS-120B model configuration.

Attention uses learnable sinks and a 128-token sliding window on every other
layer, carried here as Megatron's ``softmax_type`` and ``window_size``
settings. Miles loads the native MXFP4 experts through Megatron-Bridge and
dequantizes them to BF16 for training; rollout serves MXFP4 directly.
"""

from .base import HFModelConfiguration, ModelArchitecture, parse_gpt_oss_response


class GPT_OSS_120B(HFModelConfiguration):
    """OpenAI GPT-OSS-120B MoE model, 36 layers with 128 routed experts, 4 active."""

    model_name = "openai/gpt-oss-120b"
    response_parser = staticmethod(parse_gpt_oss_response)
    architecture = ModelArchitecture(
        num_layers=36,
        hidden_size=2880,
        ffn_hidden_size=2880,
        num_attention_heads=64,
        group_query_attention=True,
        num_query_groups=8,
        kv_channels=64,
        vocab_size=201088,
        normalization="RMSNorm",
        norm_epsilon=1e-5,
        swiglu=True,
        # Every projection, including the experts, carries a bias.
        disable_bias_linear=False,
        qk_layernorm=False,
        untie_embeddings_and_output_weights=True,
        no_masked_softmax_fusion=True,
        num_experts=128,
        moe_ffn_hidden_size=2880,
        moe_grouped_gemm=True,
        moe_router_topk=4,
        moe_token_dispatcher_type="alltoall",
        moe_router_dtype="fp32",
        moe_aux_loss_coeff=0.0,
        use_rotary_position_embeddings=True,
        rotary_base=150000,
        max_position_embeddings=131072,
        softmax_type="learnable",
        window_size="128,0",
        window_attn_skip_freq=2,
        no_rope_fusion=True,
    )
