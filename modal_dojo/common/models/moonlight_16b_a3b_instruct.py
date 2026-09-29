"""Moonlight-16B-A3B-Instruct model configuration."""

from .base import HFModelConfiguration, ModelArchitecture


class Moonlight_16B_A3B_Instruct(HFModelConfiguration):
    """Moonshot AI Moonlight model with 16B total and 3B active parameters."""

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    architecture = ModelArchitecture(
        num_layers=27,
        hidden_size=2048,
        ffn_hidden_size=11264,
        num_attention_heads=16,
        group_query_attention=False,
        num_query_groups=16,
        kv_channels=128,
        vocab_size=163840,
        normalization="RMSNorm",
        norm_epsilon=1e-5,
        swiglu=True,
        disable_bias_linear=True,
        qk_layernorm=True,
        untie_embeddings_and_output_weights=True,
        no_masked_softmax_fusion=True,
        multi_latent_attention=True,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        num_experts=64,
        moe_layer_freq="[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]",
        moe_ffn_hidden_size=1408,
        moe_shared_expert_intermediate_size=2816,
        moe_grouped_gemm=True,
        moe_router_topk=6,
        moe_router_pre_softmax=True,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_router_bias_update_rate=0,
        moe_router_group_topk=1,
        moe_router_num_groups=1,
        moe_router_topk_scaling_factor=2.446,
        moe_token_drop_policy="probs",
        moe_router_dtype="fp32",
        moe_permute_fusion=True,
        moe_aux_loss_coeff=0.0,
        megatron_model_type="moonlight",
        use_rotary_position_embeddings=True,
        rotary_base=50000,
        rotary_scaling_factor=1,
        mscale=1.0,
        mscale_all_dim=1.0,
        no_rope_fusion=True,
    )
