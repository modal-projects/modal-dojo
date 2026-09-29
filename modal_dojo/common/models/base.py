from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ModelArchitecture:
    """Megatron transformer architecture parameters.

    Args:
        num_layers: Number of transformer layers.
        hidden_size: Hidden dimension size.
        ffn_hidden_size: Feed-forward network intermediate size.
        vocab_size: Vocabulary size.
        num_attention_heads: Number of attention heads.
        group_query_attention: Whether to enable grouped-query attention.
        num_query_groups: Number of KV head groups for grouped-query attention.
        kv_channels: Per-head key/value channel dimension.
        normalization: Layer normalization type.
        norm_epsilon: Normalization epsilon.
        swiglu: Whether to use SwiGLU activation in the feed-forward network.
        disable_bias_linear: Whether to disable bias in linear layers.
        qk_layernorm: Whether to normalize query and key projections.
        untie_embeddings_and_output_weights: Whether to use separate output
            projection weights.
        num_experts: Number of mixture-of-experts experts.
        moe_ffn_hidden_size: Per-expert feed-forward intermediate size.
        moe_shared_expert_intermediate_size: Shared-expert intermediate size.
        moe_router_score_function: Router scoring function.
        moe_token_drop_policy: Token drop policy for mixture-of-experts routing.
        moe_router_dtype: Data type for router computation.
        moe_permute_fusion: Whether to enable permute fusion.
        moe_aux_loss_coeff: Auxiliary load-balancing loss coefficient.
        megatron_model_type: Slime/Megatron model type used for checkpoint
            conversion outside bridge mode.
        apply_layernorm_1p: Whether to use zero-centered LayerNorm.
        use_gated_attention: Whether to enable gated attention.
        attention_output_gate: Whether to gate attention outputs.
        use_rotary_position_embeddings: Whether to use RoPE.
        rotary_base: Base frequency for RoPE.
        rotary_percent: Fraction of hidden dimensions that use RoPE.
        no_masked_softmax_fusion: Disable fused masked softmax.
        multi_latent_attention: Use multi-latent attention instead of standard MHA.
        kv_lora_rank: LoRA rank for compressed key/value projections.
        qk_head_dim: Query/key head dimension for multi-latent attention.
        qk_pos_emb_head_dim: Query/key positional-embedding head dimension.
        v_head_dim: Value head dimension for multi-latent attention.
        moe_layer_freq: Which layers are MoE, as a Megatron frequency string.
        moe_grouped_gemm: Fuse MoE expert GEMMs into a grouped kernel.
        moe_shared_expert_gate: Apply a gate to the shared expert.
        moe_router_topk: Experts selected per token.
        moe_router_pre_softmax: Softmax router scores before top-k.
        moe_router_enable_expert_bias: Add a learned bias to router scores.
        moe_router_load_balancing_type: Load-balancing loss used by the router.
        moe_token_dispatcher_type: How tokens are dispatched to experts.
        moe_router_bias_update_rate: Step size for router expert-bias updates.
        moe_router_group_topk: Groups considered before the per-group top-k.
        moe_router_num_groups: Expert groups for grouped routing.
        moe_router_topk_scaling_factor: Scale applied to top-k router scores.
        megatron_spec: Megatron transformer-spec import path segments.
        rotary_scaling_factor: RoPE frequency scale for long context.
        mscale: YaRN mscale applied to RoPE.
        mscale_all_dim: YaRN mscale applied across all rotary dimensions.
        no_rope_fusion: Disable fused RoPE kernels.
    """

    num_layers: int = 0
    hidden_size: int = 0
    ffn_hidden_size: int = 0
    num_attention_heads: int = 0
    group_query_attention: bool = True
    num_query_groups: int = 0
    kv_channels: int = 0
    vocab_size: int = 0
    normalization: str = "RMSNorm"
    norm_epsilon: float = 1e-6
    swiglu: bool = True
    disable_bias_linear: bool = True
    qk_layernorm: bool = True
    untie_embeddings_and_output_weights: bool = False
    no_masked_softmax_fusion: bool = False
    multi_latent_attention: bool = False
    kv_lora_rank: int = 0
    qk_head_dim: int = 0
    qk_pos_emb_head_dim: int = 0
    v_head_dim: int = 0
    num_experts: int = 0
    moe_layer_freq: str = ""
    moe_ffn_hidden_size: int = 0
    moe_shared_expert_intermediate_size: int = 0
    moe_grouped_gemm: bool = False
    moe_shared_expert_gate: bool = False
    moe_router_topk: int = 0
    moe_router_pre_softmax: bool = False
    moe_router_score_function: str = ""
    moe_router_enable_expert_bias: bool = False
    moe_router_load_balancing_type: str = ""
    moe_token_dispatcher_type: str = ""
    moe_router_bias_update_rate: float | None = None
    moe_router_group_topk: int = 0
    moe_router_num_groups: int = 0
    moe_router_topk_scaling_factor: float | None = None
    moe_token_drop_policy: str = ""
    moe_router_dtype: str = ""
    moe_permute_fusion: bool = False
    moe_aux_loss_coeff: float | None = None
    megatron_spec: list[str] | None = None
    megatron_model_type: str = ""
    apply_layernorm_1p: bool = False
    use_gated_attention: bool = False
    attention_output_gate: bool = False
    use_rotary_position_embeddings: bool = True
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    rotary_scaling_factor: float | None = None
    mscale: float | None = None
    mscale_all_dim: float | None = None
    no_rope_fusion: bool = False


@dataclass
class ToolCall:
    """Tool invocation parsed from model output.

    Attributes:
        name: Function the model asked to call.
        arguments: Values the model passed for that function's parameters.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedResponse:
    """Structured result of parsing raw model output.

    Attributes:
        content: Visible text after thinking and tool-call markup is stripped.
        tool_calls: Tool invocations parsed from the response.
        thinking: Reasoning text when the response includes a thinking block.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None


ResponseParser = Callable[[str], ParsedResponse]


class ModelConfig:
    """Defines model identity, weight download, and response parsing.

    Attributes:
        model_name: Hugging Face repo id or other weight identifier.
        model_path: Local directory of already-downloaded weights, if any.
        architecture: Megatron transformer sizes used for conversion and training.
        response_parser: Turns raw model text into a ``ParsedResponse``.
        requires_bshd: Use padded (bshd) batches so training skips the THD packing path.
        audio_placeholder: Token sequence the processor expands at ``<|audio_pad|>``.
            Raw audio in the prompt OOMs.
    """

    model_name: str = ""
    model_path: str | None = None
    architecture: ModelArchitecture | None = None
    response_parser: ResponseParser | None = None
    requires_bshd: bool = False
    audio_placeholder: str = ""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def download(self) -> None:
        """Download or materialize weights into the model volume."""
        raise NotImplementedError(f"{type(self).__name__} has no download()")

    def parse_response(self, text: str) -> ParsedResponse:
        """Parse model text with ``response_parser``.

        Without a configured parser, the model text becomes ``ParsedResponse.content``.

        Returns:
            Parsed model output.
        """
        if self.response_parser is not None:
            return self.response_parser(text)
        return ParsedResponse(content=text)


def _is_populated(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    with os.scandir(path) as entries:
        return any(entries)


class HFModelConfiguration(ModelConfig):
    """Downloads Hugging Face model weights with ``snapshot_download``."""

    def download(self) -> None:
        from huggingface_hub import snapshot_download

        # Always download into the shared HF cache (no ``local_dir``): with
        # huggingface_hub >= 1.0 passing ``local_dir`` writes straight to that
        # dir and skips the cache, which leaves the weights unresolvable via
        # ``snapshot_download(..., local_files_only=True)`` on later runs and
        # forces a re-download. Populating the cache keeps base models
        # reusable across runs.
        snapshot_dir = snapshot_download(repo_id=self.model_name)
        if self.model_path and not _is_populated(str(self.model_path)):
            import shutil

            shutil.copytree(snapshot_dir, str(self.model_path), dirs_exist_ok=True)


def disable_mtp_in_config(snapshot_dir: str, log_prefix: str) -> None:
    """Zero ``num_nextn_predict_layers`` in a snapshot's config.json.

    Megatron-side loaders read this field from config.json and create MTP
    (multi-token-prediction) layers that break multi-rank checkpoint
    save/load (the MTP head duplicates the embedding across pipeline
    stages). Callers pass ``log_prefix`` (e.g. the model module name) to
    tag the patch log line. No-op when the field is already 0.
    """
    cfg_path = os.path.join(snapshot_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get("num_nextn_predict_layers", 0) == 0:
        return
    cfg["num_nextn_predict_layers"] = 0
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[{log_prefix}] Patched config.json: num_nextn_predict_layers → 0")


def _split_thinking(text: str) -> tuple[str | None, str]:
    """Split a leading ``<think>...</think>`` block off ``text``.

    Returns ``(thinking, remainder)``. When no closing ``</think>`` is
    present, ``thinking`` is ``None`` and ``remainder`` is ``text`` unchanged.
    A stray opening ``<think>`` in the remainder is stripped.
    """
    if "</think>" not in text:
        return None, text
    head, tail = text.split("</think>", 1)
    thinking = head.replace("<think>", "").strip() or None
    return thinking, tail.replace("<think>", "")


def _coerce_arg_value(raw: str) -> Any:
    """Best-effort decode of a tool-call argument value.

    Returns the JSON-decoded value when ``raw`` parses as JSON (objects,
    arrays, numbers, booleans, null, quoted strings), otherwise the raw
    string with surrounding whitespace stripped.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw.strip()


# ── Qwen family ────────────────────────────────────────────────────────

_QWEN3_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Qwen3.5/3.6 (Qwen3-Coder lineage) wire format inside <tool_call> blocks:
#   <function=NAME>
#   <parameter=KEY>
#   value (may span lines)
#   </parameter>
#   </function>
_QWEN3_XML_FN_RE = re.compile(
    r"<function=([^>\n]+)>\s*(.*?)\s*(?:</function>|\Z)", re.DOTALL
)
_QWEN3_XML_PARAM_RE = re.compile(
    r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL
)


def _parse_json_tool_block(block: str) -> ToolCall | None:
    """Qwen3 wire format: the ``<tool_call>`` body is one JSON object."""
    try:
        data = json.loads(block)
        return ToolCall(name=data.get("name", ""), arguments=data.get("arguments", {}))
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None


def _parse_xml_tool_block(block: str) -> ToolCall | None:
    """Qwen3.5/3.6 wire format: ``<function=NAME>`` + ``<parameter=KEY>`` pairs."""
    fn = _QWEN3_XML_FN_RE.search(block)
    if fn is None:
        return None
    # values are rendered raw by the chat template (JSON only for non-strings),
    # so JSON-decode where possible and keep the raw string otherwise
    args = {
        key.strip(): _coerce_arg_value(value)
        for key, value in _QWEN3_XML_PARAM_RE.findall(fn.group(2))
    }
    return ToolCall(name=fn.group(1).strip(), arguments=args)


def _parse_qwen_chat(
    text: str, block_parsers: tuple[Callable[[str], ToolCall | None], ...]
) -> ParsedResponse:
    """Shared Qwen chat scaffolding: ``<think>`` blocks, ``<|im_*|>`` delimiters,
    and ``<tool_call>`` extraction; each block is decoded by the first
    ``block_parser`` that accepts it."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if "<|im_start|>assistant" in text:
        text = text.rsplit("<|im_start|>assistant", 1)[-1]
    text = text.replace("<|im_end|>", "")

    thinking, text = _split_thinking(text)
    # Strip a stray unterminated <think> too (no closing tag emitted yet).
    text = text.replace("<think>", "")

    tool_calls: list[ToolCall] = []
    for match in _QWEN3_TOOL_CALL_RE.finditer(text):
        for block_parser in block_parsers:
            call = block_parser(match.group(1))
            if call is not None:
                tool_calls.append(call)
                break
    content = _QWEN3_TOOL_CALL_RE.sub("", text).strip()

    return ParsedResponse(
        content=content,
        tool_calls=tool_calls,
        thinking=thinking,
    )


def parse_qwen3_response(text: str) -> ParsedResponse:
    """Parse Qwen3 reasoning, chat delimiters, and JSON tool calls.

    Handles ``<think>``/``</think>`` reasoning blocks,
    ``<|im_start|>``/``<|im_end|>`` chat-template delimiters,
    and ``<tool_call>``/``</tool_call>`` tool invocations with Qwen3's
    JSON body (``{"name": ..., "arguments": {...}}``).
    """
    return _parse_qwen_chat(text, (_parse_json_tool_block,))


def parse_qwen3_6_response(text: str) -> ParsedResponse:
    """Parse Qwen3.5/3.6-family model output into structured content.

    Same chat scaffolding as :func:`parse_qwen3_response`, but tool calls use
    the Qwen3-Coder-lineage XML body (``<function=...><parameter=...>...``)
    that the Qwen3.5/3.6 chat template emits. A JSON body is tolerated as a
    fallback so format drift still parses to a real call.
    """
    return _parse_qwen_chat(text, (_parse_xml_tool_block, _parse_json_tool_block))


# ── GLM family (GLM-4.5 / 4.6 / 4.7) ───────────────────────────────────

# GLM emits tool calls as an XML-ish block: the function name, then alternating
# ``<arg_key>``/``<arg_value>`` pairs. The name may sit on its own line OR run
# straight into the first ``<arg_key>`` with no separator — the live GLM-4.7
# server emits the inline form:
#
#   <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>
#
# so the name is everything before the first ``<arg_key>`` (not just the first
# line). This mirrors SGLang's ``glm45`` tool-call parser.
_GLM_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_GLM_ARG_RE = re.compile(
    r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
    re.DOTALL,
)


def parse_glm_response(text: str) -> ParsedResponse:
    """Parse GLM-4.5/4.6/4.7 output into structured content.

    Handles ``<think>``/``</think>`` reasoning blocks, the GLM chat-template
    turn delimiters (``<|assistant|>``, ``<|user|>``, ``<|observation|>``,
    ``<|endoftext|>``), and ``<tool_call>`` blocks with ``<arg_key>``/
    ``<arg_value>`` argument pairs.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if "<|assistant|>" in text:
        text = text.rsplit("<|assistant|>", 1)[-1]
    # Anything past the assistant turn belongs to a following turn, not output.
    for turn_token in ("<|user|>", "<|observation|>", "<|system|>", "<|endoftext|>"):
        text = text.split(turn_token, 1)[0]

    thinking, text = _split_thinking(text)

    tool_calls: list[ToolCall] = []
    for match in _GLM_TOOL_CALL_RE.finditer(text):
        block = match.group(1)
        # The function name is everything up to the first <arg_key> (GLM may or
        # may not put it on its own line); the rest holds the arg pairs.
        name_part, sep, rest = block.partition("<arg_key>")
        name = name_part.strip()
        if not name:
            continue
        arguments = {
            key.strip(): _coerce_arg_value(value)
            for key, value in _GLM_ARG_RE.findall(sep + rest)
        }
        tool_calls.append(ToolCall(name=name, arguments=arguments))
    content = _GLM_TOOL_CALL_RE.sub("", text).strip()

    return ParsedResponse(
        content=content,
        tool_calls=tool_calls,
        thinking=thinking,
    )


# ── Gemma family (Gemma 4) ─────────────────────────────────────────────

# Gemma 4 mirrors its delimiters (``<|x>`` … ``<x|>``) and wraps tool-call
# strings in ``<|"|>``:
#
#   <|turn>model
#   <|channel>thought
#   ...reasoning...
#   <channel|>
#   <|tool_call>call:get_weather{city:<|"|>Beijing<|"|>,days:3}<tool_call|>
#   <turn|>
#
_GEMMA4_THOUGHT_RE = re.compile(r"<\|channel>thought\n?(.*?)\n?<channel\|>", re.DOTALL)
_GEMMA4_TOOL_CALL_RE = re.compile(
    r"<\|tool_call>call:([^{]*)(\{.*?\})<tool_call\|>", re.DOTALL
)
_GEMMA4_STR_DELIM = '<|"|>'
_GEMMA4_BARE_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_.\-]*)\s*:")
_JSON_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


def _quote_bare_keys(raw: str) -> str:
    """Quote Gemma's unquoted object keys, skipping string contents.

    A blind substitution would corrupt a value like ``"a,b:c"``.
    """
    out: list[str] = []
    last = 0
    for match in _JSON_STRING_RE.finditer(raw):
        out.append(_GEMMA4_BARE_KEY_RE.sub(r'\1"\2":', raw[last : match.start()]))
        out.append(match.group(0))
        last = match.end()
    out.append(_GEMMA4_BARE_KEY_RE.sub(r'\1"\2":', raw[last:]))
    return "".join(out)


def _parse_gemma4_tool_block(name: str, body: str) -> ToolCall:
    """Parse a ``call:NAME`` body: JSON with bare keys and ``<|"|>`` for quotes."""
    try:
        args = json.loads(_quote_bare_keys(body.replace(_GEMMA4_STR_DELIM, '"')))
    except json.JSONDecodeError:
        args = {}
    return ToolCall(name=name, arguments=args if isinstance(args, dict) else {})


def parse_gemma4_response(text: str) -> ParsedResponse:
    """Parse Gemma 4 output into structured content.

    Handles the ``<|turn>ROLE``/``<turn|>`` turn delimiters, the
    ``<|channel>thought``/``<channel|>`` reasoning channel, and
    ``<|tool_call>call:NAME{...}<tool_call|>`` blocks whose string arguments are
    wrapped in ``<|"|>``.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if "<|turn>model" in text:
        text = text.rsplit("<|turn>model", 1)[-1]
    # Anything past the turn terminator belongs to a following turn, not output.
    text = text.split("<turn|>", 1)[0]

    thoughts = [match.strip() for match in _GEMMA4_THOUGHT_RE.findall(text)]
    text = _GEMMA4_THOUGHT_RE.sub("", text)
    # The generation prompt opens the thought channel, so a response can begin
    # inside one and close it with a bare ``<channel|>``.
    if "<channel|>" in text:
        head, text = text.split("<channel|>", 1)
        thoughts.insert(0, head.replace("<|channel>thought", "").strip())
    # A channel left open means the model never stopped reasoning.
    if "<|channel>thought" in text:
        text, tail = text.split("<|channel>thought", 1)
        thoughts.append(tail.strip())
    thinking = "\n".join(thought for thought in thoughts if thought) or None

    tool_calls = [
        _parse_gemma4_tool_block(name.strip(), body)
        for name, body in _GEMMA4_TOOL_CALL_RE.findall(text)
        if name.strip()
    ]
    content = _GEMMA4_TOOL_CALL_RE.sub("", text).strip()

    return ParsedResponse(
        content=content,
        tool_calls=tool_calls,
        thinking=thinking,
    )


# ── Inkling family (Inkling-Small) ─────────────────────────────────────

_INKLING_SECTION_RE = re.compile(
    r"<\|content_(thinking|text|invoke_tool_json)\|>(.*?)(?=<\|[a-z0-9_]+\|>|$)",
    re.DOTALL,
)


def parse_inkling_response(text: str) -> ParsedResponse:
    """Parse Inkling-family output into structured content.

    Handles the ``<|content_thinking|>`` and ``<|content_text|>`` sections of a
    ``<|message_model|>`` turn, closed by ``<|end_message|>`` and
    ``<|content_model_end_sampling|>``, plus ``<|content_invoke_tool_json|>``
    tool bodies. A response truncated mid-section still yields that section.
    """
    thinking: list[str] = []
    content: list[str] = []
    tool_calls: list[ToolCall] = []
    for kind, body in _INKLING_SECTION_RE.findall(text):
        if kind == "thinking":
            thinking.append(body.strip())
        elif kind == "text":
            content.append(body.strip())
        elif call := _parse_json_tool_block(body.strip()):
            tool_calls.append(call)

    return ParsedResponse(
        content="\n".join(part for part in content if part),
        tool_calls=tool_calls,
        thinking="\n".join(part for part in thinking if part) or None,
    )
