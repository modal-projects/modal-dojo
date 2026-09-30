"""Qwen3-ASR-1.7B as a Modal Dojo model.

Qwen3-ASR is an audio-only ASR model served by SGLang on
``/v1/audio/transcriptions``. Its text backbone is a dense Qwen3-1.7B; the audio
tower (a Qwen3-Omni-style Whisper encoder) is loaded by Megatron-Bridge straight
from the HF checkpoint and isn't expressible in ``ModelArchitecture`` (which is
LLM-backbone-only).

The class holds only the model's specs. Alongside it live the Qwen3-ASR-specific
input-prep functions (prompt rendering + processor tokenization) the
transcription rollout needs — model knowledge, kept out of the generic slime glue,
mirroring ``parse_qwen3_response`` in :mod:`.base`. Framework wiring (the slime
image-build shims and audio deps) lives on ``Qwen3_ASR_1_7B_Recipe`` instead,
since it's meaningless for other backends.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modal_dojo.common.audio import decode_to_mono

from .base import HFModelConfiguration, ModelArchitecture, parse_qwen3_response

if TYPE_CHECKING:
    import torch


class Qwen3_ASR_1_7B(HFModelConfiguration):
    """Alibaba Qwen3-ASR-1.7B speech recognition model."""

    model_name = "Qwen/Qwen3-ASR-1.7B"

    # Qwen3 dense backbone, same ``<|im_start|>``/``<|im_end|>`` delimiters as the
    # rest of the family. ASR output is plain transcription (no tool calls), so
    # this just strips the chat-template scaffolding off the decoded text.
    response_parser = staticmethod(parse_qwen3_response)

    requires_bshd = True
    audio_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"

    # thinker_config.text_config (Qwen3 dense backbone), verbatim from config.json.
    architecture = ModelArchitecture(
        num_layers=28,
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=8,  # num_key_value_heads
        kv_channels=128,  # head_dim (explicit; != hidden/heads in general)
        vocab_size=151936,
        normalization="RMSNorm",
        norm_epsilon=1e-6,
        swiglu=True,  # hidden_act = silu
        disable_bias_linear=True,  # Qwen3 dropped qkv bias
        qk_layernorm=True,  # Qwen3 family adds qk-layernorm
        use_rotary_position_embeddings=True,
        rotary_base=1000000,  # rope_theta
    )

    def download(self) -> None:
        """Download the model and add the ``tokenizer.json`` required by SGLang."""
        super().download()
        self._materialize_router_tokenizer()

    def _materialize_router_tokenizer(self) -> None:
        import os

        from huggingface_hub import snapshot_download

        snapshot_dir = snapshot_download(repo_id=self.model_name, local_files_only=True)
        target = os.path.join(snapshot_dir, "tokenizer.json")
        if os.path.exists(target):
            return

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            return  # no fast tokenizer to write
        backend.save(target)
        print(f"[modal-dojo] Wrote router-loadable tokenizer.json -> {target}")


@functools.lru_cache(maxsize=None)
def _processor(checkpoint: str) -> Any:
    """Cached Qwen3-ASR processor (WhisperFeatureExtractor + tokenizer).

    Loaded from the in-image checkpoint, so the SGLang import is deferred to call
    time and the cache key is the checkpoint path.
    """
    from sglang.srt.configs.qwen3_asr import Qwen3ASRProcessor

    return Qwen3ASRProcessor.from_pretrained(checkpoint)


def _prompt_user_text(prompt: str | list) -> str:
    """User instruction text from slime's ``Sample.prompt`` (a chat-message list in
    our case, or a plain string), with audio payloads and the ``<audio>`` marker
    stripped. Empty/missing is fine for ASR — the audio placeholder alone drives
    transcription, so the instruction text is optional."""
    if not prompt:  # None / "" / [] → no instruction text
        return ""
    if not isinstance(prompt, list):
        return str(prompt).replace("<audio>", "").strip()

    parts: list[str] = []
    for msg in prompt:
        if not isinstance(msg, dict) or msg.get("role", "user") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    # drop audio/image items — never render their data payloads
                elif isinstance(item, str):
                    parts.append(item)
        elif isinstance(content, str):
            parts.append(content)
    return " ".join(p for p in parts if p).replace("<audio>", "").strip()


def render_prompt(prompt: str | list) -> str:
    """Render the Qwen3-ASR prompt text: one audio placeholder, no audio payload.

    Qwen3-ASR's tokenizer carries no chat template, so we assemble the Qwen
    ``<|im_start|>`` turns directly rather than via ``apply_chat_template``. The
    ``audio_placeholder`` is injected on the user turn; the processor later expands
    its ``<|audio_pad|>`` to the audio-encoder output length.
    """
    placeholder = Qwen3_ASR_1_7B.audio_placeholder
    user_text = _prompt_user_text(prompt)
    user = f"{placeholder}\n{user_text}".strip() if user_text else placeholder
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


@dataclass(frozen=True)
class EncodedSample:
    """Tokenized transcription sample for audio-conditioned GRPO training.

    The caller maps these onto its trainer-specific sample fields.
    """

    prompt_ids: list[int]  # prompt with <audio_pad> expanded to N tokens
    response_ids: list[int]  # response tokens
    multimodal_inputs: dict[str, torch.Tensor]  # mel features for the audio tower


def encode_training_inputs(
    checkpoint: str, prompt: str | list, response: str, audio_bytes: bytes
) -> EncodedSample:
    """Tokenize a transcription sample for audio-conditioned GRPO training."""
    proc = _processor(checkpoint)
    tokenizer = proc.tokenizer

    text = render_prompt(prompt)
    target_sr = int(getattr(proc.feature_extractor, "sampling_rate", 16000))
    waveform = decode_to_mono(audio_bytes, target_sr)

    out = proc(text=text, audio=waveform, return_tensors="pt")
    prompt_ids = [int(t) for t in out["input_ids"][0].tolist()]
    response_ids = [
        int(t) for t in tokenizer.encode(response, add_special_tokens=False)
    ]
    # An empty transcript is a valid (if poor) rollout, not an error — pad with EOS
    # so the response is never zero-length (prompt_length-1 must stay non-negative).
    if not response_ids:
        response_ids = [int(getattr(tokenizer, "eos_token_id", None) or 0)]

    multimodal_inputs = {
        key: out[key]
        for key in ("input_features", "feature_attention_mask")
        if out.get(key) is not None
    }
    return EncodedSample(prompt_ids, response_ids, multimodal_inputs)
