"""Pure helpers behind the Qwen3-ASR audio example: prompt rendering (the audio
payload must never leak into the text, or the actor OOMs) and Sample audio
extraction (fail loudly when absent).
"""

import base64
import types

import pytest

from modal_dojo.common.eval import AudioEvalRowResult, EvalRowResult
from modal_dojo.common.models.qwen3_asr_1_7b import (
    Qwen3_ASR_1_7B,
    _prompt_user_text,
    render_prompt,
)
from modal_dojo.frameworks.slime.audio_transcription_rollout import _audio_ref

_PLACEHOLDER = Qwen3_ASR_1_7B.audio_placeholder
_DATA_URI = "data:audio/wav;base64," + base64.b64encode(b"RIFFxxxx").decode()


def _audio_prompt(text="Transcribe the speech."):
    """A slime conversation-list prompt: one audio item + one text item."""
    content = [{"type": "audio", "audio": _DATA_URI}, {"type": "text", "text": text}]
    return [{"role": "user", "content": content}]


# ── render_prompt ────────────────────────────────────────────────────────────


def test_render_prompt_never_leaks_audio_payload():
    out = render_prompt(_audio_prompt())
    assert _DATA_URI not in out and "data:audio" not in out
    assert _PLACEHOLDER in out
    assert "Transcribe the speech." in out
    assert out.endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize("empty", [None, "", []])
def test_render_prompt_empty_is_placeholder_only(empty):
    out = render_prompt(empty)
    assert _PLACEHOLDER in out
    assert out.endswith("<|im_start|>assistant\n")


# ── _prompt_user_text ────────────────────────────────────────────────────────


@pytest.mark.parametrize("empty", [None, "", []])
def test_prompt_user_text_empty(empty):
    assert _prompt_user_text(empty) == ""


def test_prompt_user_text_strips_audio_marker():
    assert _prompt_user_text("<audio>\nTranscribe.") == "Transcribe."


def test_prompt_user_text_extracts_user_text_and_drops_media():
    assert _prompt_user_text(_audio_prompt("hello")) == "hello"


def test_prompt_user_text_ignores_non_user_roles():
    prompt = [
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "kept"},
    ]
    assert _prompt_user_text(prompt) == "kept"


# ── _audio_ref ───────────────────────────────────────────────────────────────


def test_audio_ref_extracts_data_uri():
    assert _audio_ref(types.SimpleNamespace(prompt=_audio_prompt())) == _DATA_URI


def test_audio_ref_raises_when_no_audio():
    text_only = [{"role": "user", "content": [{"type": "text", "text": "no audio"}]}]
    with pytest.raises(RuntimeError, match="no audio"):
        _audio_ref(types.SimpleNamespace(prompt=text_only))


# ── AudioEvalRowResult ───────────────────────────────────────────────────────


def test_audio_eval_row_folds_fields_into_metadata():
    row = AudioEvalRowResult(
        score=0.9,
        response="h",
        prompt="p",
        audio="uri",
        reference="r",
        metrics={"wer": 0.1},
    )
    assert isinstance(row, EvalRowResult)
    assert (row.score, row.response, row.prompt) == (0.9, "h", "p")
    assert row.metadata == {
        "_metadata_type": "audio",
        "audio": "uri",
        "reference": "r",
        "metrics": {"wer": 0.1},
    }
    assert "hyp" not in row.metadata and "hypothesis" not in row.metadata


def test_audio_eval_row_metrics_are_user_defined():
    row = AudioEvalRowResult(score=4.2, audio="uri", metrics={"mos": 4.2, "cer": 0.03})
    assert row.metadata["metrics"] == {"mos": 4.2, "cer": 0.03}


def test_audio_eval_row_omits_unset_optionals_keeps_extra_metadata():
    row = AudioEvalRowResult(score=1.0, audio="uri", metadata={"foo": "bar"})
    assert row.metadata == {"_metadata_type": "audio", "audio": "uri", "foo": "bar"}
