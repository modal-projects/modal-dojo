from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from modal_dojo.common import hf_secrets, hf_token


def _mock_secret_cls():
    return patch("modal.Secret")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)


def test_hf_token_reads_hf_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_123")
    assert hf_token() == "hf_test_token_123"


def test_hf_token_reads_legacy_hub_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy_456")
    assert hf_token() == "hf_legacy_456"


def test_hf_token_prefers_hf_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "hf_primary")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy")
    assert hf_token() == "hf_primary"


def test_hf_token_empty_string_is_absent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "")
    assert hf_token() is None


def test_hf_token_missing_is_none():
    assert hf_token() is None


def test_local_hf_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_123")
    sentinel = MagicMock(name="from_dict_secret")

    with _mock_secret_cls() as mock_cls:
        mock_cls.from_dict.return_value = sentinel
        result = hf_secrets()

    assert result == [sentinel]
    mock_cls.from_dict.assert_called_once_with({"HF_TOKEN": "hf_test_token_123"})
    mock_cls.from_name.assert_not_called()


def test_legacy_hugging_face_hub_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy_456")
    sentinel = MagicMock(name="from_dict_secret")

    with _mock_secret_cls() as mock_cls:
        mock_cls.from_dict.return_value = sentinel
        result = hf_secrets()

    assert result == [sentinel]
    mock_cls.from_dict.assert_called_once_with({"HF_TOKEN": "hf_legacy_456"})


def test_hf_token_takes_priority_over_legacy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "hf_primary")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy")

    with _mock_secret_cls() as mock_cls:
        mock_cls.from_dict.return_value = MagicMock()
        hf_secrets()

    mock_cls.from_dict.assert_called_once_with({"HF_TOKEN": "hf_primary"})


def test_falls_back_to_modal_secret_when_no_local_token():
    modal_secret = MagicMock(name="modal_named_secret")

    with _mock_secret_cls() as mock_cls:
        mock_cls.from_name.return_value = modal_secret
        result = hf_secrets()

    assert result == [modal_secret]
    mock_cls.from_name.assert_called_once_with("huggingface-secret")
    modal_secret.hydrate.assert_called_once()


def test_hydrate_failure_returns_empty_list():
    modal_secret = MagicMock()
    modal_secret.hydrate.side_effect = RuntimeError("secret not found")

    with _mock_secret_cls() as mock_cls:
        mock_cls.from_name.return_value = modal_secret
        result = hf_secrets()

    assert result == []


def test_no_token_and_no_modal_secret_returns_empty_list():
    modal_secret = MagicMock()
    modal_secret.hydrate.side_effect = Exception("not found")

    with _mock_secret_cls() as mock_cls:
        mock_cls.from_name.return_value = modal_secret
        result = hf_secrets()

    assert result == []
    mock_cls.from_name.assert_called_once_with("huggingface-secret")
    modal_secret.hydrate.assert_called_once()


def test_empty_hf_token_is_treated_as_absent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "")
    modal_secret = MagicMock()
    modal_secret.hydrate.side_effect = Exception("not found")

    with _mock_secret_cls() as mock_cls:
        mock_cls.from_name.return_value = modal_secret
        result = hf_secrets()

    assert result == []
    mock_cls.from_name.assert_called_once()


def test_result_is_always_a_list(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abc")

    with _mock_secret_cls() as mock_cls:
        mock_cls.from_dict.return_value = MagicMock()
        result = hf_secrets()

    assert isinstance(result, list)
    assert len(result) == 1
