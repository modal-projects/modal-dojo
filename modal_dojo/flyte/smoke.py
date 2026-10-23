"""Built-in smoke-test configs used to validate the Flyte backend remotely.

Defined at module level inside the package so Flyte's remote resolver can
re-import them from the shipped code bundle (inline/`__main__` classes do not
reach the remote container). They carry no weights and produce local rows, so a
``minimal`` run exercises the whole pipeline (image build, object-store I/O,
orchestration, checkpoint Dir) with no network downloads.
"""

from __future__ import annotations

from modal_dojo.common.dataset import DatasetConfig
from modal_dojo.common.models import ModelConfig


class SmokeDataset(DatasetConfig):
    """A few local rows for end-to-end pipeline smoke tests."""

    def input_key(self) -> str:
        return "text"

    def label_key(self) -> str | None:
        return None

    def rows(self):
        return [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "Flyte orchestrates distributed machine learning."},
            {"text": "Object storage persists checkpoints across runs."},
        ]


class SmokeModel(ModelConfig):
    """A config-only model; no weights are downloaded."""

    model_name = ""
    revision = None
    model_path = None
