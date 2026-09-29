"""Validating a model by running base training on slime."""

from __future__ import annotations

from modal_training_gym.common.dataset import (
    DatasetConfig,
    HuggingFaceDataset,
    MultimodalDataset,
)
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.models.qwen3_asr_1_7b import Qwen3_ASR_1_7B
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

VALIDATION_EPHEMERAL_DISK_MIB = 2_097_152


class Gsm8kDataset(DatasetConfig):
    def __init__(self, *, n_rows: int = 10) -> None:
        self.n_rows = n_rows

    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> str:
        return "label"

    def rows(self):
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split="train")
        dataset = dataset.select(range(min(self.n_rows, len(dataset))))
        for row in dataset:
            yield {
                "messages": [{"role": "user", "content": row["question"]}],
                "label": row["answer"].split("####")[-1].strip(),
            }


class LibriSpeechASRDataset(MultimodalDataset):
    """LibriSpeech ASR rows (prompt + audio data-URI + transcript label).

    Mirrors the audio_asr tutorial dataset. Audio models validate on a handful
    of LibriSpeech clips. gsm8k is text-only.
    """

    hf_repo = "hf-internal-testing/librispeech_asr_dummy"
    hf_config = "clean"
    hf_split = "validation"

    _INSTRUCTION = (
        "<audio>\nTranscribe the speech to text. Respond with only the transcript."
    )

    def __init__(self, *, n_rows: int = 8):
        self.n_rows = n_rows
        super().__init__(modality="audio")

    def apply_chat_template(self) -> bool:
        return False

    def source_rows(self):
        import base64 as b64
        import io

        import soundfile as sf
        from datasets import Audio, load_dataset

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        ds = ds.select(range(min(self.n_rows, len(ds))))
        ds = ds.cast_column("audio", Audio(decode=False))
        for ex in ds:
            audio = ex["audio"]
            data = (
                audio["bytes"]
                if audio.get("bytes")
                else open(audio["path"], "rb").read()
            )
            arr, sr = sf.read(io.BytesIO(data))
            buf = io.BytesIO()
            sf.write(buf, arr, sr, format="WAV")
            data_uri = "data:audio/wav;base64," + b64.b64encode(buf.getvalue()).decode(
                "ascii"
            )
            yield {
                "prompt": self._INSTRUCTION,
                "media": data_uri,
                "label": ex["text"].lower().strip(),
            }


def build_slime_validation(
    model_config: ModelConfig,
    step_count: int,
    *,
    loss_type: str = "policy_loss",
) -> tuple[SlimeRecipe, DatasetConfig]:
    """The model's base slime recipe and a dataset matching its modality.

    Audio models (Qwen3-ASR) need speech clips, so they get LibriSpeech;
    everything else validates against gsm8k, scored by ``deepscaler``.
    SFT entries train on a short conversation slice instead.
    """
    recipe = SlimeRecipe.get_base_recipe(model_config)
    recipe.train_function_kwargs = {
        **dict(recipe.train_function_kwargs or {}),
        "ephemeral_disk": VALIDATION_EPHEMERAL_DISK_MIB,
    }

    if loss_type == "sft_loss":
        recipe.loss_type = "sft_loss"
        batch_size = (
            recipe.global_batch_size
            if recipe.global_batch_size is not None
            else recipe.rollout_batch_size
        )
        n_rows = batch_size * step_count
        return recipe, HuggingFaceDataset(
            "HuggingFaceH4/no_robots",
            hf_split=f"train[:{n_rows}]",
            input_column="messages",
            input_format="messages",
            always_download=True,
        )

    recipe.rm_type = "deepscaler"
    if isinstance(model_config, Qwen3_ASR_1_7B):
        return recipe, LibriSpeechASRDataset(n_rows=8)
    return recipe, Gsm8kDataset(n_rows=10)
