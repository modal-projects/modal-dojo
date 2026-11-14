"""Classes for defining and using datasets for training.

Datasets inherit from a base ``DatasetConfig`` class and provide a ``rows()``
method that returns an iterable collection of rows. Subclasses are provided for
common sources like Hugging Face and Harbor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import Any, Literal
import hashlib
import json
from pathlib import Path

from modal_dojo.common.errors import DojoConfigError

DatasetRow = dict[str, Any]


def _materialization_fingerprint(fields: dict[str, Any]) -> str:
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class DatasetType(Enum):
    DEFAULT = "default"
    HUGGING_FACE = "hugging_face"
    HARBOR = "harbor"


class DatasetConfig(ABC):
    """Dataset fields and materialization behavior shared across training frameworks."""

    _type: DatasetType = DatasetType.DEFAULT

    def cache_key(self) -> str | None:
        return None

    @abstractmethod
    def input_key(self) -> str:
        """Prompt column name."""
        raise NotImplementedError(f"{type(self).__name__} has no input_key()")

    @abstractmethod
    def label_key(self) -> str | None:
        """Ground-truth column name, or ``None`` when rows carry no label."""
        raise NotImplementedError(f"{type(self).__name__} has no label_key()")

    def output_format(self) -> str:
        """The on-disk format written by `write()`, either `parquet` or `jsonl`."""
        return "jsonl"

    def apply_chat_template(self) -> bool:
        """Whether to apply the model's chat template to the input."""
        return True

    @abstractmethod
    def rows(self) -> Iterable[DatasetRow]:
        """Load raw examples.

        Returns:
            An iterable collection of raw examples.
        """
        raise NotImplementedError(f"{type(self).__name__} has no rows()")

    def write(self, path: str) -> None:
        """Materialize training data at ``path``."""
        with open(path, "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in self.rows())

    def _expected_columns(self) -> set[str]:
        cols: set[str] = set()
        if self.input_key():
            cols.add(self.input_key())
        if label_key := self.label_key():
            cols.add(label_key)
        return cols

    def validate_written(self, path: str) -> None:
        """Validate the materialized file format and required columns."""
        import os

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{type(self).__name__}.write() did not produce {path!r}. "
            )

        expected = self._expected_columns()
        if not expected:
            return

        try:
            if path.endswith(".parquet"):
                import pyarrow.parquet as pq

                cols = set(pq.read_schema(path).names)
            elif path.endswith((".jsonl", ".json")):
                with open(path) as f:
                    first = f.readline().strip()
                if not first:
                    raise DojoConfigError(f"{path!r} is empty")
                cols = set(json.loads(first).keys())
            else:
                return
        except Exception as e:  # don't shadow the user's real bug with a sniff bug
            print(
                f"[{type(self).__name__}.validate_written] could not sniff "
                f"{path!r} ({e!r}); skipping schema check."
            )
            return

        missing = expected - cols
        if missing:
            raise DojoConfigError(
                f"{type(self).__name__}.write() wrote {path!r} but it is "
                f"missing required column(s) {sorted(missing)} "
                f"(input_key={self.input_key()!r}, label_key={self.label_key()!r}). "
                f"Columns present: {sorted(cols)}. "
                "Either rename the column(s) your write() writes, or implement "
                "input_key()/label_key() on your DatasetConfig subclass to match."
            )


class _SftDataset(DatasetConfig):
    def __init__(self, inner: DatasetConfig) -> None:
        if isinstance(inner, HuggingFaceDataset) and inner.input_format == "raw":
            raise DojoConfigError(
                "input_format='raw' is not supported with loss_type='sft_loss'"
            )
        if isinstance(inner, HarborTasks):
            raise DojoConfigError(
                "HarborTasks is not supported with loss_type='sft_loss'"
            )
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def cache_key(self) -> str | None:
        key = self._inner.cache_key()
        return None if key is None else f"{key}:sft"

    def input_key(self) -> str:
        return self._inner.input_key()

    def label_key(self) -> str | None:
        return None

    def rows(self) -> Iterable[DatasetRow]:
        input_key, label_key = self._inner.input_key(), self._inner.label_key()
        for row in self._inner.rows():
            messages = row.get(input_key)
            if messages is None:
                raise DojoConfigError(
                    f"SFT row is missing a prompt in column {input_key!r}"
                )
            if not isinstance(messages, list):
                messages = [{"role": "user", "content": str(messages)}]
            if not messages or messages[-1].get("role") != "assistant":
                label = row.get(label_key) if label_key else None
                if label in (None, ""):
                    raise DojoConfigError(
                        "SFT row has no label and does not end with an assistant turn"
                    )
                messages = [*messages, {"role": "assistant", "content": str(label)}]
            yield {**row, input_key: messages}


class HuggingFaceDataset(DatasetConfig):
    """A dataset loaded from a Hugging Face ``datasets`` repository.

    Attributes:
        hf_repo: Hugging Face dataset repository ID.
        hf_split: Source dataset split.
        hf_config: Source dataset configuration name.
        input_column: Source prompt column.
        output_column: Source answer column.
        input_format: Shape of the source prompt: plain ``text`` to convert into
            messages, preformatted ``messages``, or ``raw`` model input.
        system_prompt: System message added to formatted examples.
        prompt_template: Template applied to each source prompt.
        always_download: When training, always download the dataset from Hugging Face instead of caching it.
    """

    _type: DatasetType = DatasetType.HUGGING_FACE

    hf_repo: str
    hf_split: str
    hf_config: str | None
    input_column: str
    output_column: str | None
    input_format: Literal["text", "messages", "raw"]
    system_prompt: str
    prompt_template: str
    always_download: bool

    def __init__(
        self,
        hf_repo: str,
        *,
        hf_split: str = "train",
        hf_config: str | None = None,
        input_column: str,
        output_column: str | None = None,
        input_format: Literal["text", "messages", "raw"] = "text",
        system_prompt: str = "",
        prompt_template: str = "{input}",
        always_download: bool = False,
    ):
        if input_format not in ("text", "messages", "raw"):
            raise DojoConfigError(
                f"input_format must be one of text/messages/raw, got {input_format!r}"
            )
        self.hf_repo = hf_repo
        self.hf_split = hf_split
        self.hf_config = hf_config
        self.input_column = input_column
        self.output_column = output_column
        self.input_format = input_format
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.always_download = always_download

    def cache_key(self) -> str | None:
        if self.always_download:
            return None
        return _materialization_fingerprint(
            {
                "hf_repo": self.hf_repo,
                "hf_split": self.hf_split,
                "hf_config": self.hf_config,
                "input_column": self.input_column,
                "output_column": self.output_column,
                "input_format": self.input_format,
                "system_prompt": self.system_prompt,
                "prompt_template": self.prompt_template,
                "output_format": self.output_format(),
            }
        )

    def input_key(self) -> str:
        if self.input_format == "text":
            return "messages"
        else:
            return self.input_column

    def label_key(self) -> str | None:
        if self.input_format == "text":
            return "label" if self.output_column else None
        else:
            return self.output_column

    def apply_chat_template(self) -> bool:
        return self.input_format != "raw"

    def _load_hf_dataset(self):
        from datasets import load_dataset

        ds = load_dataset(
            self.hf_repo,
            self.hf_config,
            split=self.hf_split,
        )

        if self.input_format == "text":
            ds = ds.map(self._to_chat, remove_columns=ds.column_names)

        return ds

    def _to_chat(self, row):
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        user_content = self.prompt_template.format(input=row[self.input_column])
        messages.append({"role": "user", "content": user_content})
        result = {self.input_key(): messages}
        if self.output_column:
            result[self.label_key()] = str(row[self.output_column])
        return result

    def rows(self) -> Iterable[DatasetRow]:
        for row in self._load_hf_dataset():
            yield dict(row)

    def write(self, path: str) -> None:
        ds = self._load_hf_dataset()
        if self.output_format() == "parquet":
            ds.to_parquet(path)
        else:
            ds.to_json(path, orient="records", lines=True)


class HarborAgent:
    """A Harbor agent harness that solves each Harbor task in its own Modal Sandbox.

    Attributes:
        name: Harbor agent name, such as ``"pi"``.
    """

    def __init__(self, name: str) -> None:
        self.name = name


class HarborTasks(DatasetConfig):
    """Harbor tasks, each solved by an agent in its own Modal Sandbox.

    Attributes:
        agent: Agent that solves each task.
        path: Local directory whose subdirectories are Harbor tasks.
        task_glob: Glob, relative to ``path``, selecting task directories.
    """

    _type: DatasetType = DatasetType.HARBOR

    def __init__(self, *, agent: HarborAgent, path: str, task_glob: str = "*") -> None:
        self.agent = agent
        self.path = path
        self.task_glob = task_glob

    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> str:
        return None

    def _resolve_task_root(self) -> Path:
        task_root = Path(self.path).resolve()
        if not task_root.exists():
            raise FileNotFoundError(f"task root does not exist: {task_root}")
        if not task_root.is_dir():
            raise DojoConfigError(f"task root is not a directory: {task_root}")
        return task_root

    def _build_row(self, task_dir: Path) -> dict[str, Any]:
        instruction_file = task_dir / "instruction.md"
        if not instruction_file.is_file():
            raise FileNotFoundError(
                f"Harbor task {task_dir.name} has no instruction.md: {instruction_file}"
            )
        instruction = instruction_file.read_text(encoding="utf-8").strip()
        return {
            self.input_key(): [{"role": "user", "content": instruction}],
            "metadata": {
                "harbor_task_path": task_dir.as_posix(),
                "harbor_agent": self.agent.name,
            },
        }

    def rows(self) -> Iterable[DatasetRow]:
        root = self._resolve_task_root()
        task_dirs = sorted(p for p in root.glob(self.task_glob) if p.is_dir())
        if not task_dirs:
            raise DojoConfigError(
                f"No Harbor tasks match {self.task_glob!r} under {root}"
            )
        return [self._build_row(task_dir) for task_dir in task_dirs]


class MultimodalDataset(DatasetConfig):
    """Dataset of text prompts paired with image, audio, or video data."""

    # TODO(ben/joy): gate-check media at this boundary so the evals dashboard can
    # reliably visualize it. Two parts: (1) normalize each emitted media item to a
    # canonical, browser-renderable container per modality (audio->wav, image->png/
    # jpeg) instead of trusting whatever format the user brings — the audio tutorial's
    # dataset already re-encodes to wav, make it the convention here; (2) validate the
    # data-URI
    # MIME matches `modality`. Pairs with the dashboard fallback in EvalsPage.svelte.
    def __init__(
        self,
        rows: Iterable[dict[str, Any]] | None = None,
        *,
        modality: Literal["image", "audio", "video"] = "audio",
        media_column: str | None = None,
    ) -> None:
        self.modality = modality
        if modality not in ("image", "audio", "video"):
            raise DojoConfigError(
                f"modality must be one of image/audio/video, got {modality!r}"
            )
        self.media_column = media_column or f"{modality}s"
        if (
            self.input_key() == self.media_column
            or self.label_key() == self.media_column
        ):
            raise DojoConfigError(
                "media_column must differ from input_key and label_key"
            )
        self.multimodal_keys = {modality: self.media_column}
        self._source_rows = list(rows or [])

    def input_key(self) -> str:
        return "prompt"

    def label_key(self) -> str:
        return "label"

    def source_rows(self) -> Iterable[dict[str, Any]]:
        return self._source_rows

    def _to_row(self, r: dict[str, Any]) -> DatasetRow:
        media = r["media"]
        return {
            self.input_key(): r["prompt"],
            self.media_column: list(media)
            if isinstance(media, (list, tuple))
            else [media],
            self.label_key(): r["label"],
        }

    def rows(self) -> Iterable[DatasetRow]:
        for row in self.source_rows():
            yield self._to_row(row)


class OnlineRollout(DatasetConfig):
    """Placeholder rows that size a live generate batch."""

    def __init__(self, n_rows: int) -> None:
        if n_rows < 1:
            raise DojoConfigError(
                f"OnlineRollout n_rows must be a positive integer; got {n_rows!r}"
            )
        self.n_rows = n_rows

    def cache_key(self) -> str | None:
        return f"online-rollout-{self.n_rows}-conversation"

    def input_key(self) -> str:
        return "prompt"

    def label_key(self) -> str:
        return "label"

    def apply_chat_template(self) -> bool:
        return False

    def rows(self) -> Iterable[DatasetRow]:
        for i in range(self.n_rows):
            yield {
                self.input_key(): [{"role": "user", "content": ""}],
                self.label_key(): str(i),
            }
