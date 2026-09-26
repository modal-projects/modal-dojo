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
import random
import shutil
import tomllib
from pathlib import Path

from modal_training_gym.common.errors import TrainingGymConfigError

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
    def label_key(self) -> str:
        """Ground-truth column name."""
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
            for row in self.rows():
                f.write(json.dumps(row) + "\n")

    def _expected_columns(self) -> set[str]:
        cols: set[str] = set()
        if self.input_key():
            cols.add(self.input_key())
        if self.label_key():
            cols.add(self.label_key())
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
                    raise TrainingGymConfigError(f"{path!r} is empty")
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
            raise TrainingGymConfigError(
                f"{type(self).__name__}.write() wrote {path!r} but it is "
                f"missing required column(s) {sorted(missing)} "
                f"(input_key={self.input_key()!r}, label_key={self.label_key()!r}). "
                f"Columns present: {sorted(cols)}. "
                "Either rename the column(s) your write() writes, or implement "
                "input_key()/label_key() on your DatasetConfig subclass to match."
            )


class _SftDataset(DatasetConfig):
    def __init__(self, inner: DatasetConfig) -> None:
        if getattr(inner, "input_format", None) == "raw":
            raise TrainingGymConfigError(
                "input_format='raw' is not supported with loss_type='sft_loss'"
            )
        if isinstance(inner, HarborDataset):
            raise TrainingGymConfigError(
                "HarborDataset is not supported with loss_type='sft_loss'"
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

    def label_key(self) -> str:
        return ""

    def rows(self) -> Iterable[DatasetRow]:
        input_key, label_key = self._inner.input_key(), self._inner.label_key()
        for row in self._inner.rows():
            messages = row.get(input_key)
            if messages is None:
                raise TrainingGymConfigError(
                    f"SFT row is missing a prompt in column {input_key!r}"
                )
            if not isinstance(messages, list):
                messages = [{"role": "user", "content": str(messages)}]
            if not messages or messages[-1].get("role") != "assistant":
                label = row.get(label_key) if label_key else None
                if label in (None, ""):
                    raise TrainingGymConfigError(
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
            raise TrainingGymConfigError(
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

    def label_key(self) -> str:
        if self.input_format == "text":
            return "label" if self.output_column else ""
        else:
            return self.output_column or ""

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


class HarborDataset(DatasetConfig):
    """A dataset loaded from Harbor tasks.

    Attributes:
        dataset_name: Harbor dataset ID.
        path: Local Harbor dataset path.
        task_root: Local directory containing Harbor tasks.
        task_glob: Glob used to select task directories.
        task_names: Explicit task directory names to select.
        instruction_path: Relative path to each task instruction.
        label_metadata_path: Relative JSON or TOML metadata path.
        test_data_dir: Relative directory containing test data.
        prompt_template: Template applied to each task instruction.
        system_prompt: System message added to each prompt.
        train_size: Number of tasks in the training split.
        eval_size: Number of tasks in the evaluation split.
        train_repeats: Repetitions of each training row.
        eval_repeats: Repetitions of each evaluation row.
        shuffle_tasks: Shuffle tasks before splitting.
        shuffle_seed: Seed used to shuffle tasks.
    """

    _type: DatasetType = DatasetType.HARBOR

    def __init__(
        self,
        *,
        split: Literal["all", "train", "eval"] = "train",
        dataset_name: str = "",
        path: str | None = None,
        task_root: str = "",
        task_glob: str = "*",
        task_names: list[str] | None = None,
        instruction_path: str = "instruction.md",
        label_metadata_path: str | None = None,
        test_data_dir: str | None = None,
        prompt_template: str = "{instruction}",
        system_prompt: str = "",
        train_size: int | None = None,
        eval_size: int | None = None,
        train_repeats: int = 1,
        eval_repeats: int = 1,
        shuffle_tasks: bool = False,
        shuffle_seed: int = 0,
        always_download: bool = False,
    ) -> None:
        if split not in ("all", "train", "eval"):
            raise TrainingGymConfigError(
                f"split must be one of all/train/eval, got {split!r}"
            )
        self.split = split
        self.dataset_name = dataset_name
        self.path = path
        self.task_root = task_root
        self.task_glob = task_glob
        self.task_names = task_names
        self.instruction_path = instruction_path
        self.label_metadata_path = label_metadata_path
        self.test_data_dir = test_data_dir
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.train_size = train_size
        self.eval_size = eval_size
        self.train_repeats = train_repeats
        self.eval_repeats = eval_repeats
        self.shuffle_tasks = shuffle_tasks
        self.shuffle_seed = shuffle_seed
        self.always_download = always_download

    def cache_key(self) -> str | None:
        if self.always_download:
            return None
        return _materialization_fingerprint(
            {
                "dataset_name": self.dataset_name,
                "path": self.path,
                "task_root": self.task_root,
                "task_glob": self.task_glob,
                "task_names": self.task_names,
                "instruction_path": self.instruction_path,
                "label_metadata_path": self.label_metadata_path,
                "test_data_dir": self.test_data_dir,
                "prompt_template": self.prompt_template,
                "system_prompt": self.system_prompt,
                "train_size": self.train_size,
                "eval_size": self.eval_size,
                "train_repeats": self.train_repeats,
                "eval_repeats": self.eval_repeats,
                "shuffle_tasks": self.shuffle_tasks,
                "shuffle_seed": self.shuffle_seed,
                "split": self.split,
            }
        )

    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> str:
        return "label"

    def _harbor_dataset_ref(self) -> str:
        if "@" in self.dataset_name:
            return self.dataset_name
        return f"{self.dataset_name}@latest"

    def _harbor_cache_dir(self) -> Path:
        slug = self._harbor_dataset_ref().replace("/", "--").replace("@", "--")
        return Path.home() / ".cache" / "harbor" / "datasets" / slug

    def _download_harbor_dataset(self, cache_dir: Path) -> None:
        import subprocess

        ref = self._harbor_dataset_ref()
        harbor_bin = shutil.which("harbor")
        if harbor_bin is not None:
            cmd = [
                harbor_bin,
                "datasets",
                "download",
                ref,
                "--output-dir",
                str(cache_dir),
            ]
        else:
            uvx_bin = shutil.which("uvx")
            if uvx_bin is None:
                raise FileNotFoundError(
                    "Harbor CLI not found. Install `harbor` or `uvx` to download "
                    f"{self.dataset_name!r}."
                )
            cmd = [
                uvx_bin,
                "harbor",
                "datasets",
                "download",
                ref,
                "--output-dir",
                str(cache_dir),
            ]
        subprocess.run(cmd, check=True)

    def _pull_harbor_dataset(self) -> Path:
        cache_dir = self._harbor_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not any(cache_dir.rglob(self.instruction_path)):
            self._download_harbor_dataset(cache_dir)
        task_root = self._discover_task_root(cache_dir)
        if not any(task_root.rglob(self.instruction_path)):
            raise FileNotFoundError(f"No Harbor tasks found under {cache_dir}")
        return task_root

    def _discover_task_root(self, search_root: Path) -> Path:
        task_dirs = sorted(
            {
                instruction_file.parent
                for instruction_file in search_root.rglob(self.instruction_path)
                if instruction_file.is_file()
            }
        )
        if not task_dirs:
            return search_root
        if len(task_dirs) == 1:
            return task_dirs[0].parent
        import os

        return Path(os.path.commonpath([str(path) for path in task_dirs]))

    def _resolve_task_root(self) -> Path:
        if self.path:
            task_root = Path(self.path).resolve()
        elif self.dataset_name:
            task_root = self._pull_harbor_dataset()
        elif self.task_root:
            task_root = Path(self.task_root).resolve()
        else:
            raise TrainingGymConfigError(
                f"{type(self).__name__} requires dataset_name, path, or task_root"
            )
        if not task_root.exists():
            raise FileNotFoundError(f"task root does not exist: {task_root}")
        if not task_root.is_dir():
            raise TrainingGymConfigError(f"task root is not a directory: {task_root}")
        return task_root

    def _candidate_task_dirs(self, task_root: Path) -> list[Path]:
        if self.task_names is not None:
            return [
                (task_root / name).resolve()
                for name in self.task_names
                if (task_root / name).is_dir()
            ]
        return sorted(
            path.resolve() for path in task_root.glob(self.task_glob) if path.is_dir()
        )

    def _iter_task_dirs(self) -> list[Path]:
        task_root = self._resolve_task_root()
        task_dirs = self._candidate_task_dirs(task_root)
        if not task_dirs:
            discovered_root = self._discover_task_root(task_root)
            if discovered_root != task_root:
                task_root = discovered_root
                task_dirs = self._candidate_task_dirs(task_root)
        if self.shuffle_tasks:
            rng = random.Random(self.shuffle_seed)
            rng.shuffle(task_dirs)
        if not task_dirs:
            raise TrainingGymConfigError(f"No Harbor tasks found under {task_root}")
        if self.train_size is not None:
            max_tasks = self.train_size + (self.eval_size or 0)
            task_dirs = task_dirs[:max_tasks]
        return task_dirs

    def _read_label_metadata(self, task_dir: Path) -> dict[str, Any]:
        if not self.label_metadata_path:
            return {}
        metadata_path = task_dir / self.label_metadata_path
        if not metadata_path.exists():
            return {}
        if metadata_path.suffix == ".json":
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        elif metadata_path.suffix == ".toml":
            data = tomllib.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            raise TrainingGymConfigError(
                f"Unsupported label metadata file type for {metadata_path}; expected .json or .toml"
            )
        if not isinstance(data, dict):
            raise TrainingGymConfigError(
                f"Label metadata must decode to an object: {metadata_path}"
            )
        return data

    def _read_test_data(self, task_dir: Path) -> list[dict[str, str]]:
        assert self.test_data_dir is not None
        tests_dir = task_dir / self.test_data_dir
        test_cases: list[dict[str, str]] = []
        if not tests_dir.is_dir():
            return test_cases
        for in_file in sorted(tests_dir.glob("*.in")):
            out_file = in_file.with_suffix(".out")
            if out_file.exists():
                test_cases.append(
                    {
                        "input": in_file.read_text(encoding="utf-8"),
                        "expected_output": out_file.read_text(encoding="utf-8"),
                    }
                )
        return test_cases

    def _build_label(self, task_root: Path, task_dir: Path) -> dict[str, Any]:
        rel = task_dir.relative_to(task_root)
        rel_with_root = (Path(task_root.name) / rel).as_posix()
        label: dict[str, Any] = {
            "harbor_task_name": task_dir.name,
            "harbor_task_path": task_dir.as_posix(),
            "harbor_task_rel": rel_with_root,
        }
        label.update(self._read_label_metadata(task_dir))
        if self.test_data_dir:
            label["test_cases"] = self._read_test_data(task_dir)
        return label

    def _format_prompt(
        self, *, instruction: str, task_dir: Path, label: dict[str, Any]
    ) -> str:
        context = {
            "instruction": instruction,
            "task_name": task_dir.name,
            "task_path": task_dir.as_posix(),
            **label,
        }
        return self.prompt_template.format(**context).strip()

    def _build_row(self, task_root: Path, task_dir: Path) -> dict[str, Any]:
        instruction_file = task_dir / self.instruction_path
        if not instruction_file.exists():
            raise FileNotFoundError(
                f"instruction file does not exist for Harbor task {task_dir.name}: {instruction_file}"
            )
        instruction = instruction_file.read_text(encoding="utf-8").strip()
        label = self._build_label(task_root, task_dir)
        user_prompt = self._format_prompt(
            instruction=instruction, task_dir=task_dir, label=label
        )
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return {
            self.input_key(): messages,
            self.label_key(): json.dumps(label, separators=(",", ":")),
        }

    @staticmethod
    def _repeat_rows(rows: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
        repeats = max(1, repeats)
        return [row for row in rows for _ in range(repeats)]

    def load(self, split: Literal["all", "train", "eval"] = "all") -> Any:
        task_root = self._resolve_task_root()
        out = []
        for task_dir in self._iter_task_dirs():
            instruction_file = task_dir / self.instruction_path
            if not instruction_file.exists():
                raise FileNotFoundError(
                    f"instruction file does not exist for Harbor task {task_dir.name}: {instruction_file}"
                )
            label = self._build_label(task_root, task_dir)
            out.append(
                {
                    "task_name": task_dir.name,
                    "task_path": task_dir.as_posix(),
                    "instruction": instruction_file.read_text(encoding="utf-8").strip(),
                    "label": label,
                }
            )
        if self.train_size is not None:
            if split == "train":
                return out[: self.train_size]
            if split == "eval":
                return out[self.train_size : self.train_size + (self.eval_size or 0)]
        return out

    def rows(self) -> Iterable[DatasetRow]:
        task_root = self._resolve_task_root()
        base_rows = [
            self._build_row(task_root, task_dir) for task_dir in self._iter_task_dirs()
        ]
        if self.split == "all":
            return base_rows
        if self.train_size is None:
            selected = base_rows
        else:
            train_size = max(1, min(int(self.train_size), len(base_rows)))
            if self.split == "train":
                selected = base_rows[:train_size]
            else:
                selected = (
                    base_rows[train_size : train_size + (self.eval_size or 0)]
                    or base_rows
                )
        repeats = self.train_repeats if self.split == "train" else self.eval_repeats
        return self._repeat_rows(selected, int(repeats))


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
            raise TrainingGymConfigError(
                f"modality must be one of image/audio/video, got {modality!r}"
            )
        self.media_column = media_column or f"{modality}s"
        if (
            self.input_key() == self.media_column
            or self.label_key() == self.media_column
        ):
            raise TrainingGymConfigError(
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
            raise TrainingGymConfigError(
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
