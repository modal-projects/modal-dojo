"""tinker-cookbook GRPO driver for a Spindle deployment.

Runs inside the GRPO driver container (tinker-cookbook image). It turns a
materialized Modal Dojo dataset into cookbook ``EnvGroupBuilder`` groups, scores
samples with the recipe's Miles-style ``custom_rm_function(args, sample)`` and
hands everything to ``tinker_cookbook.rl.train.main`` — the cookbook's GRPO loop
(group-centered advantages, importance-sampling policy loss).
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

PhaseCallback = Callable[[str], None]


@dataclass
class GrpoSettings:
    """Everything the driver needs, serializable for the Modal function call."""

    model_name: str
    base_url: str
    log_path: str
    dataset_path: str
    input_key: str
    label_key: str | None
    apply_chat_template: bool
    chat_template_kwargs: dict[str, Any]
    eval_dataset_path: str | None = None
    custom_rm_path: str | None = None
    num_rollout: int = 1
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 2
    n_samples_per_eval_prompt: int = 2
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    rollout_shuffle: bool = True
    rollout_stop_token_ids: list[int] = field(default_factory=list)
    max_context_length: int = 32768
    lr: float = 4e-5
    lora_rank: int = 32
    num_substeps: int = 1
    kl_penalty_coef: float = 0.0
    save_every: int = 0
    eval_every: int = 0
    remove_constant_reward_groups: bool = False
    seed: int = 0
    wandb_project: str | None = None
    wandb_name: str | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "GrpoSettings":
        return cls(**json.loads(data))


# ── Data loading ──────────────────────────────────────────────────────────────


def read_rows(path: str) -> list[dict[str, Any]]:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_messages(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, dict):
        return [prompt]
    return list(prompt)


@lru_cache(maxsize=4)
def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


@lru_cache(maxsize=8)
def load_callable(path: str) -> Callable[..., Any]:
    module_name, _, attr = path.rpartition(".")
    target: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        target = getattr(target, part)
    return target


def encode_prompt(tokenizer, prompt: Any, settings: GrpoSettings) -> list[int]:
    if settings.apply_chat_template:
        encoded = tokenizer.apply_chat_template(
            prompt_messages(prompt),
            tokenize=True,
            add_generation_prompt=True,
            **settings.chat_template_kwargs,
        )
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"]
        return list(encoded)
    text = prompt if isinstance(prompt, str) else json.dumps(prompt)
    return list(tokenizer.encode(text, add_special_tokens=True))


def stop_token_ids(tokenizer, settings: GrpoSettings) -> list[int]:
    ids: list[int] = []
    for candidate in (
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>")
        if "<|im_end|>" in tokenizer.get_vocab()
        else None,
        *settings.rollout_stop_token_ids,
    ):
        if isinstance(candidate, int) and candidate >= 0 and candidate not in ids:
            ids.append(candidate)
    return ids


def make_sample(
    row: dict[str, Any], index: int, settings: GrpoSettings
) -> SimpleNamespace:
    """Miles-compatible ``Sample`` stand-in handed to ``custom_rm_function``."""
    label = row.get(settings.label_key) if settings.label_key else None
    return SimpleNamespace(
        index=index,
        prompt=row[settings.input_key],
        label=label,
        response="",
        response_length=0,
        tokens=[],
        metadata={k: v for k, v in row.items() if k != settings.input_key},
        status="completed",
        reward=None,
    )


async def score_sample(settings: GrpoSettings, sample: SimpleNamespace) -> float:
    rm = load_callable(
        settings.custom_rm_path
        or "modal_dojo.frameworks.spindle.rewards.boxed_match_reward"
    )
    args = SimpleNamespace(**asdict(settings))
    result = rm(args, sample)
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, dict):
        result = result.get("reward", 0.0)
    return float(result)


# ── cookbook environment adapters ─────────────────────────────────────────────


def build_cookbook_types(settings: GrpoSettings):
    """Create the cookbook ``Env``/``EnvGroupBuilder``/``RLDataset`` adapters.

    Defined inside a factory so this module stays importable without tinker.
    """
    import chz  # pyright: ignore[reportMissingImports]
    import tinker  # pyright: ignore[reportMissingImports]
    from tinker_cookbook.rl.types import (  # pyright: ignore[reportMissingImports]
        Action,
        ActionExtra,
        Env,
        EnvGroupBuilder,
        RLDataset,
        RLDatasetBuilder,
        StepResult,
    )

    class DojoPromptEnv(Env):
        def __init__(self, row: dict[str, Any], index: int, cfg: GrpoSettings):
            self.row = row
            self.index = index
            self.cfg = cfg
            self.tokenizer = load_tokenizer(cfg.model_name)
            self.stop = stop_token_ids(self.tokenizer, cfg)

        async def initial_observation(self):
            tokens = encode_prompt(
                self.tokenizer, self.row[self.cfg.input_key], self.cfg
            )
            budget = self.cfg.max_context_length - self.cfg.rollout_max_response_len
            if len(tokens) > budget:
                tokens = tokens[-budget:]
            return tinker.ModelInput.from_ints(tokens), self.stop

        async def step(self, action: Action, *, extra: ActionExtra | None = None):
            sample = make_sample(self.row, self.index, self.cfg)
            sample.tokens = list(action)
            sample.response_length = len(action)
            sample.response = self.tokenizer.decode(action, skip_special_tokens=True)
            if extra and extra.get("stop_reason") == "length":
                sample.status = "truncated"
            reward = await score_sample(self.cfg, sample)
            sample.reward = reward
            return StepResult(
                reward=reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop,
                metrics={"response_length": float(len(action))},
            )

    @dataclass(frozen=True)
    class DojoGroupBuilder(EnvGroupBuilder):
        row: dict[str, Any]
        index: int
        group_size: int
        settings_json: str

        async def make_envs(self) -> Sequence[Env]:
            cfg = GrpoSettings.from_json(self.settings_json)
            return [
                DojoPromptEnv(self.row, self.index, cfg) for _ in range(self.group_size)
            ]

        def logging_tags(self) -> list[str]:
            return ["dojo"]

    class DojoRLDataset(RLDataset):
        def __init__(
            self,
            rows: list[dict[str, Any]],
            *,
            batch_size: int,
            group_size: int,
            num_batches: int,
            shuffle: bool,
            seed: int,
            cfg: GrpoSettings,
        ):
            if not rows:
                raise ValueError("dataset has no rows")
            self.rows = rows
            self.batch_size = batch_size
            self.group_size = group_size
            self.num_batches = num_batches
            self.shuffle = shuffle
            self.seed = seed
            self.settings_json = cfg.to_json()
            self._epoch_orders: dict[int, list[int]] = {}

        def _order(self, epoch: int) -> list[int]:
            if epoch not in self._epoch_orders:
                order = list(range(len(self.rows)))
                if self.shuffle:
                    random.Random(self.seed + epoch).shuffle(order)
                self._epoch_orders[epoch] = order
            return self._epoch_orders[epoch]

        def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
            per_epoch = max(1, len(self.rows) // self.batch_size)
            epoch, offset = divmod(index, per_epoch)
            order = self._order(epoch)
            start = offset * self.batch_size
            picked = order[start : start + self.batch_size]
            return [
                DojoGroupBuilder(self.rows[i], i, self.group_size, self.settings_json)
                for i in picked
            ]

        def __len__(self) -> int:
            return self.num_batches

    @chz.chz
    class DojoDatasetBuilder(RLDatasetBuilder):
        settings_json: str

        async def __call__(self):
            cfg = GrpoSettings.from_json(self.settings_json)
            train = DojoRLDataset(
                read_rows(cfg.dataset_path),
                batch_size=cfg.rollout_batch_size,
                group_size=cfg.n_samples_per_prompt,
                num_batches=cfg.num_rollout,
                shuffle=cfg.rollout_shuffle,
                seed=cfg.seed,
                cfg=cfg,
            )
            test = None
            if cfg.eval_dataset_path and cfg.eval_every:
                eval_rows = read_rows(cfg.eval_dataset_path)
                test = DojoRLDataset(
                    eval_rows,
                    batch_size=len(eval_rows),
                    group_size=cfg.n_samples_per_eval_prompt,
                    num_batches=1,
                    shuffle=False,
                    seed=cfg.seed,
                    cfg=cfg,
                )
            return train, test

    return SimpleNamespace(
        Env=DojoPromptEnv,
        GroupBuilder=DojoGroupBuilder,
        Dataset=DojoRLDataset,
        DatasetBuilder=DojoDatasetBuilder,
    )


# ── Training entry point ──────────────────────────────────────────────────────


def build_train_config(settings: GrpoSettings):
    from tinker_cookbook.rl import train  # pyright: ignore[reportMissingImports]

    types = build_cookbook_types(settings)
    values: dict[str, Any] = {
        "learning_rate": settings.lr,
        "dataset_builder": types.DatasetBuilder(settings_json=settings.to_json()),
        "model_name": settings.model_name,
        "recipe_name": "modal_dojo_spindle_grpo",
        "max_tokens": settings.rollout_max_response_len,
        "log_path": settings.log_path,
        "eval_every": settings.eval_every,
        "save_every": settings.save_every,
        "loss_fn": "importance_sampling",
        "num_substeps": settings.num_substeps,
        "lora_rank": settings.lora_rank,
        "temperature": settings.rollout_temperature,
        "kl_penalty_coef": settings.kl_penalty_coef,
        "remove_constant_reward_groups": settings.remove_constant_reward_groups,
        "base_url": settings.base_url,
        "max_steps": settings.num_rollout,
        "wandb_project": settings.wandb_project,
        "wandb_name": settings.wandb_name,
    }
    if settings.kl_penalty_coef > 0:
        values["kl_reference_config"] = train.KLReferenceConfig(
            base_model=settings.model_name
        )
    values.update(settings.config_overrides)
    return train.Config(**values)


def run_grpo(settings: GrpoSettings, *, on_phase: PhaseCallback | None = None) -> None:
    """Train with tinker-cookbook GRPO against ``settings.base_url``."""
    from tinker_cookbook.rl import train  # pyright: ignore[reportMissingImports]

    os.environ.setdefault("TINKER_BASE_URL", settings.base_url)
    os.makedirs(settings.log_path, exist_ok=True)
    config = build_train_config(settings)
    if on_phase:
        on_phase("training")
    asyncio.run(train.main(config))


def main(argv: Iterable[str] | None = None) -> None:
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 1:
        raise SystemExit(
            "usage: python -m modal_dojo.frameworks.spindle.grpo <settings.json>"
        )
    with open(args[0]) as f:
        run_grpo(GrpoSettings.from_json(f.read()))


if __name__ == "__main__":
    main()
