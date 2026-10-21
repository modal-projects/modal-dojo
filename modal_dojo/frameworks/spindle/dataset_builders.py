"""Optional cookbook implementations, imported only by dataset conversion."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import Env, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer

from modal_dojo.common.dataset import HarborDataset, HuggingFaceDataset
from modal_dojo.common.errors import DojoConfigError


@dataclass
class _RowEnv(Env):
    messages: list[dict]
    label: Any
    renderer: Any
    reward_fn: Callable[[str, Any], float]

    async def initial_observation(self):
        return self.renderer.build_generation_prompt(
            self.messages
        ), self.renderer.get_stop_sequences()

    async def step(self, action, *, extra=None):
        message, _ = self.renderer.parse_response(action)
        response = renderers.get_text_content(message)
        reward = float(self.reward_fn(response, self.label))
        if not math.isfinite(reward):
            raise DojoConfigError("reward_fn returned a non-finite reward")
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
        )


class _Batches(RLDataset):
    def __init__(self, groups, batch_size):
        if not groups:
            raise DojoConfigError("The selected dataset is empty")
        self.groups = groups
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.groups) / self.batch_size)

    def get_batch(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.groups[index * self.batch_size : (index + 1) * self.batch_size]


@chz.chz
class HuggingFaceBuilder(RLDatasetBuilder):
    dataset: HuggingFaceDataset
    model_name: str
    renderer_name: str
    batch_size: int
    group_size: int
    reward_fn: Callable[[str, Any], float]

    async def __call__(self):
        renderer = renderers.get_renderer(
            self.renderer_name, tokenizer=get_tokenizer(self.model_name)
        )
        groups = []
        for row in self.dataset.rows():
            messages = row[self.dataset.input_key()]
            if not isinstance(messages, list) or not messages:
                raise DojoConfigError(
                    "Expected a nonempty messages list in the prompt column"
                )
            groups.append(
                ProblemGroupBuilder(
                    env_thunk=partial(
                        _RowEnv,
                        messages,
                        row[self.dataset.label_key()],
                        renderer,
                        self.reward_fn,
                    ),
                    num_envs=self.group_size,
                    dataset_name=self.dataset.hf_repo,
                )
            )
        return _Batches(groups, self.batch_size), None


@chz.chz
class HarborBuilder(RLDatasetBuilder):
    dataset: HarborDataset
    model_name: str
    renderer_name: str
    batch_size: int
    group_size: int
    options: dict[str, Any] = chz.field(default_factory=dict)

    async def __call__(self):
        from tinker_cookbook.recipes.harbor_rl.harbor_env import (
            HarborDatasetBuilder,
            HarborTask,
        )

        tasks = []
        for item in self.dataset.load(split=self.dataset.split):
            task_dir = Path(item["task_path"])
            instruction = self.dataset._format_prompt(
                instruction=item["instruction"],
                task_dir=task_dir,
                label=item["label"],
            )
            tasks.append(
                HarborTask(
                    task_name=item["task_name"],
                    instruction=instruction,
                    task_dir=task_dir,
                    config=tomllib.loads((task_dir / "task.toml").read_text()),
                )
            )
        repeats = {
            "train": self.dataset.train_repeats,
            "eval": self.dataset.eval_repeats,
        }.get(self.dataset.split, 1)
        tasks = [task for task in tasks for _ in range(max(1, repeats))]
        builder = HarborDatasetBuilder(
            tasks=tasks,
            batch_size=self.batch_size,
            group_size=self.group_size,
            model_name=self.model_name,
            renderer_name=self.renderer_name,
            **self.options,
        )
        # Retain the selected split; do not silently evaluate on the training set.
        groups = builder._make_env_group_builders(self.group_size)
        return _Batches(groups, self.batch_size), None
