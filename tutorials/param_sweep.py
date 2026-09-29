# ---
# order: 7
# ---
#
# # Massively-parallel hyperparameter sweeps
#
# When tuning RL runs, finding the optimal set of hyperparameters is time-consuming
# and error-prone if not properly guided or documented. This is made a first-class
# operation in the Gym so you can move faster and spend less.

import re

from datasets import load_dataset

from modal_dojo import (
    DatasetConfig,
    Qwen3_5_4B,
    Qwen3_5_4B_Recipe,
    TrainConfig,
    TrainingGroup,
    TrainingRun,
)

# ## Define the training base
#
# We'll start by creating the shared model, dataset, and base training recipe
# all sweep runs will share. To stay focused on how to do sweeps, we'll keep
# this code minimal and only tune a few parameters, but the sky's the limit!

model = Qwen3_5_4B()


def _letter_answer(row) -> bool:
    return bool(re.fullmatch(r"[A-J]", str(row["expected_answer"]).strip().upper()))


class OpenScienceDataset(DatasetConfig):
    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> str:
        return "label"

    def rows(self):
        ds = load_dataset(
            "nvidia/OpenScienceReasoning-2", split="train", streaming=True
        )
        kept = 0
        for row in ds:
            if not _letter_answer(row):
                continue
            yield {
                "messages": [{"role": "user", "content": row["input"]}],
                "label": str(row["expected_answer"]).strip().upper(),
            }
            kept += 1
            if kept >= 80:
                break


train_dataset = OpenScienceDataset()

_BOXED_RE = re.compile(r"\\boxed\{([A-J])\}", re.IGNORECASE)


async def letter_rm(args, sample, **kwargs) -> float:
    matches = _BOXED_RE.findall(sample.response or "")
    pred = matches[-1].upper() if matches else ""
    return float(bool(pred) and pred == sample.label)


base = TrainConfig(
    model=model,
    dataset=train_dataset,
    recipe=Qwen3_5_4B_Recipe(
        custom_rm_function=letter_rm,
    ),
)

# ## Create the sweep
#
# We'll pass in the parameters we wish to test, then preview the set of
# runs the grid search will kick off once we're ready.

group = TrainingGroup(
    base=base,
    grid={
        "recipe.lr": [5e-7, 5e-6],
        "recipe.rollout_temperature": [0.8, 1.0],
    },
)
configs = group.get_train_configs()
print(f"{len(configs)} runs in group {group.group_id}:")
for cfg in configs:
    print(f"- lr={cfg.recipe.lr:<8}, temp={cfg.recipe.rollout_temperature}")

# ## Launch it!
#
# Once it all looks good, `.launch()` it!

if __name__ == "__main__":
    launches = group.launch()
    print(f"group {group.group_id}: {len(launches)} runs launched")
    for launch in launches:
        print(
            f"- {launch.training_run_id}, app={launch.modal_app_id}, group_id={launch.group_id}"
        )
    if group.failures:
        for overrides, err in group.failures:
            print(f"- FAILED {overrides}: {err}")

    results = TrainingRun.wait_all(launches)
    print(f"group {group.group_id}: {len(results)} runs completed")
    for run in results:
        print(f"completed {run.training_run_id} (group_id={run.group_id})")
