---
order: 0
---

# Agent-driven training

Agents are particularly useful when you need to validate hypotheses or run many experiments in parallel. However, they are less effective when forced to create and sift through thousands of lines of configuration files and training scripts. The Modal Dojo solves this with an intuitive API, a CLI for maximum observability into the run status, and skills that teach agents best practices such as smoking runs and tactics for debugging.

This guide demonstrates how to effectively use agents with the Modal Dojo by getting Claude to post-train Qwen3.5-4B to respond only in [rhyme](https://open.spotify.com/episode/5txYOHA44zWiSgNK623Epp).

## Set up

First, we'll install `modal-dojo`:

```bash
uv add 'modal-dojo @ git+https://github.com/modal-projects/modal-dojo.git@main'
```

Then, we'll install the provided skills into our current project:

```bash
modal-dojo skills install
```

The main skill agents use is `agent-driven-training`, which lays out the RL training lifecycle:

- Ask before making choices that change model behavior or GPU cost.
- Catch dataset and reward bugs locally before they waste GPU time.
- Scale up only after smoke runs indicate the training pipeline is healthy.
- Inspect actual model outputs to verify that higher rewards induce the intended behavior.
- Investigate suspicious reward trends to prevent [reward hacking](https://en.wikipedia.org/wiki/Reward_hacking).

To learn more about the CLI and the provided skills, see the [reference page](https://dojo.modal.dev/reference/cli).

## Let it cook

Here's the example prompt:

```txt
using Modal Dojo, train Qwen3.5-4B to speak only in rhymes.
```

Since it is just writing Python code, we can easily inspect what it wrote.

First, it loaded questions from [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) and added a descriptive system prompt:

```python
from datasets import load_dataset

from modal_dojo import DatasetConfig

SYSTEM_PROMPT = (
    "You only speak in rhyming couplets. Answer every request as a short poem "
    "where each pair of consecutive lines rhymes."
)
CATEGORIES = {"open_qa", "general_qa", "brainstorming", "creative_writing"}


class DollyQuestions(DatasetConfig):
    def __init__(self, start: int, stop: int) -> None:
        self.start, self.stop = start, stop

    def cache_key(self) -> str:
        return f"dolly-rhyme-{self.start}-{self.stop}"

    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> None:
        return None

    def rows(self):
        questions = dict.fromkeys(
            row["instruction"]
            for row in load_dataset("databricks/databricks-dolly-15k", split="train")
            if not row["context"].strip()
            and row["category"] in CATEGORIES
            and 0 < len(row["instruction"]) <= 300
        )
        for question in list(questions)[self.start : self.stop]:
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]
            }
```

Next, it defined the reward function. As our [intro tutorial](https://dojo.modal.dev/tutorials/rl_basics) shows, NLTK's [CMU Pronouncing Dictionary](https://github.com/prosegrinder/python-cmudict) is a useful library for determining if prose rhymes.

<details>
<summary>What's going on here</summary>

The reward function finds phonemes from each line's last stressed vowel onward and compares line endings under both the AABB and ABAB rhyme schemes. After some initial testing, the agent found two exploits the model took advantage of:

- Words missing from the dictionary fell back to matching their last three letters, so the model invented words like "qouls" to rhyme with "souls".
- Longer poems weren't penalized, so answers grew until they were cut off at the response length limit.

Luckily, these are simple problems that can be detected, and the agent implemented anti-gaming measures accordingly.

</details>

```python
import re

import nltk
from nltk.corpus import cmudict

_cmudict_cache = {}


def _rhyme_tails(word: str) -> set[tuple[str, ...]]:
    if not _cmudict_cache:
        nltk.download("cmudict", quiet=True)
        _cmudict_cache.update(cmudict.dict())
    tails = set()
    for phones in _cmudict_cache.get(word, []):
        stressed = [i for i, p in enumerate(phones) if p[-1] in "12"]
        if stressed:
            tails.add(tuple(phones[stressed[-1] :]))
    return tails


def _rhymes(a: str, b: str) -> bool:
    return bool(a and b and a != b and _rhyme_tails(a) & _rhyme_tails(b))


def _end_word(line: str) -> str:
    words = re.findall(r"[a-z]+", line.lower())
    return words[-1] if words else ""


def _pair_fraction(ends: list[str], pairs: list[tuple[int, int]]) -> float:
    return sum(_rhymes(ends[i], ends[j]) for i, j in pairs) / len(pairs)


def score_rhyme(response: str) -> float:
    lines = [
        re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line).strip()
        for line in response.splitlines()
    ]
    lines = [line for line in lines if line]
    if not 4 <= len(lines) <= 12 or any(len(line.split()) > 25 for line in lines):
        return 0.0
    if len({line.casefold() for line in lines}) < 0.7 * len(lines):
        return 0.0
    ends = [_end_word(line) for line in lines]
    aabb = [(i, i + 1) for i in range(0, len(ends) - 1, 2)]
    abab = [(s + k, s + k + 2) for s in range(0, len(ends) - 3, 4) for k in (0, 1)]
    return max(_pair_fraction(ends, aabb), _pair_fraction(ends, abab))
```

Of course, we still care that the model answers the question. For demonstration purposes, we simply halve the score of a response if it doesn't share any words with the question.

```python
from modal_dojo import Qwen3_5_4B

STOPWORDS = set(
    "that this with from have been were they them their what when where which "
    "your about into over then than also just only very more most such other "
    "because while would could should there these those some many will does "
    "each both after before being under again".split()
)

model = Qwen3_5_4B()


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOPWORDS}


async def rhyme_rm(args, sample, **kwargs) -> float:
    prompt = str(sample.prompt)
    user = re.search(r"<\|im_start\|>user\s*(.*?)<\|im_end\|>", prompt, re.DOTALL)
    question = user.group(1) if user else prompt
    response = model.parse_response(sample.response).content or ""
    score = score_rhyme(response)
    question_words = _content_words(question)
    if question_words and not question_words & _content_words(response):
        score *= 0.5
    return score
```

Then, it wrote the training code:

```python
import os

from modal_dojo import Qwen3_5_4B_Recipe, TrainConfig

NUM_ROLLOUT = int(os.environ.get("NUM_ROLLOUT", "1"))

config = TrainConfig(
    model=model,
    dataset=DollyQuestions(0, 2000),
    eval_dataset=DollyQuestions(2000, 2050),
    recipe=Qwen3_5_4B_Recipe(
        num_rollout=NUM_ROLLOUT,
        save_interval=NUM_ROLLOUT,
        rollout_batch_size=16,
        n_samples_per_prompt=8,
        global_batch_size=16,
        rollout_max_response_len=512,
        apply_chat_template_kwargs='{"enable_thinking": false}',
        custom_rm_function=rhyme_rm,
        image_overlay=lambda image: image.run_commands(
            "uv pip install --system aiohttp 'nltk>=3.8.0'",
            "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
        ),
    ),
)

if __name__ == "__main__":
    run = config.train()
    print(f"run id: {run.training_run_id}")
    print(f"checkpoint: {run.latest_checkpoint().path}")
```

Before making [GPUs go Brrr](https://hazyresearch.stanford.edu/blog/2024-05-12-tk), the provided skill prompts the agent to test the reward locally and run smoke tests.

```bash
NUM_ROLLOUT=1 uv run rhyme.py
NUM_ROLLOUT=10 uv run rhyme.py
NUM_ROLLOUT=50 uv run rhyme.py
```

## Monitor runs

Throughout the run, the agent had the following commands at its disposal:

- Confirm a run was launched successfully:

```bash
modal-dojo run list --since 2h --json
```

- See the progress of a run in more detail:

```bash
modal-dojo run get <run-id> --verbose --json
```

- Inspect the logs of a failing or hanging run:

```bash
modal-dojo run logs <run-id> --json
modal-dojo run logs <run-id> --follow --json
modal-dojo run logs <run-id> --search "checkpoint" --json
```

- Observe the raw model responses:

```bash
modal-dojo run trace <run-id> --out ./traces --yes --json
```

## Results

Over 50 rollouts on one H100, the mean reward rose from 0.20 to about 0.95.

Here's what it looks like in action:

<video controls playsinline width="100%">
  <source src="https://modal-cdn.com/cdnbot/agent-1-fast_be9e2663.webm" type="video/webm">
  <a href="https://modal-cdn.com/cdnbot/agent-1-fast_be9e2663.webm">Watch the agent-driven training demo.</a>
</video>
