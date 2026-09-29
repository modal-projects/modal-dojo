---
order: 0
---

# Agent-driven training

Agents are particularly useful when you need to validate hypotheses or run many experiments in parallel. However, they are less effective when forced to create and sift through thousands of lines of configuration files and training scripts. The Modal Dojo solves this with an intuitive API, a CLI for maximum observability into the run status, and skills that teach agents best practices such as smoking runs and tactics for debugging.

This guide demonstrates how to effectively use agents with the Modal Dojo by getting Claude to post-train a model of its choosing to respond only in [rhyme](https://open.spotify.com/episode/5txYOHA44zWiSgNK623Epp).

## Set up

First, we'll install the `modal-dojo` CLI:

```bash
pip install -q git+https://github.com/modal-projects/modal-dojo.git@main
modal-dojo --help
```

Then, we'll install the provided skills into our current project:

```bash
modal-dojo skills install
```

The main skill agents should use is `agent-driven-training`, which lays out the RL training lifecycle:

- Ask before making choices that change model behavior or GPU cost.
- Catch dataset and reward bugs locally before they waste GPU time.
- Scale up only after smoke runs indicate the training pipeline is healthy.
- Inspect actual model outputs to verify that higher rewards induce the intended behavior.
- Investigate suspicious reward trends to prevent [reward hacking](https://en.wikipedia.org/wiki/Reward_hacking).

To learn more about the CLI and the provided skills, see the [reference page](https://gym.modal.dev/reference/cli).

## Let it cook

Here's the example prompt:

```txt
can you post-train a model to rhyme in its output
```

We leave it ambiguous to demonstrate that when empowered with the right tools and skills, agents are capable of making sensible choices. Here, it chose to train [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) on prompts taken from [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca).

Since it is just writing Python code, we can easily inspect what it wrote. First, it loaded the dataset:

```python
from modal_dojo import HuggingFaceDataset

SYSTEM_PROMPT = (
    "You are a poet who answers every question in rhyme. Answer the question "
    "correctly and completely, but write the entire answer as verse: at least "
    "four lines, one clause per line, with line endings that rhyme in couplets "
    "(AABB). Do not write any prose, preamble, or explanation outside the verse."
)


rhyme_dataset = HuggingFaceDataset(
    hf_repo="tatsu-lab/alpaca",
    hf_split=f"train[:512]",
    input_column="instruction",
    output_column="output",
    input_format="text",
    system_prompt=SYSTEM_PROMPT,
)
```

Next, it defined the reward function. Here, we care about the model's ability to both rhyme and answer the user's question. As our [intro tutorial](https://gym.modal.dev/tutorials/rl_basics) shows, NLTK’s [CMU Pronouncing Dictionary](https://github.com/prosegrinder/python-cmudict) is a useful library for measuring the former.

<details>
<summary>What's going on here</summary>

The reward function finds phonemes from each line's last stressed vowel onward and compares line endings under both the AABB and ABAB rhyme schemes. After some initial testing, the agent found two exploits the model took advantage of:

- Words rhyme with themselves, so the model repeated the last word of the sentence.
- One-word lines are easy to write and rhyme, so the model found that being concise was better than trying its best.

Luckily, these are simple problems that can be detected, and the agent implemented anti-gaming measures accordingly.

</details>

```python
import re

_CMUDICT: dict = {}
_VOWELS = ("A", "E", "I", "O", "U")


def _cmudict() -> dict:
    if not _CMUDICT:
        import nltk
        from nltk.corpus import cmudict

        nltk.download("cmudict", quiet=True)
        _CMUDICT.update(cmudict.dict())
    return _CMUDICT


def _strip_thinking(text: str) -> str:
    """Drop a ``<think>`` block and any stray markdown bullets/numbering."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    return text.strip()


def _lines(text: str) -> list[str]:
    return [line.strip() for line in _strip_thinking(text).split("\n") if line.strip()]


def _end_word(line: str) -> str:
    words = re.findall(r"[a-zA-Z']+", line)
    return words[-1].lower().strip("'") if words else ""


def rhyme_tail(word: str) -> tuple:
    """Phonemes from the last stressed vowel onward, stress markers removed.

    Falls back to the last three letters for words the dictionary doesn't know
    (names, coinages), which is a decent orthographic proxy.
    """
    if not word:
        return ()
    phones = _cmudict().get(word)
    if not phones:
        return ("~", word[-3:])
    seq = phones[0]
    stressed = [i for i, p in enumerate(seq) if p[-1] in ("1", "2")]
    if stressed:
        start = stressed[-1]
    else:
        vowels = [i for i, p in enumerate(seq) if p[0] in _VOWELS]
        start = vowels[-1] if vowels else 0
    return tuple(re.sub(r"\d", "", p) for p in seq[start:])


def words_rhyme(a: str, b: str) -> bool:
    """True when two *different* words share a rhyme tail."""
    if not a or not b or a == b:
        return False
    return rhyme_tail(a) == rhyme_tail(b)


def _scheme_score(end_words: list[str], offset: int) -> float:
    """Fraction of rhyming pairs: offset 1 = AABB, offset 2 = ABAB."""
    pairs = []
    for start in range(0, len(end_words) - offset, 2 * offset):
        for k in range(offset):
            i, j = start + k, start + k + offset
            if j < len(end_words):
                pairs.append((end_words[i], end_words[j]))
    if not pairs:
        return 0.0
    return sum(words_rhyme(a, b) for a, b in pairs) / len(pairs)


def score_rhyme(response: str) -> float:
    """Rhyme quality of a response in ``[0, 1]``."""
    lines = _lines(response)
    if len(lines) < 2:
        return 0.0
    end_words = [_end_word(line) for line in lines]
    if not any(end_words):
        return 0.0

    scheme = max(_scheme_score(end_words, 1), _scheme_score(end_words, 2))
    distinct = len({w for w in end_words if w}) / len(end_words)
    substantial = sum(
        len(re.findall(r"[a-zA-Z']+", line)) >= 3 for line in lines
    ) / len(lines)
    length_factor = min(1.0, len(lines) / 4)
    return scheme * distinct * substantial * length_factor
```

Of course, we still care that the model answers the question. Interestingly, the agent decided to use a small sentence-embedding model to compare the response to the reference answer.

```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIR = "/opt/rhyme-embedder"

_EMBEDDER: dict = {}


def _embedder():
    """Mean-pooling MiniLM loaded once per worker from the baked image dir."""
    if not _EMBEDDER:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(EMBED_DIR)
        model = AutoModel.from_pretrained(EMBED_DIR)
        model.eval()
        _EMBEDDER["tokenizer"] = tokenizer
        _EMBEDDER["model"] = model
        _EMBEDDER["torch"] = torch
        print(f"[rhyme_rm] embedder ready: {EMBED_MODEL}")
    return _EMBEDDER


def embed(texts: list[str]) -> list:
    """L2-normalized mean-pooled sentence embeddings."""
    parts = _embedder()
    torch = parts["torch"]
    batch = parts["tokenizer"](
        texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    with torch.no_grad():
        out = parts["model"](**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def _lexical_overlap(a: str, b: str) -> float:
    """Token-F1 fallback used only if the embedder fails to load."""
    ta = {w for w in re.findall(r"[a-z']+", a.lower()) if len(w) > 3}
    tb = {w for w in re.findall(r"[a-z']+", b.lower()) if len(w) > 3}
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))


def score_relevance(response: str, reference: str) -> float:
    """Topical agreement with the reference answer, rescaled to ``[0, 1]``."""
    response = _strip_thinking(response)
    reference = (reference or "").strip()
    if not response or not reference:
        return 0.0
    try:
        vectors = embed([response, reference])
        cosine = float((vectors[0] * vectors[1]).sum())
    except Exception as exc:  # noqa: BLE001
        print(f"[rhyme_rm] embedder unavailable ({exc}); using lexical overlap")
        cosine = _lexical_overlap(response, reference)
    return max(0.0, min(1.0, (cosine - 0.10) / 0.45))


def rhyme_reward(response: str, reference: str) -> float:
    """Gated rhyme quality plus a smaller standalone relevance term."""
    rhyme = score_rhyme(response)
    relevance = score_relevance(response, reference)
    gate = min(1.0, relevance / 0.4)
    return gate * rhyme + 0.3 * relevance


async def rhyme_rm(args, sample, **kwargs) -> float:
    from modal_dojo import Qwen3_4B

    model = Qwen3_4B()
    response = model.parse_response(getattr(sample, "response", "") or "")
    reference = getattr(sample, "label", "") or ""
    return rhyme_reward(response.content or "", str(reference))
```

Then, it wrote the training code:

```python
from modal_dojo import Qwen3_4B, TrainConfig
from modal_dojo.train_recipes.slime_recipe import Qwen3_4B_Recipe


def _image_overlay(image):
    return image.run_commands(
        "uv pip install --system 'nltk>=3.8.0'",
        "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
        # Download through a scratch cache so the image does not leave files
        # where the shared Hugging Face Volume needs to mount.
        "HF_HOME=/tmp/hf-build HF_HUB_CACHE=/tmp/hf-build "
        'python -c "from huggingface_hub import snapshot_download; '
        f"snapshot_download('{EMBED_MODEL}', local_dir='{EMBED_DIR}')\"",
        "rm -rf /tmp/hf-build /root/.cache/huggingface",
    )


def build_config(*, num_rollout: int, save_interval: int) -> TrainConfig:
    return TrainConfig(
        model=Qwen3_4B(),
        dataset=rhyme_dataset,
        recipe=Qwen3_4B_Recipe(
            custom_rm_function=rhyme_rm,
            num_rollout=num_rollout,
            rollout_batch_size=16,
            n_samples_per_prompt=8,
            rollout_max_response_len=1024,
            save_interval=save_interval,
            apply_chat_template_kwargs='{"enable_thinking": false}',
            capture_trace=True,
            trace_sample_limit=16,
            image_overlay=_image_overlay,
        ),
    )
```

Before making [GPUs go Brrr](https://hazyresearch.stanford.edu/blog/2024-05-12-tk), the provided skill prompts the agent to run smoke tests.

Following suit, it does a one-step run:

```python
training_run = build_config(num_rollout=1, save_interval=1)
run = training_run.launch()
print(f"training_run_id: {run.training_run_id}")
```

## Monitor runs

Throughout the run, the agent used the following commands to:

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
modal-dojo run trace <run-id> --out ./traces --dry-run --json
modal-dojo run trace <run-id> --out ./traces --yes --json
```

## Results

<video controls playsinline width="100%">
  <source src="/agent-driven-training-rhyme.mp4" type="video/mp4">
  <source src="https://gym.modal.dev/agent-driven-training-rhyme.mp4" type="video/mp4">
  <a href="https://gym.modal.dev/agent-driven-training-rhyme.mp4">Watch the agent-driven training demo.</a>
</video>

After 46 minutes of training, the model makes all responses [rhyme](#1) [damn well](#2) while still [answering the user](#3).

1. <a id="1"></a>Non-rhyming answers reduced from 35/128 to 0/128.
2. <a id="2"></a>Rhyme score improved from 0.475 to 0.908.
3. <a id="3"></a>Relevance remained essentially the same from 0.877 to 0.886.
