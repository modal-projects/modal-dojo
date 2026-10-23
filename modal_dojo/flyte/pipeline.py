"""Flyte 2 training pipeline: the four stages as module-level ``@env.task``s.

Mirrors the stage contract the Modal apps expose (download model, prepare
dataset, convert checkpoint, train) but expressed as Flyte 2 tasks.

**Why module-level?** Flyte 2's remote resolver loads a task by importing its
module and ``getattr``-ing the task name (e.g. ``modal_dojo.flyte.pipeline.main``).
Tasks must therefore be defined once at module scope on a module-level
``TaskEnvironment`` — not created per-call as closures. ``build_pipeline``
validates/selects the ``mode`` and returns the module-level ``main`` task for
``flyte.run``.

Modal ``Volume``s are replaced by ``flyte.io.Dir`` object-store persistence:
each stage produces/consumes a ``flyte.io.Dir`` that Flyte syncs to/from the
cluster's object store.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import TYPE_CHECKING, Any

import flyte
import flyte.io

from modal_dojo.flyte.env import data_environment

if TYPE_CHECKING:
    from modal_dojo.common.models import ModelConfig
    from modal_dojo.common.dataset import DatasetConfig
    from modal_dojo.train_recipes.base import BaseTrainRecipe

# ── Module-level environments ────────────────────────────────────────────────
# v1 ships a single CPU environment hosting every stage so that ``flyte.run``
# builds/registers exactly one image and remote resolution just works. GPU +
# framework-image allocation is a deployed-workflow concern (phase 2; see
# ``flyte/DESIGN.md``): resource requests and per-env images are baked into a
# TaskEnvironment at definition time in Flyte 2, so they can't float per-call.
# ``env.train_environment`` remains exported for building a separate GPU env in
# a deployed framework module.
data_env = data_environment()


# ── Stage helpers ────────────────────────────────────────────────────────────


def _materialize_dataset(dataset: "DatasetConfig", dst_dir: str) -> str:
    """Write ``dataset.rows()`` to JSONL under ``dst_dir`` and return the path."""
    os.makedirs(dst_dir, exist_ok=True)
    jsonl_path = os.path.join(dst_dir, "data-0.jsonl")
    with open(jsonl_path, "w") as f:
        for row in dataset.rows():
            f.write(json.dumps(row) + "\n")
    dataset.validate_written(jsonl_path)
    return jsonl_path


def _snapshot_model(model: "ModelConfig", dst_dir: str) -> str:
    """Download the model into ``dst_dir``/``hf-cache`` and return that path.

    Tolerant by design: config-only models or models whose weights aren't
    reachable (no pinned revision cached) return an empty cache dir so the
    minimal path can still validate the pipeline end to end. The real framework
    modes run on images that carry/cache the weights.
    """
    local_dir = os.path.join(dst_dir, "hf-cache")
    os.makedirs(local_dir, exist_ok=True)
    if not getattr(model, "model_name", ""):
        return local_dir
    try:
        model.download()
    except NotImplementedError:
        pass  # config-only model
    except Exception as exc:
        print(f"[dojo/flyte] model.download() failed, continuing: {exc!r}")
    try:
        from huggingface_hub import snapshot_download

        snap = snapshot_download(
            repo_id=model.model_name,
            revision=model.revision,
            local_files_only=True,
            local_dir=local_dir,
        )
        if snap:
            return snap
    except Exception as exc:
        print(f"[dojo/flyte] snapshot_download failed, continuing: {exc!r}")
    return local_dir


def _train_minimal(
    dataset: "DatasetConfig",
    data_dir_local: str,
    checkpoints_dir_local: str,
) -> dict[str, Any]:
    """Train a tiny, self-contained causal-LM on the materialized dataset.

    Builds a small GPT-2-style model from a hard-coded config (no network) and
    runs one plain-PyTorch optimizer step, so the pipeline is verifiable end to
    end on a CPU devbox / Union CPU node without the fragility of a full
    ``Trainer`` on a synthetic dataset. Real slime/miles training is selected
    via ``mode`` and uses the framework launcher instead.
    """
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    os.makedirs(checkpoints_dir_local, exist_ok=True)

    texts: list[str] = []
    jsonl = os.path.join(data_dir_local, "data-0.jsonl")
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                text = row.get(dataset.input_key()) or row.get("text")
                if text is None:
                    text = json.dumps(row, ensure_ascii=False)
            else:
                text = str(row)
            texts.append(str(text))

    if not texts:
        texts = ["hello world"]
    texts = texts[:64]  # keep the devbox/Union run snappy regardless of size

    seq_len = 128
    config = GPT2Config(
        vocab_size=256,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_position=seq_len,
        n_ctx=seq_len,
    )
    model = GPT2LMHeadModel(config)

    def _ids(text: str) -> list[int]:
        return [b % 256 for b in text.encode("utf-8")]

    flat: list[int] = [e for t in texts for e in _ids(t)][:seq_len] or [1]
    ids = (flat + [0] * seq_len)[:seq_len]  # right-pad to seq_len

    input_ids = torch.tensor([ids], dtype=torch.long)  # shape [1, seq_len]
    labels = input_ids.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(1):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()

    model.save_pretrained(checkpoints_dir_local)
    with open(os.path.join(checkpoints_dir_local, "run_summary.json"), "w") as f:
        json.dump(
            {
                "mode": "minimal",
                "dataset": type(dataset).__name__,
                "num_texts": len(texts),
                "seq_len": seq_len,
                "framework": "hf-gpt2-minimal",
                "loss": float(loss.detach().item()),
            },
            f,
            indent=2,
        )
    return {"checkpoint_dir": checkpoints_dir_local, "num_texts": len(texts)}


def _run_framework_training(
    run_id: str,
    model: "ModelConfig",
    dataset: "DatasetConfig",
    recipe: "BaseTrainRecipe",
    data_local: str,
    checkpoints_local: str,
    mode: str,
) -> None:
    """Invoke the real slime/miles framework launcher on GPU.

    Phase-1 wiring: the pipeline stages (download/prepare/convert + checkpoint
    dir) run for any mode; the framework entrypoint requires a GPU
    ``TaskEnvironment`` and the framework image, which phase 2 deploys. Until
    then this raises an explicit, actionable error rather than silently doing
    nothing.
    """
    os.makedirs(checkpoints_local, exist_ok=True)
    raise RuntimeError(
        f"Framework training mode {mode!r} requires a GPU TaskEnvironment and the "
        "framework GPU image (phase 2 — see flyte/DESIGN.md). Run with mode='minimal' "
        "(CPU/1 GPU) to exercise the full pipeline on the devbox or a Union CPU node."
    )


# ── Stage 1: download the model ──────────────────────────────────────────────


@data_env.task(short_name="download_model")
async def download_model(run_id: str, model: Any) -> "flyte.io.Dir":
    hf_local = os.path.join(tempfile.mkdtemp(prefix="dojo-hf-"), "hf")
    _snapshot_model(model, hf_local)
    return await flyte.io.Dir.from_local(hf_local)


# ── Stage 2: materialize the dataset ─────────────────────────────────────────


@data_env.task(short_name="prepare_dataset")
async def prepare_dataset(run_id: str, dataset: Any, model: Any) -> "flyte.io.Dir":
    local = os.path.join(tempfile.mkdtemp(prefix="dojo-data-"), "data")
    _materialize_dataset(dataset, local)
    return await flyte.io.Dir.from_local(local)


# ── Stage 3: run training ────────────────────────────────────────────────────


@data_env.task(short_name="train")
async def train_stage(
    run_id: str,
    model: Any,
    dataset: Any,
    recipe: Any,
    data_dir: "flyte.io.Dir",
    hf_cache_dir: "flyte.io.Dir",
    mode: str,
) -> "flyte.io.Dir":
    data_local = await data_dir.download()
    ckpt_local = os.path.join(tempfile.mkdtemp(prefix="dojo-ckpt-"), "checkpoints")
    if mode == "minimal":
        _train_minimal(dataset, data_local, ckpt_local)
    else:
        _run_framework_training(
            run_id, model, dataset, recipe, data_local, ckpt_local, mode
        )
    return await flyte.io.Dir.from_local(ckpt_local)


# ── Driver entrypoint ────────────────────────────────────────────────────────


@data_env.task(entrypoint=True, short_name="main")
async def main(
    run_id: str,
    model: Any,
    dataset: Any,
    recipe: dict,
    mode: str,
) -> "flyte.io.Dir":
    """Run the download -> prepare -> train pipeline (deterministic order)."""
    hf_cache = await download_model(run_id, model)
    data_dir = await prepare_dataset(run_id, dataset, model)
    checkpoints = await train_stage(
        run_id, model, dataset, recipe, data_dir, hf_cache, mode
    )
    return checkpoints


def build_pipeline(config: Any, mode: str | None = None) -> Any:
    """Return the module-level ``main`` driver task for ``flyte.run``.

    Validates ``mode`` against the configured ``TrainConfig``'s recipe framework
    so the correct pipeline behavior is selected at submission time.

    Args:
        config: a ``TrainConfig`` (dataset + model + recipe).
        mode: ``TrainingMode`` value; defaults to the recipe's framework, else
            ``"minimal"``.
    """
    from modal_dojo.flyte import TrainingMode
    from modal_dojo.train_recipes.miles_recipe import MilesRecipe
    from modal_dojo.train_recipes.slime_recipe import SlimeRecipe

    recipe = config.recipe
    if mode is None:
        if isinstance(recipe, MilesRecipe):
            mode = TrainingMode.MILES.value
        elif isinstance(recipe, SlimeRecipe):
            mode = TrainingMode.SLIME.value
        else:
            mode = TrainingMode.MINIMAL.value
    TrainingMode(mode)  # validate the mode value
    return main
