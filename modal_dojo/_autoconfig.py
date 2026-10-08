"""Autoconfig: the dashboard's recipe catalogue and the sweeps that fill it.

The page reads ``GET /api/autoconfig/catalogue`` and kicks off sweeps with
``POST /api/autoconfig/sweeps``, which spawns ``launch_autoconfig_sweep`` in
the dashboard's Modal app and answers with the call id. The page then polls
``GET /api/autoconfig/sweeps/{operation_id}`` until the launch finishes, the
same submit-and-poll shape as autoinference's recipe-seed API.

Agents drive the same machinery through ``modal-dojo autoconfig``: list the
registry models and their recipe knobs, dry-run a plan with the GPU memory
estimate per variant, then submit it as a sweep.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import inspect
import json
import re
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from modal_dojo.common.catalogue import (
    AUTOCONFIG_ENTRY_TAG,
    AUTOCONFIG_GROUP_PREFIX,
    AUTOCONFIG_RECIPE_TAG,
    AUTOCONFIG_TAGS,
    CATALOGUE,
    STEPS,
    build_catalogue,
    context_length,
)
from modal_dojo.common.run_summary import RunSummary

SWEEP_FUNCTION_NAME = "launch_autoconfig_sweep"
RATES_TTL_S = 3600.0
# ``modal.FunctionCall.from_id`` accepts any id; only ours are sweeps.
_CALL_ID = re.compile(r"^fc-[A-Za-z0-9]+$")

# Recipe fields an agent reaches for first when shaping a sweep.
HIGHLIGHTED_KNOBS = (
    "gpu_type",
    "actor_num_nodes",
    "actor_num_gpus_per_node",
    "rollout_num_gpus_per_engine",
    "tensor_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "rollout_max_response_len",
    "max_tokens_per_gpu",
    "rollout_batch_size",
    "n_samples_per_prompt",
    "rollout_temperature",
    "global_batch_size",
    "lr",
    "eps_clip",
    "use_kl_loss",
    "kl_loss_coef",
    "entropy_coef",
    "optimizer_cpu_offload",
    "lora_rank",
)
# Launcher plumbing, callables, and images: not flags a sweep sets.
_HIDDEN_KNOBS = frozenset(
    {
        "name",
        "app_tags",
        "environment",
        "metrics",
        "image_overlay",
        "patch_files",
        "image_run_commands",
        "image_env",
        "train_function_kwargs",
        "local_slime",
        "slime_git_repository",
        "slime_git_revision",
        "data_volume_name",
        "slime_model_script",
        "docker_image",
        "save",
        "load",
        "ref_load",
        "hf_checkpoint",
        "source_hf_checkpoint",
        "megatron_conversion_hf_checkpoint",
        "update_weight_disk_dir",
    }
)


class DatasetSpec(BaseModel):
    """A ``HuggingFaceDataset`` the sweep trains on."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    hf_repo: str = Field(min_length=1)
    hf_split: str = Field(default="train", min_length=1)
    hf_config: str | None = None
    input_column: str = Field(min_length=1)
    output_column: str | None = None
    input_format: Literal["text", "messages", "raw"] = "text"


# The catalogue benchmarks recipes on SWE-bench-style tasks: SWE-smith's
# training split pairs each problem statement with the patch that fixed it.
DEFAULT_DATASET = DatasetSpec(
    hf_repo="SWE-bench/SWE-smith",
    hf_split="train",
    input_column="problem_statement",
    output_column="patch",
)

# Names an intent can use instead of spelling out a Hugging Face dataset.
DATASET_PRESETS: dict[str, DatasetSpec] = {
    "swe-bench": DEFAULT_DATASET,
    "swe-smith": DEFAULT_DATASET,
    "gsm8k": DatasetSpec(
        hf_repo="openai/gsm8k",
        hf_config="main",
        input_column="question",
        output_column="answer",
    ),
}


def resolve_dataset(
    name: str,
    *,
    split: str | None = None,
    input_column: str | None = None,
    output_column: str | None = None,
) -> DatasetSpec:
    """A preset by name (``swe-bench``), or a Hugging Face repo with its columns."""
    key = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    preset = DATASET_PRESETS.get(key)
    if preset is not None:
        return preset.model_copy(
            update={
                k: v
                for k, v in {
                    "hf_split": split,
                    "input_column": input_column,
                    "output_column": output_column,
                }.items()
                if v is not None
            }
        )
    if not input_column:
        presets = ", ".join(sorted(DATASET_PRESETS))
        raise ValueError(
            f"dataset {name!r} is not a preset ({presets}); pass its input "
            "column (and output column) to use a Hugging Face dataset"
        )
    return DatasetSpec(
        hf_repo=name,
        hf_split=split or "train",
        input_column=input_column,
        output_column=output_column,
    )


class SweepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Registry model names (``modal-dojo autoconfig models``) to benchmark.
    entries: list[str] = Field(min_length=1)
    steps: int = Field(default=STEPS, ge=1, le=100)
    # ``TrainingGroup`` grid: dotted ``TrainConfig`` paths to the values to try.
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    # Dotted ``TrainConfig`` paths set on every variant before the grid applies.
    overrides: dict[str, Any] = Field(default_factory=dict)
    # Longest response, in tokens; sets the recipe's response and packing caps.
    context_length: int | None = Field(default=None, ge=256, le=1_048_576)
    # Launch variants the memory estimate says will not fit their GPU.
    allow_oom: bool = False
    name: str | None = Field(default=None, max_length=60)
    dataset: DatasetSpec = DEFAULT_DATASET


class SweepError(BaseModel):
    type: str
    message: str


class SweepOperation(BaseModel):
    operation_id: str
    group_id: str
    status: Literal["pending", "succeeded", "failed"]
    result: dict[str, Any] | None = None
    error: SweepError | None = None


class SweepOperations(Protocol):
    def submit(self, request: SweepRequest, group_id: str) -> SweepOperation: ...

    def get(self, operation_id: str) -> SweepOperation: ...


class SweepBackendError(RuntimeError):
    """Modal could not serve the sweep operation."""


class SweepNotFoundError(LookupError):
    """No such sweep operation."""


def sweep_group_id(name: str | None) -> str:
    """A fresh ``autoconfig-…`` group id; the page finds sweep runs by it."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    parts = [AUTOCONFIG_GROUP_PREFIX, *([slug] if slug else []), secrets.token_hex(3)]
    return "-".join(parts)


# ── Models and knobs ─────────────────────────────────────────────────────────


def resolve_model(name: str) -> Any:
    """The registry entry for ``name`` (short name or HF id); RL entries only."""
    from modal_dojo.common.models.validation import _ValidationConfig

    config = _ValidationConfig.find(name)
    if config.loss_type != "policy_loss":
        raise ValueError(
            f"{config.name!r} is an SFT validation entry; Autoconfig sweeps "
            "benchmark RL recipes"
        )
    return config


def recipe_class_for(config: Any) -> type:
    """The recipe class a sweep of the registry model trains with.

    The catalogue pins a public recipe class per entry; other registry models
    use their framework's base recipe, the one CI validates them with.
    """
    import modal_dojo
    from modal_dojo.common.framework import Framework

    for entry in CATALOGUE:
        if entry.name == config.name:
            return getattr(modal_dojo, entry.recipe)
    if config.framework is Framework.SLIME:
        from modal_dojo.train_recipes.slime_recipe import SlimeRecipe

        recipe = SlimeRecipe.get_base_recipe(config.model_config())
    else:
        from modal_dojo.train_recipes.miles_recipe import MilesRecipe

        recipe = MilesRecipe.get_base_recipe(config.model_config())
    if recipe is None:
        raise ValueError(f"no base recipe for model {config.name!r}")
    return type(recipe)


def recipe_fields(recipe: Any) -> dict[str, Any]:
    """Every field of a recipe instance, by name."""
    return {f.name: getattr(recipe, f.name) for f in dataclasses.fields(recipe)}


def model_summary(config: Any, recipe_cls: type) -> dict[str, Any]:
    from modal_dojo.train_recipes.gpu_allocation import (
        gpu_memory_gib,
        resolve_gpu_allocation,
    )

    recipe = recipe_cls()
    allocation = resolve_gpu_allocation(recipe, warn=False)
    return {
        "name": config.name,
        "model": config.model_name,
        "framework": config.framework.value,
        "recipe": recipe_cls.__name__,
        "in_catalogue": any(entry.name == config.name for entry in CATALOGUE),
        "gpu_type": recipe.gpu_type,
        "gpu_gib": gpu_memory_gib(recipe.gpu_type),
        "gpus": allocation.total_gpus,
        "nodes": allocation.total_nodes,
        "context_length": context_length(recipe_fields(recipe)),
    }


def list_models() -> list[dict[str, Any]]:
    """Every registry model a sweep can benchmark, with its recipe's defaults."""
    from modal_dojo.common.models.validation import _ValidationConfig

    models: list[dict[str, Any]] = []
    for config in _ValidationConfig.select():
        if config.loss_type != "policy_loss":
            continue
        try:
            recipe_cls = recipe_class_for(config)
        except ValueError:
            continue
        models.append(model_summary(config, recipe_cls))
    return sorted(models, key=lambda m: (not m["in_catalogue"], m["name"]))


def _field_docs(cls: type) -> dict[str, str]:
    """``name: description`` entries of the ``Args:`` sections along the MRO."""
    docs: dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        lines = inspect.cleandoc(klass.__dict__.get("__doc__") or "").splitlines()
        try:
            start = next(i for i, line in enumerate(lines) if line.strip() == "Args:")
        except StopIteration:
            continue
        name: str | None = None
        for line in lines[start + 1 :]:
            stripped = line.strip()
            if not stripped:
                continue
            indent = len(line) - len(line.lstrip())
            if indent == 0:
                break
            if re.fullmatch(r"\*?\*?[A-Za-z_][A-Za-z0-9_]*:", stripped):
                name = stripped.rstrip(":").lstrip("*")
                docs[name] = ""
            elif name is not None:
                docs[name] = f"{docs[name]} {stripped}".strip()
    return docs


def _type_name(annotation: Any) -> str:
    if isinstance(annotation, str):
        return annotation
    if isinstance(annotation, type):
        return annotation.__name__
    return repr(annotation).replace("typing.", "")


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=repr))


def recipe_knobs(name: str) -> dict[str, Any]:
    """The recipe fields a sweep of ``name`` can set, with defaults and docs."""
    config = resolve_model(name)
    recipe_cls = recipe_class_for(config)
    docs = _field_docs(recipe_cls)
    knobs: list[dict[str, Any]] = []
    for field in dataclasses.fields(recipe_cls):
        type_name = _type_name(field.type)
        if (
            field.name.startswith("_")
            or field.name in _HIDDEN_KNOBS
            or "Callable" in type_name
            or "modal." in type_name
        ):
            continue
        if field.default is not dataclasses.MISSING:
            default = field.default
        elif field.default_factory is not dataclasses.MISSING:
            default = field.default_factory()
        else:
            default = None
        knobs.append(
            {
                "path": f"recipe.{field.name}",
                "type": type_name,
                "default": _jsonable(default),
                "doc": docs.get(field.name, ""),
                "highlight": field.name in HIGHLIGHTED_KNOBS,
            }
        )
    knobs.sort(key=lambda k: (not k["highlight"], k["path"]))
    return {**model_summary(config, recipe_cls), "knobs": knobs}


# ── Planning ─────────────────────────────────────────────────────────────────


def context_overrides(tokens: int, recipe: Any) -> dict[str, Any]:
    """Recipe paths that let one response run to ``tokens`` tokens.

    The response cap is what the catalogue reports as context length. With
    dynamic batching the packing cap must hold a whole sample too, so it is
    raised when it is lower.
    """
    overrides: dict[str, Any] = {"recipe.rollout_max_response_len": tokens}
    if hasattr(recipe, "eval_max_response_len"):
        overrides["recipe.eval_max_response_len"] = tokens
    packing = getattr(recipe, "max_tokens_per_gpu", None)
    if getattr(recipe, "use_dynamic_batch_size", False) and (packing or 0) < tokens:
        overrides["recipe.max_tokens_per_gpu"] = tokens
    return overrides


def plan_sweep(request: SweepRequest, group_id: str) -> list[tuple[str, Any]]:
    """Every ``(model name, TrainConfig)`` the sweep launches.

    Each model's recipe class, at its defaults plus ``steps`` rollouts, the
    context-length caps, and the request's overrides, is the base of one
    ``TrainingGroup`` over the grid. Every variant is tagged with the model
    and recipe it benchmarks so the catalogue can find it.

    Raises:
        ValueError: An unknown model, or a path that is not a ``TrainConfig`` field.
    """
    from modal_dojo.common.dataset import HuggingFaceDataset
    from modal_dojo.common.train import TrainConfig
    from modal_dojo.common.training_group import TrainingGroup

    dataset = HuggingFaceDataset(**request.dataset.model_dump())
    configs: list[tuple[str, Any]] = []
    for name in request.entries:
        config = resolve_model(name)
        recipe = recipe_class_for(config)(num_rollout=request.steps)
        base = TrainConfig(
            model=config.model_config(),
            dataset=dataset,
            eval_dataset=dataset,
            recipe=recipe,
        )
        fixed: dict[str, Any] = {}
        if request.context_length:
            fixed.update(context_overrides(request.context_length, recipe))
        fixed.update(request.overrides)
        grid = {**{path: [value] for path, value in fixed.items()}, **request.grid}
        group = TrainingGroup(base, grid, name=group_id)
        for overrides, cfg in group.iter_variants():
            cfg.group_id = group_id
            cfg.group_axes = list(request.grid)
            cfg.group_overrides = {
                AUTOCONFIG_ENTRY_TAG: config.name,
                AUTOCONFIG_RECIPE_TAG: type(cfg.recipe).__name__,
                **{k: v for k, v in overrides.items() if k in request.grid},
            }
            configs.append((config.name, cfg))
    return configs


def variant_overrides(cfg: Any) -> dict[str, Any]:
    return {
        k: v for k, v in (cfg.group_overrides or {}).items() if k not in AUTOCONFIG_TAGS
    }


def describe_plan(request: SweepRequest) -> dict[str, Any]:
    """What a sweep would launch: each variant's shape and GPU memory estimate.

    Costs nothing; the agent reads it to fix flags before submitting.
    """
    from modal_dojo.common.memory_estimate import estimate_gpu_memory
    from modal_dojo.train_recipes.gpu_allocation import resolve_gpu_allocation

    variants: list[dict[str, Any]] = []
    for entry_name, cfg in plan_sweep(request, f"{AUTOCONFIG_GROUP_PREFIX}-plan"):
        recipe = cfg.recipe
        allocation = resolve_gpu_allocation(recipe, warn=False)
        estimate = estimate_gpu_memory(recipe, cfg.model, fetch_architecture=False)
        fields = recipe_fields(recipe)
        variants.append(
            {
                "entry": entry_name,
                "model": cfg.model.model_name,
                "recipe": type(recipe).__name__,
                "overrides": _jsonable(variant_overrides(cfg)),
                "gpu_type": recipe.gpu_type,
                "gpus": allocation.total_gpus,
                "nodes": allocation.total_nodes,
                "context_length": context_length(fields),
                "context_plan": cfg.context_plan_line(),
                "memory": None if estimate is None else estimate.to_dict(),
            }
        )
    oom = [v for v in variants if v["memory"] is not None and not v["memory"]["fits"]]
    return {
        "steps": request.steps,
        "dataset": request.dataset.model_dump(),
        "variants": variants,
        "fits": not oom,
        "oom_variants": len(oom),
    }


def validate_request(request: SweepRequest) -> dict[str, Any]:
    """Plan the sweep, rejecting bad paths and, unless allowed, OOM variants."""
    plan = describe_plan(request)
    if not plan["fits"] and not request.allow_oom:
        shapes = "; ".join(
            f"{v['entry']} {v['overrides']}: ~{v['memory']['peak_gib']} GiB "
            f"on {v['gpu_type']} ({v['memory']['gpu_gib']:g} GiB)"
            for v in plan["variants"]
            if v["memory"] is not None and not v["memory"]["fits"]
        )
        raise ValueError(
            f"{plan['oom_variants']} variant(s) exceed GPU memory by estimate: "
            f"{shapes}. Change the shape, or set allow_oom to launch anyway."
        )
    return plan


def launch_sweep(payload: Mapping[str, Any], group_id: str) -> dict[str, Any]:
    """Body of ``launch_autoconfig_sweep``: launch every variant, detached."""
    request = SweepRequest.model_validate(payload)
    launched: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for entry_name, cfg in plan_sweep(request, group_id):
        overrides = variant_overrides(cfg)
        try:
            run = cfg.launch(show_output=False)
        except Exception as exc:  # noqa: BLE001 — recorded per variant
            failures.append(
                {
                    "entry": entry_name,
                    "overrides": overrides,
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            )
            continue
        launched.append(
            {
                "entry": entry_name,
                "overrides": overrides,
                "training_run_id": run.training_run_id,
                "modal_app_id": run.modal_app_id,
            }
        )
    return {"group_id": group_id, "launched": launched, "failures": failures}


class ModalSweepOperations:
    """Sweeps as spawned calls of the dashboard app's launcher function."""

    def __init__(self, function: Any) -> None:
        self._function = function

    def submit(self, request: SweepRequest, group_id: str) -> SweepOperation:
        import modal

        try:
            call = self._function.spawn(request.model_dump(mode="json"), group_id)
        except modal.exception.Error as exc:
            raise SweepBackendError("could not submit sweep") from exc
        return SweepOperation(
            operation_id=call.object_id, group_id=group_id, status="pending"
        )

    def get(self, operation_id: str) -> SweepOperation:
        import modal

        if not _CALL_ID.match(operation_id):
            raise SweepNotFoundError("sweep operation not found")
        try:
            call = modal.FunctionCall.from_id(operation_id)
            result = call.get(timeout=0)
        except modal.exception.NotFoundError as exc:
            raise SweepNotFoundError("sweep operation not found") from exc
        except TimeoutError:
            return SweepOperation(
                operation_id=operation_id, group_id="", status="pending"
            )
        except modal.exception.Error as exc:
            raise SweepBackendError("could not read sweep") from exc
        except Exception as exc:  # noqa: BLE001 — the launcher raised
            return SweepOperation(
                operation_id=operation_id,
                group_id="",
                status="failed",
                error=SweepError(type=type(exc).__name__, message=str(exc)[:500]),
            )
        if not isinstance(result, dict):
            return SweepOperation(
                operation_id=operation_id,
                group_id="",
                status="failed",
                error=SweepError(
                    type="invalid_result", message="sweep returned a non-object"
                ),
            )
        return SweepOperation(
            operation_id=operation_id,
            group_id=str(result.get("group_id") or ""),
            status="succeeded",
            result=result,
        )


def billing_rates() -> Mapping[str, Any]:
    """Current GPU list prices of the workspace."""
    import modal

    return modal.Workspace.from_context().billing.rates()


def mount_autoconfig_api(
    web: Any,
    *,
    load_run_summaries: Callable[[], Awaitable[list[RunSummary]]],
    operations: SweepOperations,
    rates: Callable[[], Mapping[str, Any]] = billing_rates,
) -> Any:
    from fastapi import HTTPException, status
    from fastapi.concurrency import run_in_threadpool

    rates_cache: dict[str, Any] = {"at": None, "rates": {}, "priced_at": None}

    def priced_rates() -> tuple[Mapping[str, Any], str | None]:
        at = rates_cache["at"]
        if at is None or time.monotonic() - at > RATES_TTL_S:
            try:
                rates_cache["rates"] = dict(rates())
                rates_cache["priced_at"] = dt.date.today().isoformat()
            except Exception as exc:  # noqa: BLE001 — unpriced rows beat no rows
                print(f"[autoconfig] billing rates unavailable: {exc}", flush=True)
            rates_cache["at"] = time.monotonic()
        return rates_cache["rates"], rates_cache["priced_at"]

    @web.get("/api/autoconfig/catalogue")
    async def catalogue():
        summaries = await load_run_summaries()
        current, priced_at = await run_in_threadpool(priced_rates)
        return await run_in_threadpool(build_catalogue, summaries, current, priced_at)

    @web.get("/api/autoconfig/entries")
    async def entries():
        return [{"name": e.name, "recipe": e.recipe} for e in CATALOGUE]

    @web.get("/api/autoconfig/models")
    async def models():
        return await run_in_threadpool(list_models)

    @web.get("/api/autoconfig/models/{name}/knobs")
    async def knobs(name: str):
        try:
            return await run_in_threadpool(recipe_knobs, name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @web.post("/api/autoconfig/plans")
    async def plan(request: SweepRequest):
        try:
            return await run_in_threadpool(describe_plan, request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @web.post(
        "/api/autoconfig/sweeps",
        response_model=SweepOperation,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit_sweep(request: SweepRequest) -> SweepOperation:
        try:
            await run_in_threadpool(validate_request, request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            return await run_in_threadpool(
                operations.submit, request, sweep_group_id(request.name)
            )
        except SweepBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @web.get("/api/autoconfig/sweeps/{operation_id}", response_model=SweepOperation)
    async def get_sweep(operation_id: str) -> SweepOperation:
        try:
            return await run_in_threadpool(operations.get, operation_id)
        except SweepNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SweepBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return web
