"""Flyte ``TaskEnvironment`` / image / ``Resources`` builders.

Converts a ``TrainConfig``'s recipe into Flyte 2 infrastructure: the container
image, the CPU/GPU ``Resources`` request, and the ``TaskEnvironment`` groups
the pipeline stages live on.

Two environment groups mirror the two Modal function groups the framework apps
already use:

- ``dojo_data_env`` — CPU; hosts download / dataset / convert stages.
- ``dojo_train_env`` — GPU (or CPU in ``minimal`` mode); hosts ``train``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import flyte


def gpu_spec(recipe: Any) -> str | None:
    """Return a Flyte ``Resources(gpu=...)`` string like ``"H100:8"``.

    Returns ``None`` when the recipe requests no GPUs (e.g. ``minimal`` runs
    with ``actor_num_gpus_per_node=0``), in which case no GPU is requested.
    """
    gpu_type = getattr(recipe, "gpu_type", None)
    gpus_per_node = getattr(recipe, "actor_num_gpus_per_node", None)
    if not gpu_type or not gpus_per_node:
        return None
    return f"{gpu_type}:{int(gpus_per_node)}"


def resource_kwargs(
    recipe: Any | None = None, *, minimal_cpu: bool = False
) -> dict[str, Any]:
    """Map recipe CPU/memory/GPU fields to ``flyte.Resources`` kwargs.

    Slime/miles recipes carry ``memory`` (MiB) and ``cpu`` (cores). In
    ``minimal`` mode we run on the CPU env and request modest, non-GPU
    resources regardless of the recipe's requested topology.
    """
    kwargs: dict[str, Any] = {"cpu": "1", "memory": "2Gi"}
    if recipe is not None and not minimal_cpu:
        if (gpu := gpu_spec(recipe)) is not None:
            kwargs["gpu"] = gpu
        cpu = getattr(recipe, "cpu", None)
        if cpu:
            kwargs["cpu"] = str(cpu)
        memory_mib = getattr(recipe, "memory", None)
        if memory_mib:
            kwargs["memory"] = f"{int(memory_mib) / 1024:.0f}Gi"
    return kwargs


# ── Images ───────────────────────────────────────────────────────────────────


def minimal_train_image() -> "flyte.Image":
    """Small CPU/1-GPU image for the self-contained ``minimal`` trainer."""
    import flyte

    return flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "protobuf",
    )


def framework_image(recipe: Any) -> "flyte.Image":
    """Image for the real slime/miles framework launcher.

    The full framework image (megatron, the slime checkout, patched training
    scripts) is defined by the framework launchers. The Flyte backend builds a
    base CUDA image with the framework's Python requirements and expects a
    registry-pushed image for production; documented in ``flyte/README.md``.
    """
    import flyte

    return flyte.Image.from_debian_base().with_pip_packages(
        "torch",
        "transformers",
        "datasets",
        "megatron-core",
        "accelerate",
    )


# ── Environments ─────────────────────────────────────────────────────────────


def data_environment() -> "flyte.TaskEnvironment":
    """CPU ``TaskEnvironment`` for the download / dataset / convert stages."""
    import flyte

    return flyte.TaskEnvironment(
        name="dojo_data_env",
        image=minimal_train_image(),
        resources=flyte.Resources(cpu="2", memory="4Gi"),
    )


def train_environment(
    recipe: Any | None = None, *, minimal_cpu: bool = False
) -> "flyte.TaskEnvironment":
    """GPU (or CPU for ``minimal``) ``TaskEnvironment`` hosting ``train``."""
    import flyte

    image = (
        minimal_train_image()
        if minimal_cpu or recipe is None
        else framework_image(recipe)
    )
    return flyte.TaskEnvironment(
        name="dojo_train_env",
        image=image,
        resources=flyte.Resources(**resource_kwargs(recipe, minimal_cpu=minimal_cpu)),
    )
