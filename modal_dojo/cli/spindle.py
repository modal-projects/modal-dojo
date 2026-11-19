"""Set up a persistent Spindle server or save an external Tinker connection."""

from __future__ import annotations

import os
import sys

import click


@click.group(name="spindle")
def spindle_group():
    """Configure a reusable Tinker connection in the current Modal environment."""


@spindle_group.command(name="setup")
@click.option("--base-url", help="Save this server as the environment's default.")
@click.option("--model", help="Base model for the initial Spindle backend.")
@click.option(
    "--context-length",
    type=click.IntRange(min=1),
    help="Minimum total context length (defaults to the model's smallest preset).",
)
@click.option("--trainer-gpu")
@click.option("--trainer-gpus-per-node", type=click.IntRange(min=1))
@click.option("--inference-gpu")
@click.option(
    "--env",
    "environment",
    help="Modal environment (defaults to the active environment).",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Approve creating a persistent Spindle server or replacing the saved default.",
)
def setup_command(
    base_url,
    model,
    context_length,
    trainer_gpu,
    trainer_gpus_per_node,
    inference_gpu,
    environment,
    yes,
):
    """Reuse or deploy Spindle; --base-url saves a connection using TINKER_API_KEY."""
    from modal.config import config

    from modal_dojo.frameworks.spindle.connection import (
        DEFAULT_SECRET,
        resolve_connection,
        save_default,
    )
    from modal_dojo.frameworks.spindle.deployment import Requirements

    environment = environment or config.get("environment") or "main"
    if base_url:
        if any(
            (model, context_length, trainer_gpu, trainer_gpus_per_node, inference_gpu)
        ):
            raise click.UsageError(
                "Use --base-url to save a connection, or --model to set up Spindle."
            )
        key = os.environ.get("TINKER_API_KEY")
        if not key:
            if not sys.stdin.isatty():
                raise click.UsageError("Set TINKER_API_KEY to save this connection.")
            key = click.prompt("Tinker API key", hide_input=True)
        if not yes:
            if not sys.stdin.isatty():
                raise click.UsageError(
                    "Pass --yes to save this environment's shared default."
                )
            click.confirm(
                f"Save {base_url} as the shared Tinker default in {environment!r}?",
                abort=True,
            )
        save_default(base_url, key, environment)
        click.echo(f"Saved {DEFAULT_SECRET} in {environment!r}.")
        return
    if not model:
        raise click.UsageError(
            "Provide --model to set up Spindle, or --base-url to save a connection."
        )
    overrides = {
        k: v
        for k, v in {
            "trainer_gpu": trainer_gpu,
            "trainer_gpus_per_node": trainer_gpus_per_node,
            "inference_gpu": inference_gpu,
        }.items()
        if v is not None
    }
    connection = resolve_connection(
        None,
        None,
        Requirements(model, 0, 32, context_length, overrides),
        yes=yes,
        environment=environment,
    )
    click.echo(
        f"Using {connection.base_url or DEFAULT_SECRET} in {environment!r}. Model backends are checked when training starts."
    )
