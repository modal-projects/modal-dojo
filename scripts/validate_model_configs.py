"""Validate a model config by running base training on its framework.

The model registry (``common/models/validation.py``) says which framework
trains each model; ``build_recipe_and_dataset`` in
``scripts/validation_backends/`` supplies that framework's recipe and dataset.
Everything below is framework-agnostic.

Usage:
    uv run scripts/validate_model_configs.py list
    uv run scripts/validate_model_configs.py list --names-only
    uv run scripts/validate_model_configs.py list --framework miles
    uv run scripts/validate_model_configs.py check -m qwen3-4b
    uv run scripts/validate_model_configs.py check -m Qwen3.5-4B-Miles
    uv run scripts/validate_model_configs.py summarize -d results
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cloudpickle
from modal.exception import Error
from modal._vendor import cloudpickle as modal_cloudpickle

try:
    # Run as a script: sys.path[0] is scripts/, matching download_perf_baseline.
    from validation_backends import build_recipe_and_dataset
except ImportError:  # imported as scripts.validate_model_configs, e.g. by tests
    from scripts.validation_backends import build_recipe_and_dataset

from modal_training_gym.common.models.validation import (
    Framework,
    _ValidationConfig,
)
from modal_training_gym.common.modal_lifecycle import stop_app
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.common.step_timing import measured_run_times
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.train import TrainConfig

COMMENT_MARKER = "<!-- validate-models-comment -->"
TIMING_SETTLE_WINDOW_S = 30.0


def _fmt_secs(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    n = float(seconds)
    if n >= 60:
        minutes = int(n // 60)
        rem = n - minutes * 60
        return f"{minutes}m {rem:.3f}s"
    return f"{n:.3f}s"


def _substep_label(name: str) -> str:
    _SUBSTEP_LABELS = {
        "initial_weight_sync": "Initial weight sync",
        "evaluate_rollouts": "Eval (before)",
        "generate_rollouts": "Generate rollouts",
        "offload_rollout": "Offload rollout",
        "compute_log_probs": "Compute log probs",
        "optimizer_step": "Optimizer step",
        "checkpoint_save": "Checkpoint save",
        "offload_train": "Offload train",
        "weight_sync": "Weight sync",
        "evaluate_rollouts_end": "Eval (after)",
        "wait_for_rollout": "Wait for rollout",
        "wait_for_next_rollout": "Wait for next rollout",
        "train_models": "Train models",
        "generate_samples": "Generate samples",
        "sample_generation": "Sample generation",
        "forward_backward": "Forward/backward",
        "reward_batch": "Reward batch",
        "reward": "Reward",
        "reward_post_process": "Reward post-process",
        "trainer_finalize": "Cleanup & offload",
        "train_step_finalize": "Train step finalize",
    }

    phase_name, separator, role = name.rpartition(" (")
    if separator and role.endswith(")"):
        label = _SUBSTEP_LABELS.get(phase_name, phase_name.replace("_", " "))
        return f"{label} ({role[:-1]})"
    return _SUBSTEP_LABELS.get(name, name.replace("_", " "))


def _total_step_time_s(result: "ValidationResult") -> float:
    """Sum of per-step durations.

    Reported instead of wall clock, which also covers queue, model download and
    checkpoint conversion time — variable with compute availability rather than
    gym performance.
    """
    return float(
        sum(step.get("duration_s") or 0 for step in (result.step_times or {}).values())
    )


def _comparable_step_time_s(
    result: "ValidationResult", comparable_steps: set[str] | None = None
) -> float:
    return float(
        sum(
            step.get("duration_s") or 0
            for key, step in (result.step_times or {}).items()
            if not step.get("partial")
            and (comparable_steps is None or key in comparable_steps)
        )
    )


def _comparable_step_keys(
    result: "ValidationResult", baseline: "ValidationResult"
) -> set[str]:
    return {
        key
        for key in set(result.step_times or {}) & set(baseline.step_times or {})
        if not (result.step_times or {})[key].get("partial")
        and not (baseline.step_times or {})[key].get("partial")
    }


def _step_keys(result: "ValidationResult") -> list[str]:
    keys = set(result.step_times or {}) | set(result.substep_times or {})
    return sorted(keys, key=lambda k: int(k) if k.isdigit() else k)


def _expected_timing_steps(
    step_count: int | None, resume_from_iteration: int | None
) -> set[int]:
    if step_count is None:
        return set()
    range_start = resume_from_iteration + 2 if resume_from_iteration is not None else 1
    return set(range(range_start, step_count + 1))


def _covered_timing_steps(
    step_times: dict | None, substep_times: dict | None
) -> set[int]:
    return {
        int(step)
        for step in (set(step_times or {}) | set(substep_times or {}))
        if str(step).isdigit()
    }


def _warn_if_timings_missing(
    training_run_id: str,
    status: TrainingRunStatus,
    step_times: dict | None,
    substep_times: dict | None,
    *,
    timing_read_failed: bool = False,
    expected_step_count: int | None = None,
    resume_from_iteration: int | None = None,
) -> None:
    if status == TrainingRunStatus.COMPLETED and not step_times and not substep_times:
        if timing_read_failed:
            print(
                f"warning: timing records could not be read for completed run "
                f"{training_run_id}; the metadata volume read failed"
            )
            return
        print(
            f"warning: no timing records found for completed run "
            f"{training_run_id}; the dashboard may have been unreachable or "
            "substep timing may be disabled"
        )
        return
    if status != TrainingRunStatus.COMPLETED or expected_step_count is None:
        return
    covered_steps = _covered_timing_steps(step_times, substep_times)
    expected_steps = _expected_timing_steps(expected_step_count, resume_from_iteration)
    if not expected_steps:
        return
    range_start = min(expected_steps)
    range_end = max(expected_steps)
    covered_steps &= expected_steps
    missing_steps = expected_steps - covered_steps
    if missing_steps:
        print(
            f"warning: incomplete timing records for completed run "
            f"{training_run_id}; expected steps {range_start}-{range_end}, "
            f"found {len(covered_steps)}"
        )
        if len(missing_steps) != len(expected_steps):
            print(
                f"warning: timing records have holes in executed range "
                f"{range_start}-{range_end}: "
                f"{', '.join(map(str, sorted(missing_steps)))}"
            )


def _ordered_substeps(
    subs: dict[str, dict[str, float | int | bool | None]],
) -> list[tuple[str, dict[str, float | int | bool | None]]]:
    return sorted(
        subs.items(),
        key=lambda item: (
            item[1].get("start") is None,
            item[1].get("start") or 0,
        ),
    )


def _format_substep_timing(entry: dict) -> str:
    busy = _fmt_secs(entry.get("duration_s"))
    if not entry.get("concurrent"):
        return busy
    count = entry.get("invocation_count", 0)
    wall = _fmt_secs(entry.get("wall_duration_s"))
    return f"{count} samples · {busy} busy across {wall} wall clock"


@dataclass
class ValidationResult:
    base_model_name: str
    step_count: int
    training_run_id: str
    training_run_status: TrainingRunStatus
    total_duration_s: float
    step_times: dict[str, dict[str, float | bool | None]] | None = None
    substep_times: dict[str, dict[str, dict[str, float | int | bool | None]]] | None = (
        None
    )
    framework: str = Framework.SLIME.value
    recipe_name: str | None = None
    docker_image: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.training_run_status == TrainingRunStatus.COMPLETED

    def print_summary(self) -> None:
        print(f"Training run result for {self.training_run_id}")
        print("Parameters:")
        print(f"Base model name: {self.base_model_name}")
        print(f"Framework: {self.framework}")
        if self.recipe_name:
            print(f"Recipe: {self.recipe_name}")
        if self.docker_image:
            print(f"Image: {self.docker_image}")
        print(f"Step count: {self.step_count}")
        print("Result:")
        print(f"Training run status: {self.training_run_status}")
        print(f"Total step time (s): {_total_step_time_s(self)}")
        print(f"Total duration (s): {self.total_duration_s}")

        keys = _step_keys(self)
        if not keys:
            return

        print("Timings:")
        for key in keys:
            step = (self.step_times or {}).get(key, {})
            duration = step.get("duration_s")
            partial = (
                " — partial; attempt ended mid-step" if step.get("partial") else ""
            )
            print(f"Step {key} ({_fmt_secs(duration)}{partial})")

            for name, entry in _ordered_substeps(
                (self.substep_times or {}).get(key, {})
            ):
                print(f"    {_substep_label(name)}: {_format_substep_timing(entry)}")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["training_run_status"] = self.training_run_status.value
        data["succeeded"] = self.succeeded
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ValidationResult":
        """Rebuild a result, tolerating JSON written before a field existed.

        Baselines are downloaded from artifacts on already-merged PRs, so this
        reads results produced by older revisions of this script.
        """
        return cls(
            base_model_name=data["base_model_name"],
            step_count=data["step_count"],
            training_run_id=data["training_run_id"],
            training_run_status=TrainingRunStatus(data["training_run_status"]),
            total_duration_s=data["total_duration_s"],
            step_times=data.get("step_times"),
            substep_times=data.get("substep_times"),
            framework=data.get("framework", Framework.SLIME.value),
            recipe_name=data.get("recipe_name"),
            docker_image=data.get("docker_image"),
        )


def available_model_names(framework: Framework | None = None) -> list[str]:
    """Sorted model names the harness can run."""
    return [config.name for config in _ValidationConfig.select(framework)]


def available_models(
    framework: Framework | None = None,
) -> list[dict[str, str]]:
    """Registry details for humans inspecting supported validation models."""
    return [
        {
            "name": config.name,
            "model_name": config.model_name,
            "framework": config.framework.value,
        }
        for config in _ValidationConfig.select(framework)
    ]


def _ship_dataset_definition(dataset) -> None:
    """Send the dataset's defining module by value, not by reference.

    ``resolve_caller_context`` only registers the module that calls ``train()``,
    which leaves a dataset class defined in a backend module pickled by name.
    Nothing under ``scripts/`` is importable inside the training image, so the
    container would fail to unpickle it during data preparation. Classes that
    ship with the package are importable remotely and stay by reference.

    Modal serializes with its own vendored copy of cloudpickle, which keeps a
    registry separate from the installed one, so both have to be told.
    """
    module = sys.modules.get(type(dataset).__module__)
    if module is None or module.__name__.startswith("modal_training_gym"):
        return
    cloudpickle.register_pickle_by_value(module)
    modal_cloudpickle.register_pickle_by_value(module)


def run_base_training(
    model_name: str,
    step_count: int = 1,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_secret_name: str = "wandb-secret",
    eval_interval: int | None = None,
    save_interval: int | None = None,
    non_colocated: bool = False,
    timeout: float | None = None,
) -> ValidationResult:
    config = _ValidationConfig.find(model_name)
    model_config = config.model_config()

    train_recipe, dataset = build_recipe_and_dataset(
        config.framework,
        model_config,
        step_count,
        loss_type=config.loss_type,
    )
    train_recipe.num_rollout = step_count
    if eval_interval is not None and config.loss_type != "sft_loss":
        train_recipe.eval_interval = eval_interval
    if save_interval is not None:
        train_recipe.save_interval = save_interval
    if non_colocated and config.loss_type != "sft_loss":
        train_recipe.colocate = False
        if train_recipe.rollout_num_gpus is None:
            train_recipe.rollout_num_gpus = (
                train_recipe.actor_num_nodes * train_recipe.actor_num_gpus_per_node
            )
    _ship_dataset_definition(dataset)

    dataset_name = getattr(dataset, "hf_repo", type(dataset).__name__).rsplit("/", 1)[
        -1
    ]
    model_short_name = model_config.model_name.rsplit("/", 1)[-1]
    if wandb_project is not None:
        train_recipe.metrics = WandbConfig(
            project=wandb_project
            or f"model-validation-{model_short_name}-{dataset_name}",
            group=wandb_group or f"model-validator-{model_short_name}-{dataset_name}",
            modal_wandb_secret_name=wandb_secret_name,
        )

    train_config = TrainConfig(
        model=model_config,
        dataset=dataset,
        eval_dataset=None if config.loss_type == "sft_loss" else dataset,
        recipe=train_recipe,
    )

    launch = train_config.launch()
    try:
        train_result = launch.result(timeout=timeout)
    except BaseException:
        stop_app(launch.modal_app_id)
        raise
    training_run = TrainingRun.from_id(train_result.training_run_id)
    previous = None
    step_times, substep_times = {}, {}
    timing_read_succeeded = False
    timing_read_failed = False
    timing_wait_deadline = time.monotonic() + 2 * TIMING_SETTLE_WINDOW_S + 1.0
    stable_since: float | None = None

    def _resume_iteration(run: TrainingRun) -> int | None:
        metadata = getattr(run, "metadata", None) or {}
        resume_iteration = metadata.get("resume_from_iteration")
        return int(resume_iteration) if resume_iteration is not None else None

    resume_from_iteration = _resume_iteration(training_run)
    expected_steps = _expected_timing_steps(step_count, resume_from_iteration)
    while True:
        try:
            latest_training_run = TrainingRun.from_id(train_result.training_run_id)
        except Exception:
            latest_training_run = None
        if latest_training_run is not None:
            training_run = latest_training_run
            resume_from_iteration = _resume_iteration(training_run)
            expected_steps = _expected_timing_steps(step_count, resume_from_iteration)

        try:
            current_step_times, current_substep_times = measured_run_times(
                train_result.training_run_id
            )
        except (Error, TypeError, KeyError, ValueError):
            timing_read_failed = True
            previous = None
        else:
            step_times, substep_times = current_step_times, current_substep_times
            timing_read_succeeded = True
            current = (step_times, substep_times)
            if (
                training_run.status == TrainingRunStatus.COMPLETED
                and current != ({}, {})
                and expected_steps
                and expected_steps <= _covered_timing_steps(*current)
            ):
                break
            # Require a non-empty read to remain stable for the full flush
            # interval; an early equal read may predate the dashboard flush.
            now = time.monotonic()
            if current != ({}, {}) and current == previous:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= TIMING_SETTLE_WINDOW_S:
                    break
            else:
                stable_since = None
            previous = current
        if training_run.status in {
            TrainingRunStatus.FAILED,
            TrainingRunStatus.STOPPED,
            TrainingRunStatus.CANCELLED,
        }:
            break
        if time.monotonic() >= timing_wait_deadline:
            break
        time.sleep(min(2.0, max(0.0, timing_wait_deadline - time.monotonic())))

    _warn_if_timings_missing(
        train_result.training_run_id,
        training_run.status,
        step_times,
        substep_times,
        timing_read_failed=timing_read_failed and not timing_read_succeeded,
        expected_step_count=step_count,
        resume_from_iteration=resume_from_iteration,
    )

    return ValidationResult(
        base_model_name=config.name,
        step_count=step_count,
        training_run_id=train_result.training_run_id,
        training_run_status=training_run.status,
        total_duration_s=float(training_run.duration_seconds or 0.0),
        step_times=step_times,
        substep_times=substep_times,
        framework=config.framework.value,
        recipe_name=type(train_recipe).__name__,
        # Only frameworks that pin an image have the field to report.
        docker_image=getattr(train_recipe, "docker_image", None),
    )


def _status_label(result: ValidationResult) -> str:
    if result.succeeded:
        return "✅ completed"
    return f"❌ {result.training_run_status.value}"


def _format_secs_delta(
    current: float | int | None, baseline: float | int | None
) -> str | None:
    """Compact delta vs baseline, or None when either timing is missing."""
    if current is None or baseline is None:
        return None
    current_f = float(current)
    baseline_f = float(baseline)
    delta_s = current_f - baseline_f
    if baseline_f <= 0:
        return f"{delta_s:+.3f}s"
    percent = delta_s / baseline_f * 100
    return f"{delta_s:+.3f}s ({percent:+.0f}%)"


def _training_run_link(training_run_id: str, dashboard_url: str | None) -> str:
    """Training run id in backticks, linked to the dashboard if a base URL is given."""
    if not dashboard_url:
        return f"`{training_run_id}`"
    base = dashboard_url.rstrip("/")
    return f"[`{training_run_id}`]({base}/training/{training_run_id})"


@dataclass
class BaselineMeta:
    commit_sha: str
    commit_url: str

    @classmethod
    def from_dict(cls, data: dict) -> "BaselineMeta | None":
        sha = data.get("commit_sha")
        url = data.get("commit_url")
        if not sha or not url:
            return None
        return cls(commit_sha=str(sha), commit_url=str(url))

    def commit_link(self) -> str:
        short = self.commit_sha[:7]
        return f"[`{short}`]({self.commit_url})"


def _format_duration_delta(
    result: ValidationResult, baseline_path: Path, dashboard_url: str | None
) -> str:
    """Format the duration change vs a baseline result, naming the baseline
    run, e.g. "+500.0s (+33%) from [`run-id`](https://…/training/run-id)".
    """
    if not baseline_path.is_file():
        return "—"
    baseline = ValidationResult.from_dict(json.loads(baseline_path.read_text()))
    comparable_steps = _comparable_step_keys(result, baseline)
    delta = (
        _format_secs_delta(
            _comparable_step_time_s(result, comparable_steps),
            _comparable_step_time_s(baseline, comparable_steps),
        )
        or "—"
    )
    return f"{delta} from {_training_run_link(baseline.training_run_id, dashboard_url)}"


def _load_baseline(baseline_path: Path | None) -> ValidationResult | None:
    if baseline_path is None or not baseline_path.is_file():
        return None
    return ValidationResult.from_dict(json.loads(baseline_path.read_text()))


def _load_baseline_meta(baseline_path: Path | None) -> BaselineMeta | None:
    """Load sidecar meta written by ``download_perf_baseline.py``."""
    if baseline_path is None:
        return None
    meta_path = baseline_path.with_name(baseline_path.stem + ".meta.json")
    if not meta_path.is_file():
        return None
    return BaselineMeta.from_dict(json.loads(meta_path.read_text()))


def _format_result_details(
    result: ValidationResult,
    baseline: ValidationResult | None = None,
    baseline_meta: BaselineMeta | None = None,
    dashboard_url: str | None = None,
) -> list[str]:
    """Markdown <details> block with run status and a consolidated timing table."""
    lines = [
        "<details>",
        f"<summary>{result.base_model_name}</summary>",
        "",
        f"{_training_run_link(result.training_run_id, dashboard_url)} — {_status_label(result)}",
    ]
    recipe_bits = [f"Framework: {result.framework}"]
    if result.recipe_name:
        recipe_bits.append(f"Recipe: `{result.recipe_name}`")
    if result.docker_image:
        recipe_bits.append(f"Image: `{result.docker_image}`")
    lines.append(" · ".join(recipe_bits))
    if baseline is not None:
        baseline_bits = [
            _training_run_link(baseline.training_run_id, dashboard_url),
        ]
        if baseline_meta is not None:
            baseline_bits.append(f"on {baseline_meta.commit_link()}")
        lines.append(f"Baseline: {' '.join(baseline_bits)}")
    lines.append("")

    keys = _step_keys(result)
    if not keys:
        lines.extend(["_No step timing data._", "", "</details>", ""])
        return lines

    if baseline is not None:
        lines.extend(
            [
                "| Phase | Duration | Delta |",
                "| --- | --- | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| Phase | Duration |",
                "| --- | --- |",
            ]
        )

    def _row(
        phase: str,
        duration: float | int | None,
        base: float | int | None,
        *,
        delta_duration: float | int | None = None,
        display_duration: str | None = None,
    ) -> str:
        if baseline is None:
            return f"| {phase} | {display_duration or _fmt_secs(duration)} |"
        delta = (
            _format_secs_delta(
                duration if delta_duration is None else delta_duration, base
            )
            or "—"
        )
        return f"| {phase} | {display_duration or _fmt_secs(duration)} | {delta} |"

    for key in keys:
        step = (result.step_times or {}).get(key) or {}
        baseline_step = ((baseline.step_times or {}).get(key) or {}) if baseline else {}
        comparable = not step.get("partial") and not baseline_step.get("partial")
        baseline_subs = (
            ((baseline.substep_times or {}).get(key) or {}) if baseline else {}
        )
        for name, entry in _ordered_substeps(
            (result.substep_times or {}).get(key) or {}
        ):
            base_entry = baseline_subs.get(name) or {}
            lines.append(
                _row(
                    _substep_label(name),
                    entry.get("duration_s"),
                    base_entry.get("duration_s") if comparable else None,
                    display_duration=_format_substep_timing(entry),
                )
            )
        lines.append(
            _row(
                f"Step {key}"
                + (
                    " (partial — attempt ended mid-step)" if step.get("partial") else ""
                ),
                step.get("duration_s"),
                baseline_step.get("duration_s") if comparable else None,
            )
        )
    if len(keys) > 1:
        comparable_steps = _comparable_step_keys(result, baseline) if baseline else None
        lines.append(
            _row(
                "Total step time",
                _total_step_time_s(result),
                (
                    _comparable_step_time_s(baseline, comparable_steps)
                    if baseline
                    else None
                ),
                delta_duration=_comparable_step_time_s(result, comparable_steps),
            )
        )
    lines.extend(["", "</details>", ""])
    return lines


def summarize_results(
    results_dir: str, baseline_dir: str | None, dashboard_url: str | None = None
) -> str:
    rows = []
    details: list[str] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        result = ValidationResult.from_dict(json.loads(path.read_text()))
        status = _status_label(result)
        row = (
            f"| {result.base_model_name} | {result.framework} | {status} "
            f"| {_total_step_time_s(result):.1f}s | {result.step_count} "
            f"| {_training_run_link(result.training_run_id, dashboard_url)} |"
        )
        baseline_path = (
            Path(baseline_dir) / path.name if baseline_dir is not None else None
        )
        if baseline_dir is not None:
            assert baseline_path is not None
            delta = _format_duration_delta(result, baseline_path, dashboard_url)
            row += f" {delta} |"
        rows.append(row)
        details.extend(
            _format_result_details(
                result,
                _load_baseline(baseline_path),
                _load_baseline_meta(baseline_path),
                dashboard_url,
            )
        )

    header = "| Model | Framework | Status | Step time | Steps | Run |"
    divider = "| --- | --- | --- | --- | --- | --- |"
    empty = "| _no results_ | | | | | |"
    if baseline_dir is not None:
        header += " Delta |"
        divider += " --- |"
        empty += " |"

    lines = [
        COMMENT_MARKER,
        "## Model Validation Results",
        "",
        header,
        divider,
    ]
    lines.extend(rows or [empty])
    if details:
        lines.extend(["", "### Step timings", ""])
        lines.extend(details)
    return "\n".join(lines).rstrip() + "\n"


def __main__():
    parser = argparse.ArgumentParser(
        description="Validate a model config by running base training on its framework."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="Run base training for a single model."
    )
    check_parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Base model name to run training on (e.g. qwen3-4b). One of: "
        f"{', '.join(available_model_names())}.",
    )
    check_parser.add_argument(
        "-n",
        "--num_steps",
        type=int,
        default=1,
        help="Number of training steps (rollouts) to run. Defaults to 1.",
    )
    check_parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Override the recipe eval_interval (eval every N rollouts).",
    )
    check_parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Override the recipe save_interval (checkpoint every N rollouts).",
    )
    check_parser.add_argument(
        "--non-colocated",
        action="store_true",
        help="Allocate rollout GPUs separately from trainer GPUs.",
    )
    check_parser.add_argument(
        "-o",
        "--output",
        help="Write the result as JSON to this file path.",
    )
    check_parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Print the result as JSON to stdout.",
    )
    check_parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project for validator runs. If omitted, W&B logging is disabled.",
    )
    check_parser.add_argument(
        "--wandb-group",
        default="",
        help="W&B group for validator runs. Defaults to model-validator-{model}-{dataset}.",
    )
    check_parser.add_argument(
        "--wandb-secret-name",
        default="wandb-secret",
        help="Modal Secret name containing WANDB_API_KEY.",
    )
    check_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging for this validator run.",
    )

    list_parser = subparsers.add_parser(
        "list", help="Print available models and their frameworks as JSON."
    )
    list_parser.add_argument(
        "--framework",
        choices=[framework.value for framework in Framework],
        default=None,
        help="Only list models validated on this framework.",
    )
    list_parser.add_argument(
        "--names-only",
        action="store_true",
        help="Print only model names as a JSON array, for CI matrix consumers.",
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Render a markdown table from a directory of result JSON files.",
    )
    summarize_parser.add_argument(
        "-d",
        "--results-dir",
        required=True,
        help="Directory containing result JSON files written by `check --output`.",
    )
    summarize_parser.add_argument(
        "-b",
        "--baseline-dir",
        help="Directory containing baseline result JSON files to compare against",
    )
    summarize_parser.add_argument(
        "--dashboard-url",
        help="Base URL of the training dashboard. If omitted, run ids are not linked.",
    )

    args = parser.parse_args()

    if args.command == "list":
        framework = Framework(args.framework) if args.framework else None
        print(
            json.dumps(
                available_model_names(framework)
                if args.names_only
                else available_models(framework)
            )
        )
        return

    if args.command == "summarize":
        print(
            summarize_results(args.results_dir, args.baseline_dir, args.dashboard_url)
        )
        return

    result = run_base_training(
        args.model,
        args.num_steps,
        None if args.no_wandb else args.wandb_project,
        args.wandb_group,
        args.wandb_secret_name,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        non_colocated=args.non_colocated,
    )
    result.print_summary()

    if args.output:
        Path(args.output).write_text(json.dumps(result.to_dict()))
    if args.json:
        print(json.dumps(result.to_dict()))

    if not result.succeeded:
        print("Training run failed")
        exit(1)
    print("Training run completed successfully")
    exit(0)


if __name__ == "__main__":
    __main__()
