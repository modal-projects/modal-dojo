from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import datetime
from typing import TYPE_CHECKING, Any, Callable, Literal

import modal
from pydantic import BaseModel, Field, model_validator

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.dataset import DatasetRow
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.ids import create_hash
from modal_dojo.utils.metadata import (
    MetadataStore,
    summary_items_from_payload,
    vol_get,
    vol_get_summary_items,
    vol_put,
)

from modal_dojo.common.sample import Sample
from modal_dojo.common.sandbox import Sandbox

if TYPE_CHECKING:
    from modal_dojo.common.dataset import DatasetConfig
    from modal_dojo.common.deployment import CustomDeployment
    from modal_dojo.common.models.base import ModelConfig

EVAL_SUMMARY_STORE = MetadataStore.EVALS
EVAL_SUMMARY_KEY = "summary"
EVAL_SUMMARY_PAYLOAD_KEY = "summaries"

#: How often (in completed rows) a running eval flushes partial results to the
#: metadata volume. Smaller = fresher dashboard, more volume writes.
_INTERMEDIATE_SAVE_EVERY = 5


def _callable_name(fn: Callable[..., Any]) -> str:
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    if name:
        return name
    return type(fn).__name__


@exclude_from_api_reference
class EvalConfigDurable(BaseModel):
    """JSON-serializable audit record for an :class:`EvalConfig`."""

    eval_config_id: str
    dataset_name: str
    eval_fn_name: str
    prompt_column: str | None = None
    generate_kwargs: dict[str, Any] = Field(default_factory=dict)

    def save(self) -> None:
        vol_put(
            MetadataStore.EVAL_CONFIGS,
            self.eval_config_id,
            self.model_dump(mode="json"),
        )

    @classmethod
    def from_id(cls, eval_config_id: str) -> "EvalConfigDurable":
        return cls.model_validate(vol_get(MetadataStore.EVAL_CONFIGS, eval_config_id))


# An eval row is just a Sample. Kept as an alias for the public API / existing
# imports; new code should use Sample directly.
exclude_from_api_reference("EvalRowResult")
EvalRowResult = Sample


@exclude_from_api_reference
class AudioEvalRowResult(Sample):
    """``Sample`` for an audio eval, with the audio fields lifted to
    constructor arguments.

    ``audio`` (a browser-playable data-URI), ``reference`` (the ground truth), and
    ``metrics`` (a ``{name: value}`` dict — the eval picks its own metrics, e.g.
    ``{"wer": 0.1}`` or ``{"mos": 4.2}``) are folded into ``metadata`` under
    ``_metadata_type="audio"`` so the evals dashboard auto-detects and renders an
    audio cell. The model output stays on ``response`` (there is no separate
    ``hypothesis``); ``score`` remains the canonical headline number. Extra
    ``metadata`` is kept.

    Usage:
        AudioEvalRowResult(
            score=1.0 - wer, response=hypothesis, prompt=prompt,
            audio=audio_uri, reference=reference, metrics={"wer": wer},
        )
    """

    @model_validator(mode="before")
    @classmethod
    def _fold_audio_into_metadata(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        metadata = dict(data.pop("metadata", None) or {})
        metadata["_metadata_type"] = "audio"
        for key in ("audio", "reference", "metrics"):
            value = data.pop(key, None)
            if value is not None:
                metadata[key] = value
        data["metadata"] = metadata
        return data


@exclude_from_api_reference
class ImageEvalRowResult(Sample):
    """``Sample`` for an image eval, with the image fields lifted to
    constructor arguments.

    ``image`` (a browser-renderable data-URI / URL — e.g. the screenshot the
    model was shown), ``reference`` (the ground truth), and ``metrics`` (a
    ``{name: value}`` dict the eval picks itself) are folded into ``metadata``
    under ``_metadata_type="image"`` so the evals dashboard auto-detects and
    renders an image cell. The model output stays on ``response``; ``score``
    remains the canonical headline number. Extra ``metadata`` is kept.

    Usage:
        ImageEvalRowResult(
            score=1.0 if hit else 0.0, response=prediction, prompt=prompt,
            image=screenshot_uri, reference=label, metrics={"dist": dist},
        )
    """

    @model_validator(mode="before")
    @classmethod
    def _fold_image_into_metadata(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        metadata = dict(data.pop("metadata", None) or {})
        metadata["_metadata_type"] = "image"
        for key in ("image", "reference", "metrics"):
            value = data.pop(key, None)
            if value is not None:
                metadata[key] = value
        data["metadata"] = metadata
        return data


#: Lifecycle of an eval run, surfaced live on the dashboard:
#: ``deploying_model`` while the deployment cold-starts, ``running_eval`` once it
#: is serving and rows are streaming in, ``completed`` on success, and ``failed``
#: when a run raises partway through. ``running`` is the legacy in-progress value
#: kept so older records still validate. The default is ``completed`` so records
#: written before this field existed validate as finished runs.
EvalStatus = Literal[
    "deploying_model", "running_eval", "running", "completed", "failed"
]


class EvalSummary(BaseModel):
    eval_id: str
    eval_config_id: str
    created_at: datetime.datetime
    total: int
    mean: float
    status: EvalStatus = "completed"
    model_name: str = ""

    @classmethod
    def list_summaries(cls) -> list["EvalSummary"]:
        summaries: dict[str, EvalSummary] = {}
        try:
            payload = vol_get(EVAL_SUMMARY_STORE, EVAL_SUMMARY_KEY)
        except KeyError:
            payload = None
        for item in summary_items_from_payload(payload, EVAL_SUMMARY_PAYLOAD_KEY):
            if isinstance(item, dict) and item.get("eval_id"):
                try:
                    summaries[item["eval_id"]] = cls.model_validate(item)
                except Exception:
                    continue
        for item in vol_get_summary_items(MetadataStore.EVAL_SUMMARIES) or []:
            if isinstance(item, dict) and item.get("eval_id"):
                try:
                    summaries[item["eval_id"]] = cls.model_validate(item)
                except Exception:
                    continue
        return sorted(
            summaries.values(), key=lambda summary: summary.created_at, reverse=True
        )


@exclude_from_api_reference
class EvalResult(BaseModel):
    """Saved results for one evaluation run across a dataset."""

    eval_id: str
    eval_config_id: str
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )
    status: EvalStatus = "completed"
    model_name: str = ""
    rows: list[EvalRowResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def mean(self) -> float:
        return sum(r.score for r in self.rows) / self.total if self.total else 0.0

    def to_summary(self) -> EvalSummary:
        return EvalSummary(
            eval_id=self.eval_id,
            eval_config_id=self.eval_config_id,
            created_at=self.created_at,
            total=self.total,
            mean=self.mean,
            status=self.status,
            model_name=self.model_name,
        )

    def save(self) -> None:
        vol_put(MetadataStore.EVAL_RESULTS, self.eval_id, self.model_dump(mode="json"))
        vol_put(
            MetadataStore.EVAL_SUMMARIES,
            self.eval_id,
            self.to_summary().model_dump(mode="json"),
        )


def eval_summary_item(payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce a canonical eval result payload to its summary row."""
    return EvalResult.model_validate(payload).to_summary().model_dump(mode="json")

    @classmethod
    def from_id(cls, eval_id: str) -> "EvalResult":
        return cls.model_validate(vol_get(MetadataStore.EVAL_RESULTS, eval_id))


Response = str
EvalResponseFn = Callable[[DatasetRow, Response], EvalRowResult]  # TOOD: bad name
EvalFn = Callable[["CustomDeployment", DatasetRow], EvalRowResult]


@exclude_from_api_reference
@dataclass
class EvalConfig:
    """Evaluate a deployed model on a dataset config.

    The dataset must expose ``rows()`` and return iterable dict examples.
    """

    dataset: "DatasetConfig"
    eval_fn: EvalFn | None = None
    eval_response_fn: EvalResponseFn | None = None
    prompt_column: str | None = None
    eval_config_id: str | None = None
    generate_kwargs: dict[str, Any] = field(default_factory=dict)

    def _build_eval_fn(self, eval_response_fn: EvalResponseFn) -> EvalFn:
        def eval_fn(
            deployment: CustomDeployment,
            example: DatasetRow,
        ) -> EvalRowResult:
            prompt = self.build_prompt(example)
            text = deployment.generate(
                prompt,
                **self.generate_kwargs,
            )
            result = eval_response_fn(example, text)
            return EvalRowResult(
                score=result.score,
                response=text,
                prompt=prompt,
                metadata=result.metadata,
            )

        return eval_fn

    def __post_init__(self):
        if self.eval_config_id is None:
            class_name = type(self).__name__
            dataset_name = type(self.dataset).__name__
            eval_fn_name = _callable_name(self.eval_fn or self.eval_response_fn)
            self.eval_config_id = create_hash(
                "eval-config",
                class_name,
                dataset_name,
                eval_fn_name,
                self.prompt_column or "",
            )
        if self.eval_fn is None:
            assert self.eval_response_fn is not None, (
                "eval_fn or eval_response_fn must be set"
            )
            self.eval_fn = self._build_eval_fn(self.eval_response_fn)

    def to_durable(self) -> EvalConfigDurable:
        eval_callable = (
            self.eval_response_fn if self.eval_response_fn is not None else self.eval_fn
        )
        return EvalConfigDurable(
            eval_config_id=self.eval_config_id,
            dataset_name=type(self.dataset).__name__,
            eval_fn_name=_callable_name(eval_callable),
            prompt_column=self.prompt_column,
            generate_kwargs=self.generate_kwargs,
        )

    def save(self) -> EvalConfigDurable:
        durable = self.to_durable()
        durable.save()
        return durable

    def build_prompt(self, row: DatasetRow) -> str:
        prompt_column = (self.prompt_column or "").strip()
        input_column = getattr(self.dataset, "input_column", "")
        dataset_column = input_column if isinstance(input_column, str) else ""

        preferred_columns: list[str] = []
        if prompt_column:
            preferred_columns.append(prompt_column)
        if dataset_column and dataset_column not in preferred_columns:
            preferred_columns.append(dataset_column)
        for fallback in ("prompt", "input", "instruction", "question"):
            if fallback not in preferred_columns:
                preferred_columns.append(fallback)

        for column in preferred_columns:
            if column not in row:
                continue
            raw = str(row[column])
            if column in {prompt_column, dataset_column}:
                template = (
                    getattr(self.dataset, "prompt_template", "{input}") or "{input}"
                )
                row_context = {
                    key: str(value)
                    for key, value in row.items()
                    if isinstance(key, str)
                }
                try:
                    return template.format(input=raw, **row_context)
                except (KeyError, ValueError):
                    return raw
            return raw

        raise DojoConfigError(
            "EvalConfig.build_prompt() could not resolve a prompt column. "
            "Set EvalConfig.prompt_column or dataset.input_column, or include one of "
            "['prompt', 'input', 'instruction', 'question'] in dataset rows."
        )

    def evaluate(
        self,
        deployment: "CustomDeployment",
        debug: bool = False,
        max_concurrency: int = 1,
        ready_timeout: int = 15 * 60,
    ) -> EvalResult:
        from modal_dojo.cli.setup import ensure_dashboard_deployed

        if max_concurrency < 1:
            raise DojoConfigError("max_concurrency must be >= 1")

        ensure_dashboard_deployed()

        self.save()

        # Persist the record before waiting on the deployment so the dashboard
        # surfaces the run while the model cold-starts (``deploying_model``),
        # flips to ``running_eval`` once it is serving and streams rows in as
        # they complete, then lands on ``completed``/``failed``.
        eval_id = create_hash(
            "eval",
            self.eval_config_id,
            deployment.deployment_id,
            type(self.dataset).__name__,
            _callable_name(self.eval_fn or self.eval_response_fn),
        )
        result = EvalResult(
            eval_id=eval_id,
            eval_config_id=self.eval_config_id,
            created_at=datetime.datetime.now(datetime.UTC),
            status="deploying_model",
            model_name=deployment.model.model_name,
            rows=[],
        )
        result.save()

        try:
            deployment.wait_until_ready(timeout=ready_timeout)
        except Exception:
            result.status = "failed"
            result.save()
            raise

        result.status = "running_eval"
        result.save()

        def _evaluate_indexed(
            item: tuple[int, DatasetRow],
        ) -> tuple[int, EvalRowResult]:
            idx, example = item
            return idx, self.eval_fn(deployment, example)

        # Consume results as they complete (not in submission order) so one slow
        # row can't stall progress, partial saves, or the dashboard behind it.
        rows_by_idx: dict[int, EvalRowResult] = {}
        try:
            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                futures = [
                    executor.submit(_evaluate_indexed, item)
                    for item in enumerate(self.dataset.rows(), start=1)
                ]
                for future in as_completed(futures):
                    idx, row_result = future.result()
                    if debug:
                        print(
                            f"Finished example {idx}: "
                            f"response={row_result.response!r} "
                            f"score={row_result.score}",
                            flush=True,
                        )
                    rows_by_idx[idx] = row_result
                    result.rows.append(row_result)
                    # Flush partial progress periodically rather than per row:
                    # each save rewrites the shared summary list, so throttle to
                    # keep volume writes bounded on large datasets.
                    if result.total % _INTERMEDIATE_SAVE_EVERY == 0:
                        result.save()
        except Exception:
            result.status = "failed"
            result.save()
            raise

        # Restore dataset order for the persisted result (intermediate saves
        # land in completion order, which is fine for the live view).
        result.rows = [rows_by_idx[idx] for idx in sorted(rows_by_idx)]
        result.status = "completed"
        result.save()
        return result


# ── Harbor evaluation helpers ────────────────────────────────────────────

_CODE_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str, model: "ModelConfig | None" = None) -> str:
    """Extract Python code from an LLM response."""
    if model is not None:
        parsed = model.parse_response(text)
        for tool_call in parsed.tool_calls:
            code = tool_call.arguments.get("code", "")
            if code:
                return code
        content = parsed.content
    else:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if "<|im_start|>assistant" in normalized:
            normalized = normalized.rsplit("<|im_start|>assistant", 1)[-1]
        if "</think>" in normalized:
            normalized = normalized.split("</think>", 1)[-1]
        normalized = normalized.replace("<think>", "").replace("<|im_end|>", "").strip()
        content = normalized

    if match := _CODE_FENCE_RE.search(content):
        return match.group(1).strip()
    return content


# Modal's per-container default resource request (modal.com/docs/guide/resources).
# Used as the request "floor" for the "limit" enforcement policy so sandboxes bill by
# actual CPU-/RAM-second usage rather than a static reservation.
_MODAL_DEFAULT_CPU_REQUEST = 0.125
_MODAL_DEFAULT_MEMORY_REQUEST = 128  # MiB

#: Accepted CPU/memory enforcement policies, mirroring Harbor v0.8.0's ``--cpus`` /
#: ``--memory`` flags. Modal bills for ``max(request, actual usage)``, so reserving more
#: than a sandbox uses over-provisions and inflates cost.
RESOURCE_POLICIES = ("reserve", "limit", "ignore")


def _sandbox_resource(
    value: float, policy: str, default_request: float
) -> float | tuple[float, float] | None:
    """Translate a Harbor-style enforcement *policy* into a Modal cpu/memory kwarg.

    - ``"reserve"`` — reserve *value* outright (billed for the full reservation, even
      when idle). This is the static-reservation behavior that over-provisions on Modal.
    - ``"limit"`` — request the small Modal default and cap bursting at *value*, so the
      sandbox is billed by actual usage up to that ceiling.
    - ``"ignore"`` — no enforcement; returns ``None`` so the caller omits the kwarg and
      the sandbox bursts freely on Modal's default request, billed by actual usage.
    """
    if policy == "reserve":
        return value
    if policy == "limit":
        return (min(default_request, value), value)
    if policy == "ignore":
        return None
    raise DojoConfigError(
        f"invalid resource policy {policy!r}; expected one of {RESOURCE_POLICIES}"
    )


def score_in_sandbox(
    code: str,
    *,
    test_cases: list[dict[str, str]],
    timeout_sec: int = 60,
    sandbox_cpu: float = 1.0,
    sandbox_memory: int = 1024,
    python_version: str = "3.11",
    cpu_policy: str = "limit",
    memory_policy: str = "limit",
) -> tuple[float, dict[str, Any]]:
    """Run code against test cases in a Modal sandbox."""
    if not test_cases:
        return 0.0, {"error": "no test cases"}

    runner = (
        "import sys, json, io, contextlib\n"
        "cases = json.loads(sys.argv[1])\n"
        "results = []\n"
        "for case in cases:\n"
        "    old_stdin = sys.stdin\n"
        '    sys.stdin = io.StringIO(case["input"])\n'
        "    buf = io.StringIO()\n"
        "    ok = False\n"
        "    try:\n"
        "        with contextlib.redirect_stdout(buf):\n"
        '            exec(compile(case["code"], "<solution>", "exec"))\n'
        '        ok = buf.getvalue().strip() == case["expected_output"].strip()\n'
        "    except Exception as exc:\n"
        '        buf.write(f"ERROR: {exc}")\n'
        "    finally:\n"
        "        sys.stdin = old_stdin\n"
        '    results.append({"passed": ok, "stdout": buf.getvalue()})\n'
        "print(json.dumps(results))\n"
    )

    cases_payload = json.dumps(
        [
            {
                "code": code,
                "input": tc.get("input", ""),
                "expected_output": tc.get("expected_output", ""),
            }
            for tc in test_cases
        ]
    )

    resource_kwargs: dict[str, Any] = {}
    cpu_arg = _sandbox_resource(sandbox_cpu, cpu_policy, _MODAL_DEFAULT_CPU_REQUEST)
    if cpu_arg is not None:
        resource_kwargs["cpu"] = cpu_arg
    memory_arg = _sandbox_resource(
        sandbox_memory, memory_policy, _MODAL_DEFAULT_MEMORY_REQUEST
    )
    if memory_arg is not None:
        resource_kwargs["memory"] = memory_arg

    with Sandbox(
        image=modal.Image.debian_slim(python_version=python_version),
        timeout=timeout_sec,
        app_name="training-gym-sandbox-rm",
        **resource_kwargs,
    ) as sb:
        result = sb.run("python", "-c", runner, cases_payload, timeout=timeout_sec)

    metadata: dict[str, Any] = {"stderr": result.stderr}
    try:
        results = json.loads(result.stdout)
        passed = sum(1 for r in results if r.get("passed"))
        metadata["per_case"] = results
        return passed / len(test_cases), metadata
    except (json.JSONDecodeError, TypeError):
        metadata["raw_stdout"] = result.stdout
        return 0.0, metadata


@exclude_from_api_reference
@dataclass
class HarborEval(EvalConfig):
    """Evaluate a deployed model on a Harbor dataset using sandbox execution.

    Automates the common pattern of generating code from a Harbor task,
    extracting it from the LLM response, running it in a Modal sandbox,
    and comparing stdout against expected test-case outputs.

    When neither ``eval_fn`` nor ``eval_response_fn`` is provided, a
    default sandbox-backed scorer is used automatically.  Pass
    ``extract_code_fn`` to override how code is pulled from the model
    response, or supply your own ``eval_fn`` to take full control.
    """

    model: "ModelConfig | None" = None
    test_cases: list[dict[str, str]] | None = None
    sandbox_timeout: int = 60
    sandbox_cpu: float = 1.0
    sandbox_memory: int = 1024
    sandbox_cpu_policy: str = "limit"
    sandbox_memory_policy: str = "limit"
    sandbox_python_version: str = "3.11"
    extract_code_fn: Callable[[str], str] | None = None

    def _resolve_test_cases(self, example: DatasetRow) -> list[dict[str, str]]:
        label = example.get("label", {})
        if isinstance(label, str):
            try:
                label = json.loads(label)
            except (json.JSONDecodeError, ValueError):
                label = {}
        if isinstance(label, dict):
            cases = label.get("test_cases")
            if isinstance(cases, list) and cases:
                return cases
        if self.test_cases is not None:
            return self.test_cases
        return []

    def _extract_code(self, text: str) -> str:
        if self.extract_code_fn is not None:
            return self.extract_code_fn(text)
        return extract_code(text, model=self.model)

    def _build_messages(self, example: DatasetRow, prompt: str) -> list[dict[str, str]]:
        messages = example.get("messages")
        if isinstance(messages, list) and messages:
            return messages
        msgs: list[dict[str, str]] = []
        sys_prompt = getattr(self.dataset, "system_prompt", "")
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _harbor_eval_fn(
        self,
        deployment: "CustomDeployment",
        example: DatasetRow,
    ) -> EvalRowResult:
        prompt = self.build_prompt(example)
        messages = self._build_messages(example, prompt)
        response = deployment.generate(
            prompt,
            messages=messages,
            **self.generate_kwargs,
        )
        code = self._extract_code(response)
        test_cases = self._resolve_test_cases(example)
        score, metadata = score_in_sandbox(
            code,
            test_cases=test_cases,
            timeout_sec=self.sandbox_timeout,
            sandbox_cpu=self.sandbox_cpu,
            sandbox_memory=self.sandbox_memory,
            cpu_policy=self.sandbox_cpu_policy,
            memory_policy=self.sandbox_memory_policy,
            python_version=self.sandbox_python_version,
        )

        parsed = self.model.parse_response(response) if self.model is not None else None

        return EvalRowResult(
            score=score,
            response=response,
            prompt=prompt,
            parsed_response=parsed,
            metadata=metadata,
        )

    def __post_init__(self) -> None:
        if self.eval_fn is None and self.eval_response_fn is None:
            self.eval_fn = self._harbor_eval_fn
        super().__post_init__()
