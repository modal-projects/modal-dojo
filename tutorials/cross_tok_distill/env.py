import ast
import hashlib
import inspect
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Callable

from transformers import AutoTokenizer

from modal_dojo.common.dataset import DatasetConfig
from modal_dojo.common.models.base import ToolCall


@cache
def _tokenizer(name: str):
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


@dataclass
class Observation:
    text: str = ""
    is_error: bool = False


@dataclass
class EvalVerdict:
    passed: bool
    detail: str = ""


_JSON_TYPE_MAP = {
    "dict": "object",
    "list": "array",
    "tuple": "array",
    "float": "number",
    "integer": "integer",
    "string": "string",
    "boolean": "boolean",
}


def _data_dir() -> str:
    try:
        import bfcl_eval
    except ImportError as e:
        raise ImportError(
            "This requires the `bfcl-eval` package (`uv pip install bfcl-eval`); "
            "see https://pypi.org/project/bfcl-eval/."
        ) from e
    return os.path.join(os.path.dirname(bfcl_eval.__file__), "data")


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_call_string(call: str) -> dict[str, Any]:
    node = ast.parse(call.strip(), mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError(f"Not a call expression: {call!r}")
    name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
    arguments: dict[str, Any] = {
        f"_pos{i}": ast.literal_eval(arg) for i, arg in enumerate(node.args)
    }
    for kw in node.keywords:
        arguments[kw.arg] = ast.literal_eval(kw.value)
    return {"name": name, "arguments": arguments}


def _normalize_arguments(
    owner: Any, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    positional = sorted(
        ((k, v) for k, v in arguments.items() if k.startswith("_pos")),
        key=lambda kv: int(kv[0][4:]),
    )
    if not positional:
        return arguments
    try:
        param_names = [
            p for p in inspect.signature(getattr(owner, name)).parameters if p != "self"
        ]
    except (TypeError, ValueError):
        param_names = []
    normalized = dict(zip(param_names, (v for _, v in positional)))
    normalized.update({k: v for k, v in arguments.items() if not k.startswith("_pos")})
    return normalized


def _method_owner(instances: dict[str, Any], method_name: str) -> Any | None:
    if method_name.startswith("_"):
        return None
    for instance in instances.values():
        if hasattr(type(instance), method_name):
            return instance
    return None


def execute_call(instances: dict[str, Any], call: dict[str, Any]) -> tuple[str, bool]:
    owner = _method_owner(instances, call["name"])
    if owner is None:
        return f"Error during execution: unknown function {call['name']!r}", True
    try:
        result = getattr(owner, call["name"])(**deepcopy(call.get("arguments") or {}))
    except Exception as e:
        return f"Error during execution: {e}", True
    if isinstance(result, str):
        return result, False
    if isinstance(result, dict):
        try:
            return json.dumps(result), False
        except TypeError:
            return str(result), False
    return str(result), False


def replay(
    involved_classes: list[str], initial_config: dict, calls: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    import importlib

    from bfcl_eval.constants.executable_backend_config import (
        CLASS_FILE_PATH_MAPPING,
        STATELESS_CLASSES,
    )

    instances = {}
    for class_name in involved_classes:
        module = importlib.import_module(CLASS_FILE_PATH_MAPPING[class_name])
        instance = getattr(module, class_name)()
        if class_name not in STATELESS_CLASSES:
            instance._load_scenario(
                deepcopy(initial_config.get(class_name, {})), long_context=False
            )
        instances[class_name] = instance
    observations = []
    for call in calls:
        owner = _method_owner(instances, call["name"])
        if owner is not None:
            call["arguments"] = _normalize_arguments(
                owner, call["name"], call.get("arguments") or {}
            )
        text, _is_error = execute_call(instances, call)
        observations.append(text)
    return instances, observations


def to_json_schema(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: to_json_schema(v) for k, v in node.items() if k != "default"}
        if out.get("type") in _JSON_TYPE_MAP:
            out["type"] = _JSON_TYPE_MAP[out["type"]]
        return out
    if isinstance(node, list):
        return [to_json_schema(v) for v in node]
    return node


def tool_schemas_to_openai(tool_schemas: dict) -> list[dict]:
    tools = []
    for name in sorted(tool_schemas or {}):
        spec = tool_schemas[name]
        if isinstance(spec, dict) and ("parameters" in spec or "description" in spec):
            desc = spec.get("description", "")
            params = to_json_schema(spec.get("parameters", {}))
        else:
            desc, params = "", to_json_schema(spec)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def load_func_docs(
    involved_classes: list[str], excluded_function: list[str] | None = None
) -> dict:
    from bfcl_eval.constants.executable_backend_config import (
        MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    )

    excluded = set(excluded_function or [])
    data_dir = _data_dir()
    schemas: dict[str, dict] = {}
    for class_name in involved_classes:
        doc_file = MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(class_name)
        if not doc_file:
            continue
        for doc in _load_jsonl(os.path.join(data_dir, "multi_turn_func_doc", doc_file)):
            if doc["name"] in excluded:
                continue
            schemas[doc["name"]] = {
                "description": doc.get("description", ""),
                "parameters": to_json_schema(doc.get("parameters", {})),
            }
    return schemas


DEFAULT_SYSTEM_PROMPT = """\
You are a tool-using agent completing a user's request with the function tools provided to you. Work one step at a time.

Rules:
- Make EXACTLY ONE tool call per turn. Emit only the tool call — no extra prose, narration, or markdown fences around it.
- Use only the tools provided to you, with their exact names. Do not invent tools, arguments, or file paths.
- Each user message may require several tool calls before the request is satisfied; keep calling tools until the request is complete, then stop calling tools.
- After each tool result, check whether it succeeded before continuing; do not blindly repeat a failed call with the same arguments.
Use the model's provided tool-calling interface."""


def build_prefix_messages(label: dict, K: int) -> list[dict]:
    observations = label.get("observations")
    if observations is None:
        _, observations = replay(
            label["involved_classes"],
            label["initial_config"],
            deepcopy(label["flattened_calls"]),
        )
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    shown = 0
    for turn in label["turns"]:
        messages.append({"role": "user", "content": turn["user"]})
        for call in turn["calls"]:
            if shown >= K:
                return messages
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{shown}",
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{shown}",
                    "content": str(observations[shown]),
                }
            )
            shown += 1
    return messages


def prefix_turn_index(label: dict, K: int) -> int:
    latest_turn = -1
    shown = 0
    for turn_index, turn in enumerate(label.get("turns", [])):
        latest_turn = turn_index
        for _call in turn.get("calls", []):
            if shown >= K:
                return turn_index
            shown += 1
    return latest_turn


@dataclass
class BfclTurnEnvironment:
    label: dict
    instances: dict[str, Any] = field(default_factory=dict)
    exec_results: list[str] = field(default_factory=list)
    K: int = 0

    def step(self, action: ToolCall) -> Observation:
        call = {"name": action.name, "arguments": action.arguments or {}}
        text, is_error = execute_call(self.instances, call)
        self.exec_results.append(text)
        return Observation(text=text, is_error=is_error)

    def evaluate(self) -> EvalVerdict:
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
            response_checker,
            state_checker,
        )

        ground_truth_instances, ground_truth_results = replay(
            self.label["involved_classes"],
            self.label["initial_config"],
            deepcopy(self.label["flattened_calls"]),
        )
        state_result = state_checker(self.instances, ground_truth_instances)
        if not state_result["valid"]:
            return EvalVerdict(passed=False, detail=state_result["error_message"])
        agent_results_from_k = ground_truth_results[self.K :]
        response_result = response_checker(self.exec_results, agent_results_from_k, 0)
        if not response_result["valid"]:
            return EvalVerdict(passed=False, detail=response_result["error_message"])
        return EvalVerdict(passed=True)


def build_env(label: dict, K: int) -> BfclTurnEnvironment:
    calls = deepcopy(label["flattened_calls"][:K])
    instances, _ = replay(label["involved_classes"], label["initial_config"], calls)
    return BfclTurnEnvironment(label=label, instances=instances, K=K)


@dataclass
class BfclEpisodeResult:
    messages: list[dict]
    calls: list[dict[str, Any]]
    execution_successes: list[bool]
    verdict: EvalVerdict
    final_response: str = ""
    exit_reason: str = "max_turns"

    @property
    def first_call(self) -> dict[str, Any] | None:
        return self.calls[0] if self.calls else None


def run_bfcl_episode(
    label: dict,
    *,
    start_step: int,
    generate: Callable[[list[dict], list[dict]], dict],
    parse_response: Callable[[dict], tuple[str, list[ToolCall]]],
    max_turns: int,
    max_consecutive_errors: int = 3,
    observation_limit: int = 2000,
) -> BfclEpisodeResult:
    messages = build_prefix_messages(label, start_step)
    tools = tool_schemas_to_openai(label.get("tool_schemas", {}))
    env = build_env(label, start_step)
    calls: list[dict[str, Any]] = []
    execution_successes: list[bool] = []
    final_response = ""
    exit_reason = "max_turns"
    consecutive_errors = 0
    current_turn = prefix_turn_index(label, start_step)

    for turn in range(max_turns):
        message = generate(messages, tools)
        content, actions = parse_response(message)
        final_response = content
        if not actions:
            next_turn = current_turn + 1
            turns = label.get("turns", [])
            if next_turn < len(turns):
                messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": turns[next_turn]["user"]},
                    ]
                )
                current_turn = next_turn
                consecutive_errors = 0
                continue
            exit_reason = "no_further_calls"
            break

        observations: list[str] = []
        stop = False
        for action in actions:
            calls.append({"name": action.name, "arguments": action.arguments or {}})
            try:
                observation = env.step(action)
            except Exception:
                execution_successes.append(False)
                exit_reason = "execution_error"
                stop = True
                break

            execution_successes.append(not observation.is_error)
            observations.append(observation.text)
            consecutive_errors = consecutive_errors + 1 if observation.is_error else 0
            if (
                max_consecutive_errors > 0
                and consecutive_errors >= max_consecutive_errors
            ):
                exit_reason = "repeated_errors"
                stop = True
                break

        if stop:
            break

        call_ids = [f"call_t{turn}_{index}" for index in range(len(actions))]
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": action.name,
                            "arguments": action.arguments or {},
                        },
                    }
                    for call_id, action in zip(call_ids, actions)
                ],
            }
        )
        messages.extend(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": observation[:observation_limit],
            }
            for call_id, observation in zip(call_ids, observations)
        )

    try:
        verdict = env.evaluate()
    except Exception as exc:
        verdict = EvalVerdict(passed=False, detail=str(exc))

    return BfclEpisodeResult(
        messages=messages,
        calls=calls,
        execution_successes=execution_successes,
        verdict=verdict,
        final_response=final_response,
        exit_reason=exit_reason,
    )


@dataclass(frozen=True)
class BfclMultiTurnConfig:
    category: str = "multi_turn_base"
    eval_tail: int = 30
    obs_limit: int = 1500


class BfclMultiTurnDataset(DatasetConfig):
    def __init__(
        self,
        split: str = "train",
        config: BfclMultiTurnConfig | None = None,
    ) -> None:
        self._split = split
        self.config = config if config is not None else BfclMultiTurnConfig()

    def cache_key(self) -> str:
        payload = json.dumps(
            {
                "split": self._split,
                "config": {
                    "category": self.config.category,
                    "eval_tail": self.config.eval_tail,
                    "obs_limit": self.config.obs_limit,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return f"bfcl-{self.config.category}-{self._split}-{fingerprint}"

    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> str:
        return "label"

    def _make_row(self, entry: dict, ground_truth: list[list[str]]) -> dict:
        turns = []
        for turn_messages, calls in zip(entry["question"], ground_truth):
            user = next(
                (
                    str(m["content"])
                    for m in turn_messages
                    if m.get("role") == "user" and str(m.get("content", "")).strip()
                ),
                "",
            )
            turns.append(
                {"user": user, "calls": [parse_call_string(c) for c in calls]}
            )
        flattened_calls = [c for turn in turns for c in turn["calls"]]
        _, observations = replay(
            entry["involved_classes"], entry["initial_config"], flattened_calls
        )
        label = {
            "task_id": entry["id"],
            "initial_config": entry["initial_config"],
            "involved_classes": entry["involved_classes"],
            "excluded_function": entry.get("excluded_function", []),
            "turns": turns,
            "flattened_calls": flattened_calls,
            "observations": [str(o)[: self.config.obs_limit] for o in observations],
            "tool_schemas": load_func_docs(
                entry["involved_classes"], entry.get("excluded_function")
            ),
            "total_steps": len(flattened_calls),
        }
        messages = [
            {
                "role": "system",
                "content": "You are a tool-calling agent. Output a JSON tool call.",
            },
            {"role": "user", "content": turns[0]["user"] if turns else ""},
        ]
        return {"messages": messages, "label": json.dumps(label)}

    def _load_split(self) -> list[dict]:
        from bfcl_eval.constants.category_mapping import VERSION_PREFIX

        data_dir = _data_dir()
        filename = f"{VERSION_PREFIX}_{self.config.category}.json"
        entries_by_id = {e["id"]: e for e in _load_jsonl(os.path.join(data_dir, filename))}
        gt_by_id = {
            e["id"]: e["ground_truth"]
            for e in _load_jsonl(os.path.join(data_dir, "possible_answer", filename))
        }
        ids = list(entries_by_id.keys())
        tail = self.config.eval_tail
        if self._split == "eval":
            ids = ids[-tail:] if tail else []
        elif self._split == "train":
            ids = ids[:-tail] if tail else ids
        return [
            self._make_row(entries_by_id[i], gt_by_id[i])
            for i in ids
            if i in entries_by_id and i in gt_by_id and gt_by_id[i] and any(gt_by_id[i])
        ]

    def rows(self) -> list[dict]:
        return self._load_split()
