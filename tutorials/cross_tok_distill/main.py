# ---
# order: 4
# deps: bfcl-eval, jsonschema
# ---
#
# # On-policy distillation across model families
#
# In the [last OPD tutorial](https://dojo.modal.dev/tutorials/on_policy_distillation),
# we saw how same-family OPD minimizes reverse-KL on shared token IDs. However, if you
# wanted to use a teacher model from another model family, you'll quickly find out that
# they don't share a vocabulary, so reverse-KL on raw token logprobs is undefined.
#
# This tutorial uses [SimCT](https://arxiv.org/abs/2605.07711) to align tokenizers,
# then trains [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) with a
# [DeepSeek-V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
# teacher on
# [BFCL](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)
# multi-turn tool calling.

import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import (
    BfclMultiTurnConfig,
    BfclMultiTurnDataset,
    _tokenizer,
    build_env,
    build_prefix_messages,
    prefix_turn_index,
    run_bfcl_episode,
    to_json_schema,
    tool_schemas_to_openai,
)
from modal_dojo import (
    CustomDeployment,
    Endpoint,
    Qwen3_6_35B,
    SglangRecipe,
    TrainConfig,
)
from modal_dojo.common.models.base import HFModelConfiguration, ToolCall
from modal_dojo.train_recipes.slime_recipe import Qwen3_6_35B_Recipe

# ## Deploy the base models
#
# First, we'll deploy the teacher and base models to derive a baseline.
# You'll notice that even if the Modal Dojo doesn't have a native model class for a model you want to use,
# you can just use [HFModelConfiguration](https://dojo.modal.dev/reference/hfmodelconfiguration)!

STUDENT_READY_TIMEOUT = 15 * 60
TEACHER_READY_TIMEOUT = 30 * 60

student_model = Qwen3_6_35B()


def deploy_base_models():
    print("deploying base student endpoint...")
    base_student_deployment = Endpoint.launch(
        student_model, unauthenticated=True, recreate_if_existing=True
    )

    teacher_model = HFModelConfiguration(model_name="deepseek-ai/DeepSeek-V4-Flash")
    print("deploying teacher deployment...")
    teacher_deployment = CustomDeployment.launch(
        teacher_model,
        recipe=SglangRecipe(
            gpu="B200",
            tp=4,
            dp=4,
            context_length=16384,
            mem_fraction_static=0.85,
            chunked_prefill_size=4096,
            max_running_requests=64,
            sglang_image="lmsysorg/sglang:v0.5.12.post1-cu130",
            install_transformers_from_git=False,
            env_vars={
                "NCCL_CUMEM_ENABLE": "1",
                "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "8320",
                "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS": "1",
                "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND": "1",
            },
            extra_server_args={
                "--trust-remote-code": "",
                "--moe-a2a-backend": "megamoe",
                "--enable-breakable-cuda-graph": "",
                "--enable-mixed-chunk": "",
                "--piecewise-cuda-graph-max-tokens": "4096",
                "--tool-call-parser": "deepseekv4",
                "--reasoning-parser": "deepseek-v4",
            },
            startup_timeout=TEACHER_READY_TIMEOUT,
        ),
        app_name="dsv4-teacher-model",
        served_model_name="deepseek-v4-flash",
    )

    base_student_deployment.wait_until_ready(timeout=STUDENT_READY_TIMEOUT)
    print(f"base student model deployed to {base_student_deployment.url}")

    teacher_deployment.wait_until_ready(timeout=TEACHER_READY_TIMEOUT)
    print(f"teacher model deployed to {teacher_deployment.url}")

    return base_student_deployment, teacher_deployment


TEACHER_RM_CONCURRENCY = 24

STUDENT_ENABLE_THINKING = False

# ## Define a scoring function
#
# Trajectories are scored as follows:
#
# $$
# 0.25 \times \mathrm{mean\_partial} + 0.20 \times \mathrm{first\_call} + 0.10 \times \mathrm{exec\_score} + 0.45 \times \mathrm{terminal\_pass}
# $$

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
_TMP_RE = re.compile(r"/tmp/[^\s/]+")


def _normalize_value(v) -> str:
    s = str(v).strip().lower()
    s = _UUID_RE.sub("<uuid>", s)
    s = _TS_RE.sub("<timestamp>", s)
    s = _TMP_RE.sub("/tmp/<tmp>", s)
    return s


def _coerce_args(call: dict | None) -> dict:
    if not call:
        return {}
    args = call.get("arguments", {})
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _validates(call: dict, schema: dict) -> bool:
    try:
        import jsonschema

        jsonschema.validate(_coerce_args(call), to_json_schema(schema))
        return True
    except Exception:
        return False


def _score_structural_match(student_call: dict | None, expert_call: dict) -> float:
    if not student_call:
        return 0.0
    if student_call.get("name") != expert_call.get("name"):
        return 0.0
    score = 0.4

    s_vals = _coerce_args(student_call)
    e_vals = _coerce_args(expert_call)
    s_args = set(s_vals.keys())
    e_args = set(e_vals.keys())
    if s_args == e_args:
        score += 0.3
    elif s_args & e_args:
        score += 0.15
    else:
        return score

    shared = s_args & e_args
    if not shared:
        score += 0.3
    else:
        matches = sum(
            1
            for k in shared
            if _normalize_value(s_vals.get(k)) == _normalize_value(e_vals.get(k))
        )
        score += 0.3 * (matches / len(shared))
    return score


def _partial_credit(
    student_call: dict | None, expert_call: dict, tool_schemas: dict
) -> float:
    if not student_call or "name" not in student_call:
        return 0.0
    score = 0.20
    spec = tool_schemas.get(student_call.get("name"))
    if spec is not None:
        score += 0.15
        schema = spec.get("parameters", spec) if isinstance(spec, dict) else spec
        if _validates(student_call, schema):
            score += 0.15
    score += 0.50 * _score_structural_match(student_call, expert_call)
    return min(1.0, score)


def trajectory_reward(
    student_calls: list,
    exec_successes: list,
    expert_calls: list,
    tool_schemas: dict,
    task_passed: bool | None,
    tail_len: int,
) -> float:
    T = max(int(tail_len), 1)
    graded = min(len(student_calls), len(expert_calls), T)
    partial_sum = sum(
        _partial_credit(student_calls[j], expert_calls[j], tool_schemas)
        for j in range(graded)
    )
    mean_partial = partial_sum / T
    first_call_score = (
        _partial_credit(student_calls[0], expert_calls[0], tool_schemas)
        if graded and expert_calls
        else 0.0
    )
    successful_execs = sum(1.0 for ok in exec_successes if ok)
    exec_score = min(successful_execs, T) / T
    if task_passed is not None:
        verify_score = 1.0 if task_passed else 0.0
        return (
            0.25 * mean_partial
            + 0.20 * first_call_score
            + 0.10 * exec_score
            + 0.45 * verify_score
        )
    return mean_partial


# ## Get the dataset
#
# The [dataset preprocessing code](https://github.com/modal-projects/modal-dojo/blob/main/tutorials/cross_tok_distill/env.py)
# is verbose, so we simply instantiate the datasets here.

dataset_config = BfclMultiTurnConfig(eval_tail=30)
dataset = BfclMultiTurnDataset(split="train", config=dataset_config)
eval_dataset = BfclMultiTurnDataset(split="eval", config=dataset_config)

# ## Implement a custom curriculum
#
# A well-known technique in machine learning is
# [gradually increasing the difficulty](https://dl.acm.org/doi/epdf/10.1145/1553374.1553380)
# of the training task to speed up convergence and improve the quality of local optima.
# Here, we implement a reverse-K learning cirriculum. First, we'll initialize the student model's
# context up to the Kth tool call in the conversation, where K is initialized to N calls - 1.
# As the student model completes a task, K is decremented until the student is given the
# starter prompt or rollouts have completed.

CURRICULUM_TAIL_MIN = 1


def curriculum_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.rollout.sglang_rollout import generate_rollout

    if evaluation:
        return generate_rollout(args, rollout_id, data_source, evaluation=True)

    if not hasattr(args, "curriculum_tail") or rollout_id == 0:
        args.curriculum_tail = CURRICULUM_TAIL_MIN

    T = args.curriculum_tail
    result = generate_rollout(args, rollout_id, data_source, evaluation=False)
    args.curriculum_tail = T + 1
    print(
        f"[curriculum] iter={rollout_id} tail={T} -> {args.curriculum_tail}",
        flush=True,
    )
    return result


# ## Baseline eval
#
# Using our K curriculum, we initialize the agent context and the task's
# class instances to the Kth ground-truth call, where K is set to N calls - 1.
# Then, we get our baseline performance for both models.

SERVED_CONTEXT_LEN = 16384
RESPONSE_TOKEN_CAP = 8192
CONTEXT_SAFETY_MARGIN = 512
EVAL_TAIL_STEPS = CURRICULUM_TAIL_MIN
EVAL_MAX_TURNS = EVAL_TAIL_STEPS * 2
MAX_CONSECUTIVE_TOOL_ERRORS = 3


def _chat(
    deployment,
    messages,
    tools=None,
    max_tokens=RESPONSE_TOKEN_CAP,
    max_attempts=12,
    *,
    qwen_thinking=False,
):
    try:
        tok = _tokenizer("Qwen/Qwen3.6-35B-A3B")
        text = tok.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = len(tok(text, add_special_tokens=False)["input_ids"])
    except Exception:
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 3
    remaining = SERVED_CONTEXT_LEN - prompt_tokens - CONTEXT_SAFETY_MARGIN
    capped = min(max_tokens, remaining)
    if capped <= 0:
        return {"content": "", "tool_calls": []}

    extra = {"temperature": 0.0, "max_tokens": capped}
    if tools is not None:
        extra["tools"] = tools
    if qwen_thinking is not None:
        extra["chat_template_kwargs"] = {"enable_thinking": qwen_thinking}
    return deployment.chat(messages, max_attempts=max_attempts, **extra)


def _actions_from_message(msg: dict) -> tuple[str, list[ToolCall]]:
    content = msg.get("content") or msg.get("reasoning_content") or ""
    structured = msg.get("tool_calls") or []
    if structured:
        actions: list[ToolCall] = []
        for tc in structured:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    raw_args = {}
            if not isinstance(raw_args, dict):
                raw_args = {}
            name = fn.get("name") or ""
            if name:
                actions.append(ToolCall(name=name, arguments=raw_args))
        return content, actions
    parsed = student_model.parse_response(content)
    return parsed.content, parsed.tool_calls


def bfcl_eval_fn(deployment, example: dict) -> dict:
    label = json.loads(example["label"])
    N = label["total_steps"]
    flattened_calls = label["flattened_calls"]
    K = max(0, N - EVAL_TAIL_STEPS)
    expert_call = flattened_calls[K] if K < len(flattened_calls) else {}
    is_student = getattr(deployment, "served_model_name", None) != "deepseek-v4-flash"

    episode = run_bfcl_episode(
        label,
        start_step=K,
        generate=lambda messages, tools: _chat(
            deployment,
            messages,
            tools=tools,
            qwen_thinking=STUDENT_ENABLE_THINKING if is_student else None,
        ),
        parse_response=_actions_from_message,
        max_turns=EVAL_MAX_TURNS,
        max_consecutive_errors=MAX_CONSECUTIVE_TOOL_ERRORS,
    )
    expert_calls = flattened_calls[K : K + len(episode.calls)]
    shaped_score = trajectory_reward(
        episode.calls,
        episode.execution_successes,
        expert_calls,
        label.get("tool_schemas", {}),
        bool(episode.verdict.passed),
        EVAL_TAIL_STEPS,
    )
    first_call = episode.first_call
    return {
        "score": shaped_score,
        "response": episode.final_response,
        "metadata": {
            "task": label["task_id"],
            "step_K": K,
            "eval_tail": EVAL_TAIL_STEPS,
            "task_passed": bool(episode.verdict.passed),
            "terminal_score": 1.0 if episode.verdict.passed else 0.0,
            "shaped_reward": shaped_score,
            "exec_successes": sum(episode.execution_successes),
            "exec_calls": len(episode.execution_successes),
            "parsed_call": first_call is not None,
            "tool_match": bool(
                first_call and first_call.get("name") == expert_call.get("name")
            ),
        },
    }


def _frac(rows, key):
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["metadata"].get(key)) / len(rows)


def run_eval(deployment, *, ready_timeout, max_concurrency: int = 4):
    from concurrent.futures import ThreadPoolExecutor

    deployment.wait_until_ready(timeout=ready_timeout)
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        rows = list(
            executor.map(
                lambda example: bfcl_eval_fn(deployment, example), eval_dataset.rows()
            )
        )
    mean = sum(r["score"] for r in rows) / len(rows) if rows else float("nan")
    return mean, rows


def _print_eval_summary(mean: float, rows: list[dict], *, prefix: str = "") -> None:
    label = f"{prefix} " if prefix else ""
    print(f"{label}shaped reward: {mean:.3f}")
    print(f"{label}terminal pass rate: {_frac(rows, 'task_passed'):.1%}")
    if rows:
        print(f"{label}parsed tool call: {_frac(rows, 'parsed_call'):.1%}")
        print(f"{label}first-call tool match: {_frac(rows, 'tool_match'):.1%}")


def run_baseline_evals(base_student_deployment, teacher_deployment):
    print("running teacher model evaluation...")
    try:
        teacher_mean, teacher_rows = run_eval(
            teacher_deployment,
            ready_timeout=TEACHER_READY_TIMEOUT,
            max_concurrency=4,
        )
        _print_eval_summary(teacher_mean, teacher_rows)
    except Exception as e:
        print(
            f"[teacher-eval] failed: ({e!r})",
            flush=True,
        )

    print("running base student model evaluation...")
    try:
        base_mean, base_rows = run_eval(
            base_student_deployment,
            ready_timeout=STUDENT_READY_TIMEOUT,
            max_concurrency=4,
        )
        _print_eval_summary(base_mean, base_rows)
    except Exception as e:
        print(
            f"[base-eval] failed: ({e!r})",
            flush=True,
        )


# ## Creating a reward function
#
# Slime folds the reverse-KL loss from OPD into the GRPO advantage:
#
# $$
# A_t = A_t^{\mathrm{GRPO}} + \lambda
# \left(\log \pi_{\mathrm{student}}(y_t) - \log \pi_{\mathrm{teacher}}(y_t)\right)
# $$
#
# where $\lambda$ is `--opd-kl-coef`, set to `0.3` here.

_teacher_rm_sem: asyncio.Semaphore | None = None


async def cross_tokenizer_reward(args, sample, **kwargs):
    import random

    import aiohttp
    from modal_dojo.common.deployment import _modal_proxy_auth_headers

    tokenizer = _tokenizer("Qwen/Qwen3.6-35B-A3B")
    resp_len = max(1, sample.response_length)
    split = max(0, len(sample.tokens) - resp_len)
    prefix_text = tokenizer.decode(sample.tokens[:split], skip_special_tokens=True)
    response_text = tokenizer.decode(sample.tokens[split:], skip_special_tokens=True)
    full_text = prefix_text + response_text

    teacher_tok = _tokenizer("deepseek-ai/DeepSeek-V4-Flash")
    try:
        offsets = teacher_tok(
            full_text, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        prompt_length = sum(1 for start, _end in offsets if start < len(prefix_text))
    except (TypeError, KeyError, NotImplementedError, ValueError):
        prompt_length = len(
            teacher_tok(prefix_text, add_special_tokens=False)["input_ids"]
        )

    payload = {
        "text": full_text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": prompt_length,
        "return_text_in_logprobs": True,
    }
    skip_opd = {"meta_info": {"input_token_logprobs": []}}
    request_timeout = max(180, 60 + resp_len // 20)
    max_attempts = 8
    rm_limit = int(getattr(args, "teacher_rm_concurrency", 24) or 24)

    global _teacher_rm_sem
    if _teacher_rm_sem is None:
        _teacher_rm_sem = asyncio.Semaphore(max(1, rm_limit))
    async with _teacher_rm_sem:
        for attempt in range(max_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        args.rm_url,
                        json=payload,
                        headers=_modal_proxy_auth_headers(),
                        timeout=aiohttp.ClientTimeout(total=request_timeout),
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                status = getattr(e, "status", None)
                if status == 400:
                    return skip_opd
                is_retryable = status in (503, 502, 504, 429, None)
                if attempt == max_attempts - 1 or not is_retryable:
                    return skip_opd
                base = min(30.0, 2 ** min(attempt, 5))
                await asyncio.sleep(base + random.uniform(0, base * 0.25))
    return skip_opd


# ## Cross-tokenizer alignment
#
# [SimCT](https://arxiv.org/abs/2605.07711) constructs minimally aligned units (MAUs)
# between boundaries of character-aligned tokens by greedily finding shared character
# boundaries between both tokenizations.
#
# As an example, for the sentence "I am happy today", the student might tokenize
# `["I", "am", "ha", "pp", "y", "today"]` and the teacher
# `["I", "am", "hap", "py", "today"]`. While `"I"` and `"am"` already match, `"happy"`
# does not, so SimCT groups `["ha", "pp", "y"]` and `["hap", "py"]` into one MAU.
#
# The MAU log-probability is the joint probability in log space, or the sum of its
# constituent token logprobs (or the log of the product of token probabilities).
# To integrate with slime's per-token KL, we distribute the teacher's MAU logprob
# sum equally across student tokens in the MAU, so that slime's per-token summation
# reconstructs the correct MAU-level reverse KL.

MAX_TURNS = 16


def align_cross_tokenizer(
    teacher_token_texts: list[str],
    teacher_logprobs: list[float],
    student_token_texts: list[str],
):
    import torch

    def _offsets(texts):
        out, pos = [], 0
        for t in texts:
            out.append((pos, pos + len(t)))
            pos += len(t)
        return out

    t_off = _offsets(teacher_token_texts)
    s_off = _offsets(student_token_texts)
    t_bounds = {s for s, e in t_off} | {e for s, e in t_off}
    s_bounds = {s for s, e in s_off} | {e for s, e in s_off}
    shared = sorted(t_bounds & s_bounds)

    result = torch.zeros(len(student_token_texts), dtype=torch.float32)
    covered = torch.zeros(len(student_token_texts), dtype=torch.bool)
    for i in range(len(shared) - 1):
        lo, hi = shared[i], shared[i + 1]
        t_idx = [j for j, (s, e) in enumerate(t_off) if s >= lo and e <= hi]
        s_idx = [j for j, (s, e) in enumerate(s_off) if s >= lo and e <= hi]
        if t_idx and s_idx:
            mau_lp_sum = sum(teacher_logprobs[j] for j in t_idx)
            per_student_token = mau_lp_sum / len(s_idx)
            for j in s_idx:
                result[j] = per_student_token
                covered[j] = True
    return result, covered


async def tool_step_generate(args, sample, sampling_params):
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    label = json.loads(getattr(sample, "label", "{}"))

    task_id = label["task_id"]
    N = label.get("total_steps", 1)
    flattened_calls = label.get("flattened_calls", [])
    tool_schemas = label.get("tool_schemas", {})

    T = int(getattr(args, "curriculum_tail", EVAL_TAIL_STEPS))
    K = max(0, N - T)
    max_turns = min(int(getattr(args, "max_turns", MAX_TURNS)), max(T * 2, 4))

    prefix_msgs = build_prefix_messages(label, K)
    tools_list = tool_schemas_to_openai(tool_schemas)
    prompt_text = state.tokenizer.apply_chat_template(
        prefix_msgs,
        tools=tools_list,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=STUDENT_ENABLE_THINKING,
    )
    prompt_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    base_render = state.tokenizer.apply_chat_template(
        prefix_msgs,
        tools=tools_list,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=STUDENT_ENABLE_THINKING,
    )
    gen_suffix = prompt_text[len(base_render) :]
    probe = state.tokenizer.apply_chat_template(
        prefix_msgs
        + [
            {"role": "assistant", "content": "\x01A\x01"},
            {"role": "tool", "content": "\x00OBS\x00"},
        ],
        tools=tools_list,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=STUDENT_ENABLE_THINKING,
    )
    after_assistant = probe.split("\x01A\x01", 1)[1]
    obs_open, _rest = after_assistant.split("\x00OBS\x00", 1)
    obs_close = _rest[: len(_rest) - len(gen_suffix)]
    user_probe = state.tokenizer.apply_chat_template(
        prefix_msgs
        + [
            {"role": "assistant", "content": "\x01A\x01"},
            {"role": "user", "content": "\x00USER\x00"},
        ],
        tools=tools_list,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=STUDENT_ENABLE_THINKING,
    )
    after_assistant = user_probe.split("\x01A\x01", 1)[1]
    user_open, _rest = after_assistant.split("\x00USER\x00", 1)
    user_close = _rest[: len(_rest) - len(gen_suffix)]
    stop_tok = obs_open.split("\n", 1)[0]

    def _log(msg):
        print(f"[rollout:{task_id} T={T} K={K}] {msg}", flush=True)

    def _abort_sample():
        pad_id = (
            state.tokenizer.pad_token_id
            if state.tokenizer.pad_token_id is not None
            else state.tokenizer.eos_token_id
            if state.tokenizer.eos_token_id is not None
            else prompt_ids[-1]
        )
        sample.status = Sample.Status.ABORTED
        sample.tokens = prompt_ids + [pad_id]
        sample.response_length = 1
        sample.response = ""
        sample.loss_mask = [0]
        return sample

    try:
        env = await asyncio.to_thread(build_env, label, K)
    except Exception as e:
        _log(f"env build failed: {e!r} — skipping rollout")
        return _abort_sample()

    trajectory_text = ""
    response_segments: list[tuple[str, int]] = []
    student_calls: list = []
    exec_successes: list = []
    finish_type = "stop"
    current_turn = prefix_turn_index(label, K)
    for turn in range(max_turns):
        output = await post(
            url,
            {
                "text": prompt_text + trajectory_text,
                "sampling_params": sampling_params,
            },
        )
        finish_type = output["meta_info"]["finish_reason"]["type"]
        if finish_type == "abort":
            return _abort_sample()

        model_text = output["text"]
        trajectory_text += model_text
        response_segments.append((model_text, 1))

        actions = student_model.parse_response(model_text).tool_calls
        action = actions[0] if actions else None

        if action is None:
            next_turn = current_turn + 1
            turns = label.get("turns", [])
            if finish_type != "length" and next_turn < len(turns):
                seg_open = (
                    user_open[len(stop_tok) :]
                    if model_text.endswith(stop_tok)
                    else user_open
                )
                user_segment = (
                    seg_open + turns[next_turn]["user"] + user_close + gen_suffix
                )
                trajectory_text += user_segment
                response_segments.append((user_segment, 0))
                current_turn = next_turn
                continue
            break

        student_calls.append({"name": action.name, "arguments": action.arguments})
        try:
            result = await asyncio.to_thread(env.step, action)
            obs_text, is_error = result.text, result.is_error
        except Exception as e:
            _log(f"turn {turn} execution error: {e!r} — ending episode")
            exec_successes.append(False)
            finish_type = "stop"
            break
        exec_successes.append(not is_error)
        seg_open = (
            obs_open[len(stop_tok) :] if model_text.endswith(stop_tok) else obs_open
        )
        obs_segment = seg_open + obs_text[:2000] + obs_close + gen_suffix
        trajectory_text += obs_segment
        response_segments.append((obs_segment, 0))
        if finish_type == "length":
            break

    try:
        verdict = env.evaluate()
        task_passed = verdict.passed
    except Exception as e:
        _log(f"evaluate() failed: {e!r} — marking failed")
        task_passed = False

    expert_calls = flattened_calls[K : K + len(student_calls)]
    shaped = trajectory_reward(
        student_calls,
        exec_successes,
        expert_calls,
        tool_schemas,
        bool(task_passed),
        T,
    )

    response_token_ids: list[int] = []
    loss_masks: list[int] = []
    for seg, trainable in response_segments:
        tids = state.tokenizer(seg, add_special_tokens=False)["input_ids"]
        response_token_ids += tids
        loss_masks += [trainable] * len(tids)

    sample.tokens = prompt_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = trajectory_text
    sample.loss_mask = loss_masks
    sample.status = (
        Sample.Status.TRUNCATED if finish_type == "length" else Sample.Status.COMPLETED
    )
    sample.metadata = {
        "student_calls": student_calls,
        "exec_successes": exec_successes,
        "expert_calls": expert_calls,
        "tool_schemas": tool_schemas,
        "task_passed": bool(task_passed),
        "tail_len": T,
        "curriculum_tail": T,
        "step_K": K,
        "shaped_reward": shaped,
        "task_reward": shaped,
    }
    sample.bfcl_student_calls = student_calls
    sample.bfcl_exec_successes = exec_successes
    sample.bfcl_expert_calls = expert_calls
    sample.bfcl_tool_schemas = tool_schemas
    sample.bfcl_task_passed = bool(task_passed)
    sample.bfcl_tail_len = T
    return sample


# ## Cross-tokenizer post-process
#
# Upon successful completion of a response from the student trainer node, we take the raw prompt prefix
# and response token IDs, and decode these (using the Qwen tokenizer) into the raw text.
# This raw text is posted to the teacher `/generate`` endpoint for logprob computation.
# Upon successful logprob computation, the student receives a response from the teacher that contains
# the logprob, token ID, and decoded text (using the DeepSeek tokenizer).
#
# To align the text, we create two arrays that contain the start and end character boundaries for
# teacher and student tokens. The intersection of start and end boundaries between the teacher and
# student arrays is the aligned text, and any tokens that are in between these two boundaries
# are used for the SimCT MAU calculation described above.


def cross_tokenizer_post_process(args, samples, **kwargs):
    import torch

    tokenizer = _tokenizer("Qwen/Qwen3.6-35B-A3B")
    raw_rewards = [s.get_reward_value(args) for s in samples]
    rewards = []
    first_call_hits = 0
    for sample in samples:
        meta = getattr(sample, "metadata", {}) or {}
        sc_list = (
            getattr(sample, "bfcl_student_calls", None)
            or meta.get("student_calls")
            or []
        )
        ec_list = (
            getattr(sample, "bfcl_expert_calls", None) or meta.get("expert_calls") or []
        )
        sc0 = sc_list[0] if sc_list else None
        ec0 = ec_list[0] if ec_list else {}
        if sc0 and ec0 and sc0.get("name") == ec0.get("name"):
            first_call_hits += 1
        task_passed = getattr(sample, "bfcl_task_passed", None)
        if task_passed is None:
            task_passed = meta.get("task_passed")
        r = trajectory_reward(
            sc_list,
            getattr(sample, "bfcl_exec_successes", None)
            or meta.get("exec_successes")
            or [],
            ec_list,
            getattr(sample, "bfcl_tool_schemas", None)
            or meta.get("tool_schemas")
            or {},
            task_passed,
            getattr(sample, "bfcl_tail_len", None) or meta.get("tail_len", 1),
        )
        rewards.append(r)
        sample.reward = r
        if isinstance(meta, dict):
            meta["shaped_reward"] = r
            meta["task_reward"] = r
            sample.metadata = meta

    cur_tail = getattr(args, "curriculum_tail", None)
    if rewards:
        n = len(samples)
        n_pass = sum(
            1 for s in samples if (getattr(s, "metadata", {}) or {}).get("task_passed")
        )
        all_exec = [
            ok
            for s in samples
            for ok in ((getattr(s, "metadata", {}) or {}).get("exec_successes") or [])
        ]
        exec_ok = (sum(1 for ok in all_exec if ok) / len(all_exec)) if all_exec else 0.0
        print(
            f"[bfcl] tail={cur_tail} "
            f"pass={n_pass}/{n} first_call={first_call_hits}/{n} exec_ok={exec_ok:.2f}",
            flush=True,
        )

    unaligned = total_resp = opd_dropped = 0
    for sample, reward in zip(samples, raw_rewards):
        entries = (reward or {}).get("meta_info", {}).get("input_token_logprobs", [])
        t_lps = [e[0] if e[0] is not None else 0.0 for e in entries if e is not None]
        t_texts = [e[2] if len(e) > 2 else "" for e in entries if e is not None]
        resp_tokens = (
            sample.tokens[-sample.response_length :] if sample.response_length else []
        )
        s_texts = [
            tokenizer.decode([tid], skip_special_tokens=True) for tid in resp_tokens
        ]
        aligned, covered = align_cross_tokenizer(t_texts, t_lps, s_texts)

        if not entries:
            sample.teacher_log_probs = torch.full(
                (len(aligned),), float("nan"), dtype=torch.float32
            )
            opd_dropped += 1
            continue

        sample.teacher_log_probs = aligned
        unaligned += int((~covered).sum().item())
        total_resp += int(covered.numel())

    if total_resp:
        print(f"[simct] aligned={1 - (unaligned / total_resp):.1%}", flush=True)
    if opd_dropped:
        print(
            f"[opd] dropped={opd_dropped}/{len(samples)} "
            "(teacher unavailable; GRPO retained)",
            flush=True,
        )
    return rewards, rewards


# ## Start training
#
# With all that in place, it's dead simple to kick off training.


def build_config(teacher_deployment):
    return TrainConfig(
        model=student_model,
        dataset=dataset,
        recipe=Qwen3_6_35B_Recipe(
            colocate=False,
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            rollout_num_gpus=8,
            rollout_num_gpus_per_engine=8,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            expert_model_parallel_size=4,
            sglang_dp_size=8,
            sglang_enable_dp_attention=True,
            sglang_ep_size=8,
            sglang_cuda_graph_bs=[1, 2, 4, 8, 16, 24, 32, 48],
            sglang_max_running_requests=48,
            num_rollout=5,
            rollout_batch_size=16,
            n_samples_per_prompt=8,
            global_batch_size=16,
            rollout_max_response_len=4000,
            sglang_mem_fraction_static=0.75,
            use_kl_loss=True,
            kl_loss_coef=0.02,
            no_save_optim=True,
            custom_rm_function=cross_tokenizer_reward,
            custom_generate_function=tool_step_generate,
            custom_reward_post_process_function=cross_tokenizer_post_process,
            rollout_function=curriculum_rollout,
            image_overlay=lambda img: img.pip_install(
                "modal~=1.5.5",
                "huggingface_hub~=1.12.0",
                "aiohttp~=3.13.0",
                "jsonschema~=4.23.0",
                "bfcl-eval==2026.3.23",
            ).add_local_file(
                str(Path(__file__).with_name("env.py")),
                remote_path="/root/env.py",
                copy=True,
            ),
            environment={
                "PYTHONPATH": "/root/Megatron-LM/:/root",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
            extra_config={
                "use_opd": True,
                "opd_type": "sglang",
                "opd_kl_coef": 0.3,
                "rm_url": f"{teacher_deployment.url}/generate",
                "teacher_rm_concurrency": TEACHER_RM_CONCURRENCY,
                "max_turns": MAX_TURNS,
            },
        ),
    )


def train(config):
    with config.launch() as run:
        print(f"run id: {run.training_run_id}")
        checkpoint = None
        while True:
            done = run.done()
            latest = run.latest_checkpoint()
            if latest is not None and latest != checkpoint:
                checkpoint = latest
                print(f"new checkpoint: {checkpoint.path}")
            if done:
                break
            time.sleep(30)
        if checkpoint is None:
            raise RuntimeError("run produced no checkpoint")
        print(f"checkpoint: {checkpoint.path}")
    return checkpoint


# ## Evaluate the trained student
#
# We'll deploy our trained student and run the same evaluation on it.


def deploy_trained_model(checkpoint):
    print("deploying trained student endpoint...")
    trained_student_deployment = Endpoint.launch(
        student_model, checkpoint, unauthenticated=True, recreate_if_existing=True
    )
    trained_student_deployment.wait_until_ready(timeout=STUDENT_READY_TIMEOUT)
    print(f"checkpoint deployed to {trained_student_deployment.url}")

    return trained_student_deployment


def run_trained_evals(trained_student_deployment):
    print("running trained student model evaluation...")
    trained_mean, trained_rows = run_eval(
        trained_student_deployment,
        ready_timeout=STUDENT_READY_TIMEOUT,
        max_concurrency=4,
    )
    _print_eval_summary(trained_mean, trained_rows)


if __name__ == "__main__":
    base_student_deployment, teacher_deployment = deploy_base_models()
    run_baseline_evals(base_student_deployment, teacher_deployment)
    config = build_config(teacher_deployment)
    checkpoint = train(config)
    trained_student_deployment = deploy_trained_model(checkpoint)
    run_trained_evals(trained_student_deployment)

# ## Results
#
# | Metric | Teacher | Base | Trained | Delta |
# | --- | ---: | ---: | ---: | ---: |
# | Eval rows | 30 | 30 | 30 | — |
# | Shaped reward | 0.751 | 0.707 | 0.789 | +0.082 |
# | Terminal pass rate | 83.3% | 53.3% | 63.3% | +10.0 pp |
# | Parsed tool call | 100.0% | 90.0% | 96.7% | +6.7 pp |
# | First-call tool match | 93.3% | 86.7% | 93.3% | +6.6 pp |
#
# This is after only 5 rollouts; more rollouts would likely see further improvement.
#
# ## Future Possibilities
#
# Some possible next steps for the tutorial:
# 1. Augment the training dataset with more long-context tasks from BFCL's Long-Context Multi-Turn category.
# 2. Experiment with different student and teacher model configurations.
# 3. Add in privledged information to the teacher model for an even stronger distillation signal.
# 4. Extend Slime's OPD loss to include top-k logprobs from the teacher model.
# 5. Try a different cross-tokenizer alignment strategy than averaging logprobs for reverse-KL
# such as [f-divergence](https://neurips.cc/virtual/2025/loc/san-diego/poster/119176).
# 6. Increase the `--opd-kl-coef` value from 0.3 to see if a stronger reverse-KL signal improves training.
# 7. Ablate the OPD term completely from the GRPO advantage by setting `--opd-kl-coef` to 0.0.
#
# Cross-tokenizer distillation is a novel OPD technique, and the misalignment of tokenizers across model
# families makes it an interesting research problem!
