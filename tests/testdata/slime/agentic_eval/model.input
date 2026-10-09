"""In-process mini-swe ``Model`` that records exact tokens — the model-API seam.

Token-in-token-out: each turn appends the new-context delta (loss_mask 0) and the
exact ids SGLang returned (loss_mask 1) plus per-token logprobs and the
``weight_version`` live at generation. The prompt is the one running token
sequence; the new-context delta is rendered from the *observation* messages only
(never re-rendering a prior assistant turn), so neither Qwen's ``<think>``
history-stripping nor the tool-call-arg whitespace rstrip can drift the prompt
away from the training target. Tool calls are parsed from the raw output with the
served model's SGLang parser (native tool-calls, qwen3_coder family).

A fresh ``DefaultAgent`` per episode leg (harbor multi-step) appears as a new
chain → a sibling training Sample. One instance per episode; read ``chains`` /
``aborted`` / stats after the run.
"""

from __future__ import annotations

import json
import secrets
import time
import urllib.request
from dataclasses import dataclass, field

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError, LimitsExceeded
from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages

from slime.agent.parsing import parse_model_output

from .prompts import BASH_TOOL, FORMAT_ERROR_TEMPLATE, OBSERVATION_TEMPLATE

_RENDER_KEYS = ("role", "content", "tool_calls", "tool_call_id", "reasoning_content", "name")

# Cap the stored rolled-back-generation text (debug/dashboard only) so a runaway
# 24k-token think doesn't bloat the dump; keep head + tail with an elision marker.
_MAX_TAIL_CHARS = 60000

# Injected to force-close a dangling <think> on a length-capped turn (s1-style
# budget forcing). Injected ids are loss_mask=0: the policy is never trained to
# emit them, only to continue from the closed-think state.
_THINK_CLOSE_TEXT = "\n</think>\n\n"


@dataclass
class Chain:
    """One contiguous conversation = one training Sample.

    ``prompt_len`` is the masked-out prefix (first render); ``versions`` holds one
    weight_version per generated turn in this chain.
    """

    tokens: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    versions: list[str] = field(default_factory=list)
    prompt_len: int = 0
    seen_msgs: int = 0
    msg_hashes: list[str] = field(default_factory=list)
    # One (start, end) token span per SURVIVING turn (query() call), in absolute
    # chain-token indices: every generated segment plus injected think-closure
    # ids, excluding the observation delta. Rolled-back turns leave no span.
    turn_spans: list[tuple[int, int]] = field(default_factory=list)
    # Wall-clock (epoch sec) at each turn's generation END — i.e. just before its
    # tool call executes — so mid-episode judge submissions (server ts) can be
    # aligned to the turn that produced them (turn-level reward shaping).
    turn_ts: list[float] = field(default_factory=list)
    # Decoded text of a rolled-back terminal generation (a turn with no tool call,
    # dropped from the trained token stream). Kept for the dashboard so a runaway /
    # length-truncated <think> is visible instead of lost. {"text","tokens","finish"}.
    truncated_tail: dict | None = None
    # Decoded first-turn model input (system + tools + instance + the injected
    # generation prompt) — convert.py ships torch only and can't re-decode, so record
    # it here for the dashboard's "exact prompt" view.
    full_prompt: str | None = None

    @property
    def has_response(self) -> bool:
        return len(self.tokens) > self.prompt_len


def _stable_hash(msg: dict) -> str:
    return json.dumps({k: msg.get(k) for k in _RENDER_KEYS}, sort_keys=True, ensure_ascii=False, default=str)


class RecordingModel:
    def __init__(
        self,
        tokenizer,
        sampling_params: dict,
        router_url: str,
        session_id: str,
        *,
        tool_parser: str | None,
        reasoning_parser: str | None,
        observation_template: str = OBSERVATION_TEMPLATE,
        format_error_template: str = FORMAT_ERROR_TEMPLATE,
        tools: list[dict] | None = None,
        max_context_len: int = 0,
        query_timeout: int = 600,
        max_empty_turns: int = 3,
        close_think_on_length: bool = False,
        max_think_closures: int = 2,
        think_closure_budget: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.sampling_params = dict(sampling_params)
        self.url = f"{router_url.rstrip('/')}/generate"
        self.tool_parser = tool_parser
        self.reasoning_parser = reasoning_parser
        self.observation_template = observation_template
        self.format_error_template = format_error_template
        self.tools = tools if tools is not None else [BASH_TOOL]
        self.max_context_len = int(max_context_len or 0)
        self.query_timeout = query_timeout
        self.max_empty_turns = max_empty_turns
        # Forced think closure (agentic_close_think_on_length): on finish=length with
        # no tool call, keep the generation, close the dangling <think>, and continue
        # the SAME turn (budgeted) instead of rolling the turn back into a null.
        self.close_think_on_length = bool(close_think_on_length)
        self.max_think_closures = int(max_think_closures)
        self.think_closure_budget = int(think_closure_budget)
        # Consistent-hash routing → every turn hits the same worker (prefix cache).
        self.headers = {"Content-Type": "application/json", "X-SMG-Routing-Key": session_id}

        self.chains: list[Chain] = []
        self.aborted = False
        # Terminal reason this model stopped the episode (the LimitsExceeded status:
        # Aborted / ContextLengthExceeded / NoProgress). Read by generate.py to label
        # a no-usable-turn episode with the real cause instead of a blanket fallback.
        self.exit_status: str | None = None
        self.gen_time = 0.0
        self.cached_tokens = 0
        self.input_tokens = 0
        self.n_format_errors = 0
        self.n_think_closures = 0
        self._empty_streak = 0

    @property
    def cur(self) -> Chain:
        return self.chains[-1]

    # mini-swe Model protocol -----------------------------------------------
    def query(self, messages: list[dict], **kwargs) -> dict:
        c = self.chains[-1] if self.chains else None
        hashes = [_stable_hash(m) for m in messages]
        is_append = c is not None and c.seen_msgs and len(hashes) >= c.seen_msgs and hashes[: c.seen_msgs] == c.msg_hashes
        if not is_append:
            c = Chain()
            self.chains.append(c)

        if c.seen_msgs == 0:
            self._extend(self._render_initial(messages), 0)
            c.prompt_len = len(c.tokens)
            c.full_prompt = self.tokenizer.decode(c.tokens, skip_special_tokens=False)
        else:
            self._extend(self._render_continuation(messages[c.seen_msgs :]), 0)
        c.seen_msgs = len(messages)
        c.msg_hashes = hashes

        turn_start = len(c.tokens)
        turn_versions = 0
        closures_left = self.max_think_closures if self.close_think_on_length else 0
        # Qwen3-style templates END the generation prompt with a pre-seeded '<think>\n',
        # so the model's own text never contains the opener — detect it from the
        # rendered prompt tail or closure injection can never fire (observed: run
        # 20260710-080841, 0 injections across 31 closure attempts).
        seeded_think = self._tail_seeds_think(c.tokens[max(0, turn_start - 8) : turn_start])
        while True:
            out_ids, logps, finish, version = self._generate(
                c.tokens, max_new_tokens=self.think_closure_budget if turn_versions else None
            )
            self._append_generated(out_ids, logps, version)
            turn_versions += 1
            if finish == "abort":
                self.aborted = True
                self.exit_status = "Aborted"
                raise LimitsExceeded(self._exit_message("Aborted", "generation aborted by weight update"))

            # Parse the WHOLE turn so far (all generated segments + any injected
            # closure ids), not just the last segment: a tool call may span the
            # forced-closure boundary.
            raw = self.tokenizer.decode(c.tokens[turn_start:], skip_special_tokens=False) if len(c.tokens) > turn_start else ""
            parsed = parse_model_output(
                raw,
                tools_schema=self.tools,
                tool_parser_name=self.tool_parser,
                reasoning_parser_name=self.reasoning_parser,
            )
            try:
                actions, tool_calls = self._actions_from_tool_uses(parsed.tool_uses)
                break
            except FormatError:
                if finish == "length" and closures_left > 0 and out_ids:
                    # Forced think closure (s1-style budget forcing): keep the truncated
                    # generation — the model's own on-policy tokens — close the dangling
                    # <think> with injected loss_mask=0 ids, and continue the same turn
                    # under a small budget so the model can act on what it worked out,
                    # instead of the turn being rolled back into a reward-0 null.
                    closures_left -= 1
                    self.n_think_closures += 1
                    if self._has_unclosed_think(raw, seeded=seeded_think):
                        self._extend(self._encode_text(_THINK_CLOSE_TEXT), 0)
                    continue
                # Record the about-to-be-discarded generation (esp. a length-truncated
                # runaway think) so the dashboard can show it; then drop the whole turn
                # (every segment + injected closure ids) from training.
                c.truncated_tail = {"text": self._clip_tail(raw), "tokens": len(c.tokens) - turn_start, "finish": finish}
                self._rollback_turn(turn_start, turn_versions)
                self.n_format_errors += 1
                self._empty_streak += 1
                if finish == "length" or self._empty_streak >= self.max_empty_turns:
                    status = "ContextLengthExceeded" if finish == "length" else "NoProgress"
                    self.exit_status = status
                    raise LimitsExceeded(self._exit_message(status, f"no tool call (finish={finish})"))
                raise

        self._empty_streak = 0
        c.turn_spans.append((turn_start, len(c.tokens)))
        c.turn_ts.append(time.time())
        message: dict = {"role": "assistant", "content": parsed.text or None, "extra": {"actions": actions, "cost": 0.0}}
        if parsed.reasoning:
            message["reasoning_content"] = parsed.reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def format_message(self, **kwargs) -> dict:
        return dict(kwargs)

    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None):
        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=self.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs) -> dict:
        return {}

    def serialize(self) -> dict:
        return {}

    # internals -------------------------------------------------------------
    def _actions_from_tool_uses(self, tool_uses: list[dict]) -> tuple[list[dict], list[dict]]:
        if not tool_uses:
            raise self._format_error("No tool calls found in the response. Every response MUST include a tool call.")
        actions, tool_calls = [], []
        for tu in tool_uses:
            name, args = tu.get("name"), tu.get("input") or {}
            err = ""
            if name != "bash":
                err += f"Unknown tool '{name}'."
            if not isinstance(args, dict) or "command" not in args:
                err += "Missing 'command' argument in bash tool call."
            if err:
                raise self._format_error(err.strip())
            call_id = f"call_{secrets.token_hex(12)}"
            actions.append({"command": args["command"], "tool_call_id": call_id})
            tool_calls.append(
                {"id": call_id, "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": args["command"]})}}
            )
        return actions, tool_calls

    def _format_error(self, error: str) -> FormatError:
        content = Template(self.format_error_template, undefined=StrictUndefined).render(error=error, actions=[])
        return FormatError({"role": "user", "content": content, "extra": {"interrupt_type": "FormatError"}})

    @staticmethod
    def _exit_message(status: str, content: str) -> dict:
        return {"role": "exit", "content": content, "extra": {"exit_status": status, "submission": ""}}

    def _template_ids(self, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
        enc = self.tokenizer.apply_chat_template(
            messages, tools=self.tools, tokenize=True, add_generation_prompt=add_generation_prompt
        )
        ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
        ids = list(ids)
        if ids and isinstance(ids[0], list):  # transformers>=5 may return [[...]]
            ids = ids[0]
        return ids

    @staticmethod
    def _clean(messages: list[dict]) -> list[dict]:
        return [{k: v for k, v in m.items() if k in _RENDER_KEYS and v is not None} for m in messages]

    def _render_initial(self, messages: list[dict]) -> list[int]:
        return self._template_ids(self._clean(messages), add_generation_prompt=True)

    def _render_continuation(self, new_messages: list[dict]) -> list[int]:
        """Token delta to append after the previous turn's raw output_ids: the
        observation message(s) added this turn plus the next generation prompt.

        Anchored on a sentinel user message so a prior assistant turn is never
        re-rendered. The model's raw output stops at ``<|im_end|>``; the template
        emits ``<|im_end|>\\n`` after a finished turn, so we keep the sentinel's
        trailing newline as the inter-turn separator.
        """
        obs = self._clean([m for m in new_messages if m.get("role") != "assistant"])
        if not obs:
            return []
        sentinel = [{"role": "user", "content": ""}]
        base = self._template_ids(sentinel, add_generation_prompt=False)
        full = self._template_ids(sentinel + obs, add_generation_prompt=True)
        if len(base) < 1 or full[: len(base)] != base:
            return self._template_ids(obs, add_generation_prompt=True)  # unexpected shape; best-effort
        return full[len(base) - 1 :]

    def _extend(self, ids: list[int], mask: int) -> None:
        c = self.cur
        c.tokens += ids
        c.loss_mask += [mask] * len(ids)
        c.logprobs += [0.0] * len(ids)

    def _append_generated(self, out_ids: list[int], logps: list[float], version: str | None) -> None:
        c = self.cur
        c.tokens += out_ids
        c.loss_mask += [1] * len(out_ids)
        c.logprobs += logps
        c.versions.append(version)

    def _rollback_turn(self, start: int, n_versions: int) -> None:
        """Drop everything the current turn appended: every generated segment plus
        any injected think-closure ids (which carry no version entry)."""
        c = self.cur
        del c.tokens[start:], c.loss_mask[start:], c.logprobs[start:]
        if n_versions:
            del c.versions[-n_versions:]

    def _encode_text(self, text: str) -> list[int]:
        try:
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:  # tokenizers without the kwarg (test fakes)
            return list(self.tokenizer.encode(text))

    def _tail_seeds_think(self, ids: list[int]) -> bool:
        """True when the rendered generation prompt ends inside a think block (the
        template pre-seeded the opener), so the turn is mid-think from token 0."""
        text = self.tokenizer.decode(ids, skip_special_tokens=False) if ids else ""
        return text.rfind("<think>") > text.rfind("</think>")

    @staticmethod
    def _has_unclosed_think(text: str, *, seeded: bool = False) -> bool:
        return text.count("<think>") + (1 if seeded else 0) > text.count("</think>")

    @staticmethod
    def _clip_tail(text: str) -> str:
        """Bound the stored rolled-back-generation text (debug/dashboard only)."""
        if len(text) <= _MAX_TAIL_CHARS:
            return text
        return f"{text[: _MAX_TAIL_CHARS - 10000]}\n\n…[{len(text) - _MAX_TAIL_CHARS} chars elided]…\n\n{text[-10000:]}"

    def _generate(self, input_ids: list[int], *, max_new_tokens: int | None = None) -> tuple[list[int], list[float], str, str | None]:
        sp = dict(self.sampling_params)
        if max_new_tokens is not None:
            sp["max_new_tokens"] = int(max_new_tokens)
        if self.max_context_len > 0:
            remaining = self.max_context_len - len(input_ids)
            if remaining <= 0:
                return [], [], "length", None
            cap = int(sp.get("max_new_tokens") or remaining)
            sp["max_new_tokens"] = min(cap, remaining)

        payload = {"input_ids": input_ids, "sampling_params": sp, "return_logprob": True}
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=self.headers)
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.query_timeout) as resp:
            data = json.loads(resp.read())
        self.gen_time += time.perf_counter() - t0

        meta = data.get("meta_info") or {}
        self.cached_tokens += meta.get("cached_tokens", 0)
        self.input_tokens += len(input_ids)
        lps = meta.get("output_token_logprobs") or []
        out_ids = [t[1] for t in lps]
        logps = [float(t[0]) for t in lps]
        finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
        return out_ids, logps, finish, meta.get("weight_version")
