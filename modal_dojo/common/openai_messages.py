"""Serialize chat messages for the OpenAI Chat Completions wire format."""

from __future__ import annotations

import json


def _messages_to_openai(messages: list[dict]) -> list[dict]:
    """Serialize internal tool-call arguments without mutating caller messages."""
    wire_messages = []
    for message in messages:
        wire_message = dict(message)
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            wire_tool_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    wire_tool_calls.append(tool_call)
                    continue
                wire_tool_call = dict(tool_call)
                function = tool_call.get("function")
                if isinstance(function, dict):
                    wire_function = dict(function)
                    arguments = function.get("arguments")
                    if isinstance(arguments, dict):
                        wire_function["arguments"] = json.dumps(arguments)
                    wire_tool_call["function"] = wire_function
                wire_tool_calls.append(wire_tool_call)
            wire_message["tool_calls"] = wire_tool_calls
        wire_messages.append(wire_message)
    return wire_messages
