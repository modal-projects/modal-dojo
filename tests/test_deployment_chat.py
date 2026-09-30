from modal_dojo.common import deployment as deployment_module
from modal_dojo.common.deployment import CustomDeployment


def test_chat_serializes_tool_arguments_without_mutating_messages(monkeypatch) -> None:
    structured_message = {
        "content": "",
        "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}],
    }

    class _Response:
        status_code = 200

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {"choices": [{"message": structured_message}]}

    request_body = {}

    def post(url, *, json, timeout, headers):
        request_body.update(json)
        return _Response()

    import requests

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(deployment_module, "_modal_proxy_auth_headers", lambda: {})
    deployment = CustomDeployment.model_construct(
        deployment_id="test",
        served_model_name="test-model",
        url="https://example.test",
    )
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"key": "value"},
                    },
                }
            ],
        }
    ]

    response = deployment.chat(messages)

    assert response == structured_message
    assert request_body["model"] == "test-model"
    assert request_body["messages"][0]["tool_calls"][0]["function"]["arguments"] == (
        '{"key": "value"}'
    )
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"key": "value"}


def test_generate_accepts_caller_supplied_messages(monkeypatch) -> None:
    deployment = CustomDeployment.model_construct(
        deployment_id="test",
        served_model_name="test-model",
        url="https://example.test",
    )
    supplied_messages = [
        {"role": "system", "content": "Write Python."},
        {"role": "user", "content": "Return hello world."},
    ]
    captured = {}

    def chat(self, messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return {"content": "print('hello world')"}

    monkeypatch.setattr(CustomDeployment, "chat", chat)

    response = deployment.generate(
        "unused fallback",
        messages=supplied_messages,
        temperature=0,
    )

    assert response == "print('hello world')"
    assert captured == {
        "messages": supplied_messages,
        "kwargs": {"temperature": 0},
    }
