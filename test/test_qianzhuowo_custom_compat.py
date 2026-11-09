import json

import pytest

from core.channels.gemini_channel import get_gemini_payload
from core.dialects.gemini import parse_gemini_request, render_gemini_stream
from core.models import RequestModel


@pytest.mark.asyncio
async def test_code_execution_parts_survive_native_to_canonical():
    req = await parse_gemini_request(
        {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"executableCode": {"code": "print('ok')"}},
                        {"codeExecutionResult": {"output": "ok"}},
                    ],
                }
            ]
        },
        {"model": "gemini-4-argon", "action": "generateContent"},
        {},
    )
    assert [item.text for item in req.messages[0].content] == ["print('ok')", "ok"]


@pytest.mark.asyncio
async def test_incomplete_stream_tool_arguments_use_display_stream():
    chunk = "data: " + json.dumps({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"command":"un'},
                }]
            }
        }]
    }) + "\n\n"
    rendered = await render_gemini_stream(chunk)
    payload = json.loads(rendered[6:].strip())
    part = payload["candidates"][0]["content"]["parts"][0]
    assert part["functionCall"] == {
        "name": "shell",
        "args": {"display_stream": '{"command":"un'},
    }


@pytest.mark.asyncio
async def test_third_party_gemini_response_name_and_request_roles_are_preserved():
    req = RequestModel(
        model="gemini-4-argon",
        messages=[
            {"role": "user", "content": "run"},
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "shell", "content": "done"},
        ],
    )
    provider = {
        "base_url": "https://third-party.example/v1beta",
        "model": ["gemini-4-argon"],
        "image": False,
    }
    _, _, payload = await get_gemini_payload(req, "gemini", provider)
    response_part = payload["contents"][-1]["parts"][0]["functionResponse"]
    assert response_part["name"] == "shell"
    assert response_part["response"]["name"] == "shell"
    assert response_part["response"]["result"] == "done"
    assert [message.role for message in req.messages] == ["user", "assistant", "tool"]


def test_handler_callable_accepts_keyword_is_defined_and_detects_dialect_id():
    from core.handler import _callable_accepts_keyword, process_request

    assert _callable_accepts_keyword(process_request, "dialect_id") is True


@pytest.mark.asyncio
async def test_claude_passthrough_system_prompt_merges_opus46_assistant_prefill():
    from core.channels.claude_channel import patch_passthrough_claude_payload

    payload = {
        "model": "claude-opus-4-6",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Prefill:"},
        ],
    }
    out = await patch_passthrough_claude_payload(
        payload,
        {"system_prompt": "Sys"},
        None,
        "claude",
        {"provider": "test"},
    )
    assert out["system"] == "Sys"
    assert len(out["messages"]) == 1
    assert out["messages"][0]["role"] == "user"
    assert "Prefill:" in out["messages"][0]["content"]


@pytest.mark.asyncio
async def test_stream_convert_assemble_stream_to_json_handles_refusal_cleanly():
    from core.stream_convert import assemble_stream_to_json

    async def gen():
        yield 'data: {"id":"1","model":"m","choices":[{"delta":{"role":"assistant","content":"ok"}}]}\n\n'
        yield "data: [DONE]\n\n"

    res = await assemble_stream_to_json(gen())
    assert res["choices"][0]["message"]["content"] == "ok"
    assert res["choices"][0]["message"]["refusal"] is None
