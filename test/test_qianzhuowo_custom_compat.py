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
