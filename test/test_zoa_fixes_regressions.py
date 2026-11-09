import json
from types import SimpleNamespace

import httpx
import pytest

from core.channels.gemini_channel import get_gemini_payload
from core.channels.openai_channel import fetch_gpt_response_stream, get_gpt_payload
from core.channels.openai_responses_channel import get_responses_payload
from core.channels.vertex_channel import get_vertex_gemini_payload
from core.dialects.gemini import (
    parse_gemini_request,
    render_gemini_response,
    render_gemini_stream,
)
from core.dialects.openai_responses import (
    convert_responses_input_to_messages,
    parse_responses_request,
)
from core.middleware import StatsMiddleware
from core.models import RequestModel
from core.stats import get_current_model_prices
import core.default_prices as default_prices


# ============================================================
# 1. OpenAI Channel (get_gpt_payload & fetch_gpt_response_stream)
# ============================================================


@pytest.mark.asyncio
async def test_openai_payload_preserves_assistant_text_tool_calls_extra_content_and_tool_fields():
    req = RequestModel(
        model="gpt-6-astra",
        messages=[
            {"role": "user", "content": "Read config"},
            {
                "role": "assistant",
                "content": "我将先读取配置文件",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"config.yaml"}'},
                        "extra_content": {"google": {"thoughtSignature": "sig_abc"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "read_file",
                "content": [{"type": "text", "text": "port: 8000"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "继续检查第二项"}],
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "check_port", "arguments": '{"port":8000}'},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {"name": "final_tool", "arguments": "{}"},
                    }
                ],
            },
        ],
    )
    provider = {"base_url": "https://api.openai.com/v1", "model": ["gpt-6-astra"]}
    _, _, payload = await get_gpt_payload(req, "openai", provider)

    msgs = payload["messages"]
    # Assistant with text + tool_calls retains both text and extra_content
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "我将先读取配置文件"
    assert msgs[1]["tool_calls"][0]["id"] == "call_1"
    assert msgs[1]["tool_calls"][0]["extra_content"] == {"google": {"thoughtSignature": "sig_abc"}}

    # Role == "tool" retains tool_call_id, name, and stringified content
    assert msgs[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "read_file",
        "content": "port: 8000",
    }

    # Assistant with list content + tool_calls retains both content and tool_calls
    assert msgs[3]["role"] == "assistant"
    assert msgs[3]["content"] == [{"type": "text", "text": "继续检查第二项"}]
    assert msgs[3]["tool_calls"][0]["id"] == "call_2"

    # Assistant with None content + tool_calls uses empty string "" instead of null
    assert msgs[4]["role"] == "assistant"
    assert msgs[4]["content"] == ""
    assert msgs[4]["tool_calls"][0]["id"] == "call_3"


@pytest.mark.asyncio
async def test_openai_stream_splits_merged_frame_and_flattens_reasoning_list():
    sse_lines = [
        'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":[{"type":"text","text":"step 1 "},{"text":"step 2"}],"content":"Let me call tool","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"my_fn","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\n',
        "data: [DONE]\n\n",
    ]

    class FakeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for line in sse_lines:
                yield line.encode("utf-8")

    async def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=FakeStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chunks = []
        async for chunk in fetch_gpt_response_stream(
            client,
            "https://api.openai.com/v1/chat/completions",
            {},
            {"model": "gpt-6-astra", "messages": []},
            "gpt-6-astra",
            10,
        ):
            chunks.append(chunk)

    data_events = [
        json.loads(c[6:].strip())
        for c in chunks
        if isinstance(c, str) and c.startswith("data: ") and c[6:].strip() != "[DONE]"
    ]
    # Should be split into 4 separate frames: reasoning_content -> content -> tool_calls -> finish_reason
    assert len(data_events) == 4
    assert data_events[0]["choices"][0]["delta"] == {
        "role": "assistant",
        "reasoning_content": "step 1 step 2",
    }
    assert data_events[0]["choices"][0]["finish_reason"] is None

    assert data_events[1]["choices"][0]["delta"] == {"content": "Let me call tool"}
    assert data_events[1]["choices"][0]["finish_reason"] is None

    assert data_events[2]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_1"
    assert data_events[2]["choices"][0]["finish_reason"] is None

    assert data_events[3]["choices"][0]["delta"] == {}
    assert data_events[3]["choices"][0]["finish_reason"] == "tool_calls"


# ============================================================
# 2. OpenAI Responses Dialect & Channel
# ============================================================


@pytest.mark.asyncio
async def test_responses_input_preserves_output_text_and_all_tool_histories():
    long_text = "x" * 5000
    native_body = {
        "model": "gpt-6-astra",
        "max_output_tokens": 4096,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Run two tools"}],
            },
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Thinking about tools..."}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will invoke two tools now."}],
            },
            # Parallel function calls
            {
                "type": "function_call",
                "call_id": "call_a",
                "name": "tool_a",
                "arguments": '{"q":1}',
            },
            {
                "type": "custom_tool_call",
                "call_id": "call_b",
                "name": "tool_b",
                "input": {"k": "v"},
            },
            # Outputs: object, array
            {
                "type": "function_call_output",
                "call_id": "call_a",
                "output": {"status": "ok", "count": 2},
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call_b",
                "output": [1, 2, 3],
            },
            # Turn 2: standalone function call without preceding assistant text, empty & long output, plus Chat tool message mix
            {
                "type": "function_call",
                "call_id": "call_c",
                "name": "tool_c",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_c",
                "output": "",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_d",
                        "type": "function",
                        "function": {"name": "tool_d", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_d",
                "name": "tool_d",
                "content": long_text,
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "All tools completed."}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Summarize"}],
            },
        ],
    }

    req = await parse_responses_request(native_body, {}, {})
    assert req.max_tokens == 4096

    provider = {"base_url": "https://api.openai.com/v1", "model": ["gpt-6-astra"]}
    _, _, chat_payload = await get_gpt_payload(req, "openai", provider)
    msgs = chat_payload["messages"]

    # 0: user
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == [{"type": "text", "text": "Run two tools"}]

    # 1: assistant with output_text, reasoning_content, and 2 merged parallel tool_calls
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == [{"type": "text", "text": "I will invoke two tools now."}]
    assert msgs[1]["reasoning_content"] == "Thinking about tools..."
    assert [tc["id"] for tc in msgs[1]["tool_calls"]] == ["call_a", "call_b"]
    assert msgs[1]["tool_calls"][1]["function"] == {"name": "tool_b", "arguments": '{"k":"v"}'}

    # 2 & 3: tool outputs (dict and list serialized)
    assert msgs[2] == {"role": "tool", "tool_call_id": "call_a", "content": '{"status":"ok","count":2}'}
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_b", "content": "[1,2,3]"}

    # 4 & 5: standalone function_call and empty string output
    assert msgs[4]["role"] == "assistant"
    assert msgs[4]["content"] == ""
    assert msgs[4]["tool_calls"][0]["id"] == "call_c"
    assert msgs[5] == {"role": "tool", "tool_call_id": "call_c", "content": ""}

    # 6 & 7: mixed Chat format assistant tool_calls and tool message with long text
    assert msgs[6]["role"] == "assistant"
    assert msgs[6]["content"] == ""
    assert msgs[6]["tool_calls"][0]["id"] == "call_d"
    assert msgs[7] == {"role": "tool", "tool_call_id": "call_d", "name": "tool_d", "content": long_text}

    # 8: historical assistant output_text
    assert msgs[8]["role"] == "assistant"
    assert msgs[8]["content"] == [{"type": "text", "text": "All tools completed."}]

    # Also verify get_responses_payload round-trip preserves list content + tool_calls and tool list content
    _, _, resp_payload = await get_responses_payload(req, "openai-responses", provider)
    fc_items = [item for item in resp_payload["input"] if item.get("type") == "function_call"]
    fco_items = [item for item in resp_payload["input"] if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in fc_items] == ["call_a", "call_b", "call_c", "call_d"]
    assert [item["call_id"] for item in fco_items] == ["call_a", "call_b", "call_c", "call_d"]


# ============================================================
# 3. Gemini Dialect ({model:path}, pairing queue, thoughtSignature, media_ref)
# ============================================================


@pytest.mark.asyncio
async def test_gemini_dialect_pairing_reset_thought_signature_and_media_ref_strip():
    native_body = {
        "contents": [
            {"role": "user", "parts": [{"text": "Turn 1"}]},
            # Turn 1 model had an unclosed functionCall with the same name "search"
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "search", "args": {"q": "old_unclosed"}}},
                ],
            },
            {"role": "user", "parts": [{"text": "Turn 2"}]},
            # Turn 2 model calls "search" again with thoughtSignature
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {"name": "search", "args": {"q": "new_query"}},
                        "thoughtSignature": "sig_vertex_123",
                    }
                ],
            },
            # Turn 2 user returns functionResponse with dangling media_ref_0 and $ref
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "search",
                            "response": {
                                "result": "found 1 item",
                                "media_ref_0": {"$ref": "media_ref_0"},
                                "nested_ref": {"$ref": "other_ref"},
                            },
                        }
                    }
                ],
            },
        ]
    }

    req = await parse_gemini_request(
        native_body,
        {"model": "vendor/gemini-4-argon", "action": "streamGenerateContent"},
        {},
    )
    assert req.model == "vendor/gemini-4-argon"
    assert req.stream is True

    # Turn 2 assistant tool_call id has __thought__ encoded and matches Turn 2 tool response id (not Turn 1's call_0!)
    turn2_assistant = req.messages[3]
    turn2_tool = req.messages[4]
    expected_tc_id = "call_1__thought__sig_vertex_123"
    assert turn2_assistant.tool_calls[0].id == expected_tc_id
    assert turn2_tool.tool_call_id == expected_tc_id

    # Dangling media_ref_0 and $ref are stripped from tool content
    assert json.loads(turn2_tool.content) == {"result": "found 1 item"}

    # Outbound Gemini response & stream restore thoughtSignature even when extra_content was stripped and only id remains
    canonical_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": expected_tc_id,
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q":"new_query"}'},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    gem_resp = await render_gemini_response(canonical_resp, "vendor/gemini-4-argon")
    fc_part = gem_resp["candidates"][0]["content"]["parts"][0]
    assert fc_part["functionCall"] == {"name": "search", "args": {"q": "new_query"}}
    assert fc_part["thoughtSignature"] == "sig_vertex_123"

    sse_chunk = "data: " + json.dumps({
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "id": expected_tc_id,
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q":"new_query"}'},
                        }
                    ]
                }
            }
        ]
    }) + "\n\n"
    gem_sse = await render_gemini_stream(sse_chunk)
    sse_payload = json.loads(gem_sse[6:].strip())
    sse_part = sse_payload["candidates"][0]["content"]["parts"][0]
    assert sse_part["functionCall"] == {"name": "search", "args": {"q": "new_query"}}
    assert sse_part["thoughtSignature"] == "sig_vertex_123"


def test_gemini_slash_model_paths_matched_by_middleware():
    mw = StatsMiddleware(app=None, debug=False)
    assert mw._is_dialect_endpoint("/v1beta/models/google/gemini-2.5-pro") is True
    assert mw._is_dialect_endpoint("/v1/models/google/gemini-2.5-pro") is True
    assert mw._is_dialect_endpoint("/v1beta/models/google/gemini-2.5-pro:generateContent") is True
    assert mw._is_dialect_endpoint("/v1/models/google/gemini-2.5-pro:generateContent") is True
    assert mw._is_dialect_endpoint("/v1beta/models/google/gemini-2.5-pro:streamGenerateContent") is True
    assert mw._is_dialect_endpoint("/v1/models/google/gemini-2.5-pro:streamGenerateContent") is True


# ============================================================
# 4. Stats Price Matching (<渠道>-<上游>-<上下文>-<模型>)
# ============================================================


def test_get_current_model_prices_strips_upstream_and_context_tags(monkeypatch):
    app = SimpleNamespace(
        state=SimpleNamespace(
            config={
                "providers": [
                    {
                        "provider": "sap",
                        "model_prefix": "sap-",
                        "preferences": {
                            "model_price": {
                                "gpt-6-astra": "2.5,10,0.25",
                            }
                        },
                    },
                    {
                        "provider": "sumopod",
                        "model_prefix": "sumopod-",
                        "preferences": {},
                    },
                ],
                "preferences": {
                    "model_price": {
                        "claude-opus-5-5": "15,75,1.5",
                        "gemini-4-argon": "1.25,5,0.3",
                    }
                },
            }
        )
    )

    # 1. Provider model_prefix + <上游>-<上下文>- stripped to match provider model_price
    assert get_current_model_prices(app, "sap-az-400k-gpt-6-astra", provider_name="sap") == (2.5, 10.0, 0.25)

    # 2. Provider model_prefix + <上游>-<上下文>- stripped to match global model_price
    assert get_current_model_prices(app, "sumopod-ant-200k-claude-opus-5-5", provider_name="sumopod") == (15.0, 75.0, 1.5)

    # 3. Without provider_name: <渠道>-<上游>-<上下文>- or <上游>-<上下文>- stripped to match global model_price
    assert get_current_model_prices(app, "meganova-vx-1m-gemini-4-argon") == (1.25, 5.0, 0.3)
    assert get_current_model_prices(app, "ais-1.5m-gemini-4-argon") == (1.25, 5.0, 0.3)

    # 4. External default_prices fallback with <上游>-<上下文>- tag
    monkeypatch.setattr(
        default_prices,
        "_price_cache",
        {"gpt-4o": {"id": "gpt-4o", "input": 2.5, "output": 10.0, "input_cached": 1.25}},
    )
    assert get_current_model_prices(app, "oai-128k-gpt-4o") == (2.5, 10.0, 1.25)
    assert get_current_model_prices(app, "custom-4t-128k-gpt-4o") == (2.5, 10.0, 1.25)


@pytest.mark.asyncio
async def test_network_protocol_and_proxy_enhancements(monkeypatch):
    """
    7. 客户端网络协议增强：
       - get_proxy 支持 socks5 / socks5h(rdns=True) / socks4 / socks4a(rdns=True) / http，且不污染默认字典
       - ClientManager 对主流 HTTPS 上游默认开启 HTTP/2 与 keepalive_expiry=120s
    """
    from core.client_manager import ClientManager
    from core.utils import get_proxy

    # 1) get_proxy 不再有可变默认参数污染
    cfg_none = get_proxy(None)
    assert cfg_none == {}
    cfg_http = get_proxy("http://127.0.0.1:7890")
    assert cfg_http == {"proxy": "http://127.0.0.1:7890"}
    assert get_proxy(None) == {}

    # 2) socks5 vs socks5h (rdns=False vs rdns=True) & socks4a (rdns=True)
    cfg_s5 = get_proxy("socks5://127.0.0.1:1080")
    cfg_s5h = get_proxy("socks5h://127.0.0.1:1080")
    cfg_s4a = get_proxy("socks4a://127.0.0.1:1080")
    assert cfg_s5["transport"]._pool._rdns is False
    assert cfg_s5h["transport"]._pool._rdns is True
    assert cfg_s4a["transport"]._pool._rdns is True
    await cfg_s5["transport"].aclose()
    await cfg_s5h["transport"].aclose()
    await cfg_s4a["transport"].aclose()

    # 3) ClientManager HTTP/2 默认启用与 proxy client_key 隔离
    cm = ClientManager()
    try:
        async with cm.get_client("https://api.openai.com/v1") as c_oai:
            assert c_oai._transport._pool._http2 is True
        async with cm.get_client("https://generativelanguage.googleapis.com/v1beta") as c_gem:
            assert c_gem._transport._pool._http2 is True
    finally:
        await cm.close()

