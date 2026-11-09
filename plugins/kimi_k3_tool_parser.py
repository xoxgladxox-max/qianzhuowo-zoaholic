"""Kimi K3 non-stream XTML tool-call parser for Zoaholic.

The plugin is activated only when a channel includes ``kimi_k3_tool_parser``
in ``provider.preferences.enabled_plugins``.  It intentionally handles only
complete, non-stream OpenAI-compatible responses.  Kimi K3 XTML tool blocks
are converted into standard ``message.tool_calls`` entries.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.log_config import logger
from core.plugins import register_response_interceptor, unregister_response_interceptor


PLUGIN_INFO = {
    "name": "kimi_k3_tool_parser",
    "version": "1.1.0",
    "description": "将 Kimi K3 非流式 XTML 工具调用转换为标准 OpenAI tool_calls",
    "author": "Local",
    "dependencies": [],
    "metadata": {
        "category": "interceptors",
        "tags": ["kimi", "k3", "xtml", "tools", "openai"],
        "params_hint": "无需参数；在需要转换的渠道中选择启用。仅处理非流式完整响应；流式客户端可搭配渠道 stream_mode=force_non_stream。",
    },
}

EXTENSIONS = ["interceptors:kimi_k3_tool_parser_response"]


_O = r"<\|open\|>"
_C = r"<\|close\|>"
_S = r"<\|sep\|>"

_TOOLS_OPEN_RE = re.compile(_O + r"\s*tools\s*" + _S)
_TOOLS_CLOSE_RE = re.compile(_C + r"\s*tools\s*" + _S)
_RESPONSE_OPEN_RE = re.compile(_O + r"\s*response\s*" + _S)
_RESPONSE_CLOSE_RE = re.compile(_C + r"\s*response\s*" + _S)
_RESPONSE_RE = re.compile(
    _O + r"\s*response\s*" + _S + r"(?P<body>.*?)" + _C + r"\s*response\s*" + _S,
    re.DOTALL,
)
_MESSAGE_CLOSE_RE = re.compile(_C + r"\s*message\s*" + _S)
_END_OF_MESSAGE_RE = re.compile(r"<\|end_of_msg\|>")
_CALL_RE = re.compile(
    _O
    + r"\s*call\s+(?P<attrs>.*?)"
    + _S
    + r"(?P<body>.*?)"
    + _C
    + r"\s*call\s*"
    + _S,
    re.DOTALL,
)
_ARGUMENT_RE = re.compile(
    _O
    + r"\s*argument\s+(?P<attrs>.*?)"
    + _S
    + r"(?P<value>.*?)"
    + _C
    + r"\s*argument\s*"
    + _S,
    re.DOTALL,
)
_ATTRIBUTE_RE = re.compile(r'(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*"(?P<value>[^"]*)"')
_ANY_XTML_TOKEN_RE = re.compile(r"<\|(?:open|close|sep)\|>")


class KimiK3ParseError(ValueError):
    """Raised when a K3 XTML response is incomplete or malformed."""


@dataclass(frozen=True)
class ParsedXTML:
    content: Optional[str]
    tool_calls: List[Dict[str, Any]]
    changed: bool


def _decode_attribute(value: str) -> str:
    # K3 encodes '&' before quotes, so decode quotes before ampersands.
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _parse_attributes(raw: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    pos = 0
    for match in _ATTRIBUTE_RE.finditer(raw):
        if raw[pos : match.start()].strip():
            raise KimiK3ParseError("invalid XTML attributes")
        key = match.group("key")
        if key in attrs:
            raise KimiK3ParseError("duplicate XTML attribute")
        attrs[key] = _decode_attribute(match.group("value"))
        pos = match.end()
    if raw[pos:].strip():
        raise KimiK3ParseError("invalid XTML attributes")
    return attrs


def _only_whitespace_between_matches(text: str, matches: List[re.Match[str]]) -> bool:
    pos = 0
    for match in matches:
        if text[pos : match.start()].strip():
            return False
        pos = match.end()
    return not text[pos:].strip()


def _tool_call_id(ordinal: int, tool_name: str, attrs: str, body: str) -> str:
    digest = hashlib.sha256(
        f"{ordinal}\0{tool_name}\0{attrs}\0{body}".encode("utf-8")
    ).hexdigest()[:12]
    return f"call_kimi_k3_{ordinal}_{digest}"


def _decode_call(match: re.Match[str], ordinal: int) -> Dict[str, Any]:
    call_attrs = _parse_attributes(match.group("attrs"))
    tool_name = call_attrs.get("tool", "")
    if not tool_name or _ANY_XTML_TOKEN_RE.search(tool_name):
        raise KimiK3ParseError("missing or invalid tool name")

    body = match.group("body")
    argument_matches = list(_ARGUMENT_RE.finditer(body))
    if not _only_whitespace_between_matches(body, argument_matches):
        raise KimiK3ParseError("malformed argument block")

    arguments: Dict[str, Any] = {}
    for argument_match in argument_matches:
        argument_attrs = _parse_attributes(argument_match.group("attrs"))
        key = argument_attrs.get("key", "")
        if not key or _ANY_XTML_TOKEN_RE.search(key):
            raise KimiK3ParseError("missing or invalid argument key")
        if key in arguments:
            raise KimiK3ParseError("duplicate argument key")

        argument_type = argument_attrs.get("type", "string").lower()
        raw_value = argument_match.group("value")
        if argument_type == "string":
            value: Any = raw_value
        else:
            try:
                value = json.loads(raw_value)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise KimiK3ParseError("invalid typed argument value") from exc
        arguments[key] = value

    return {
        "id": _tool_call_id(ordinal, tool_name, match.group("attrs"), body),
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        },
    }


def _clean_trailing_protocol_markers(text: str) -> str:
    cleaned = _MESSAGE_CLOSE_RE.sub("", text)
    cleaned = _END_OF_MESSAGE_RE.sub("", cleaned)
    return cleaned


def _extract_response_content(text: str) -> Optional[str]:
    response_matches = list(_RESPONSE_RE.finditer(text))
    if len(response_matches) > 1:
        raise KimiK3ParseError("multiple response channels")

    if response_matches:
        match = response_matches[0]
        outside = text[: match.start()] + text[match.end() :]
        outside = _clean_trailing_protocol_markers(outside)
        if outside.strip() or _ANY_XTML_TOKEN_RE.search(outside):
            raise KimiK3ParseError("unexpected data outside response channel")
        content = match.group("body")
    else:
        open_match = _RESPONSE_OPEN_RE.search(text)
        if open_match is not None:
            raise KimiK3ParseError("incomplete response channel")

        # The response-open marker may have been consumed as the generation
        # prefix, leaving only body + response-close in the model output.
        close_matches = list(_RESPONSE_CLOSE_RE.finditer(text))
        if len(close_matches) > 1:
            raise KimiK3ParseError("multiple response closing markers")
        if close_matches:
            close_match = close_matches[0]
            content = text[: close_match.start()]
            remainder = text[close_match.end() :]
        else:
            content = text
            remainder = ""

        remainder = _clean_trailing_protocol_markers(remainder)
        if remainder.strip() or _ANY_XTML_TOKEN_RE.search(remainder):
            raise KimiK3ParseError("unexpected data after response content")

    content = _clean_trailing_protocol_markers(content)
    if _ANY_XTML_TOKEN_RE.search(content):
        raise KimiK3ParseError("unparsed XTML marker in response content")
    return content or None


def parse_kimi_k3_xtml(model_output: str) -> Optional[ParsedXTML]:
    """Parse one complete K3 output, returning ``None`` for ordinary text."""
    if not isinstance(model_output, str):
        return None

    tools_open = _TOOLS_OPEN_RE.search(model_output)
    has_xtml = _ANY_XTML_TOKEN_RE.search(model_output) is not None
    if tools_open is None:
        if not has_xtml:
            return None
        content = _extract_response_content(model_output)
        return ParsedXTML(content=content, tool_calls=[], changed=True)

    tools_close = _TOOLS_CLOSE_RE.search(model_output, tools_open.end())
    if tools_close is None:
        raise KimiK3ParseError("incomplete tools channel")

    before = model_output[: tools_open.start()]
    section = model_output[tools_open.end() : tools_close.start()]
    after = _clean_trailing_protocol_markers(model_output[tools_close.end() :])
    if after.strip() or _ANY_XTML_TOKEN_RE.search(after):
        raise KimiK3ParseError("unexpected data after tools channel")

    call_matches = list(_CALL_RE.finditer(section))
    if not call_matches:
        raise KimiK3ParseError("tools channel has no complete calls")
    if not _only_whitespace_between_matches(section, call_matches):
        raise KimiK3ParseError("malformed call block")

    tool_calls = [
        _decode_call(call_match, ordinal)
        for ordinal, call_match in enumerate(call_matches)
    ]
    content = _extract_response_content(before)
    return ParsedXTML(content=content, tool_calls=tool_calls, changed=True)


def transform_openai_response(response: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Transform K3 XTML inside an OpenAI-compatible non-stream response."""
    if not isinstance(response, dict):
        return response, False
    choices = response.get("choices")
    if not isinstance(choices, list):
        return response, False

    transformed = copy.deepcopy(response)
    changed = False
    for choice in transformed.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue

        parsed = parse_kimi_k3_xtml(content)
        if parsed is None or not parsed.changed:
            continue

        message["content"] = parsed.content
        existing_calls = message.get("tool_calls")
        if parsed.tool_calls and not (isinstance(existing_calls, list) and existing_calls):
            message["tool_calls"] = parsed.tool_calls
        if parsed.tool_calls or (isinstance(existing_calls, list) and existing_calls):
            choice["finish_reason"] = "tool_calls"
        changed = True

    return transformed if changed else response, changed


def _error_response() -> Dict[str, Any]:
    return {
        "error": "Kimi K3 tool response could not be parsed.",
        "status_code": 502,
        "details": "kimi_k3_tool_parser rejected malformed or incomplete XTML",
    }


async def kimi_k3_tool_parser_response_interceptor(
    response_chunk: Any,
    engine: str,
    model: str,
    is_stream: bool,
) -> Any:
    """Convert complete K3 XTML responses; stream chunks are left untouched."""
    if is_stream:
        return response_chunk

    input_was_json_text = False
    payload = response_chunk
    if isinstance(response_chunk, str):
        stripped = response_chunk.strip()
        if not stripped or stripped.startswith("data:"):
            return response_chunk
        try:
            payload = json.loads(stripped)
        except (json.JSONDecodeError, TypeError, ValueError):
            return response_chunk
        input_was_json_text = True

    if not isinstance(payload, dict):
        return response_chunk

    try:
        transformed, changed = transform_openai_response(payload)
    except KimiK3ParseError as exc:
        logger.warning(
            f"[kimi_k3_tool_parser] Rejected malformed non-stream XTML "
            f"for model={model}: {exc}"
        )
        error = _error_response()
        return json.dumps(error, ensure_ascii=False) if input_was_json_text else error
    except Exception as exc:
        # The framework's generic interceptor wrapper is fail-open. Catch any
        # unexpected parser failure here so raw XTML cannot fall through.
        logger.error(
            f"[kimi_k3_tool_parser] Internal parser failure "
            f"for model={model}: {type(exc).__name__}"
        )
        error = _error_response()
        return json.dumps(error, ensure_ascii=False) if input_was_json_text else error

    if not changed:
        return response_chunk

    tool_call_count = 0
    for choice in transformed.get("choices", []):
        message = choice.get("message") if isinstance(choice, dict) else None
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if isinstance(calls, list):
            tool_call_count += len(calls)
    logger.info(
        f"[kimi_k3_tool_parser] Converted non-stream XTML response "
        f"for model={model} tool_calls={tool_call_count}"
    )
    return (
        json.dumps(transformed, ensure_ascii=False, separators=(",", ":"))
        if input_was_json_text
        else transformed
    )


def setup(manager: Any) -> None:
    logger.info(f"[{PLUGIN_INFO['name']}] Initializing...")
    register_response_interceptor(
        interceptor_id="kimi_k3_tool_parser_response",
        callback=kimi_k3_tool_parser_response_interceptor,
        priority=50,
        plugin_name=PLUGIN_INFO["name"],
        metadata={
            "description": "Kimi K3 非流式 XTML → OpenAI tool_calls",
            "stage": "response_interceptors",
        },
    )


def teardown(manager: Any) -> None:
    unregister_response_interceptor("kimi_k3_tool_parser_response")
