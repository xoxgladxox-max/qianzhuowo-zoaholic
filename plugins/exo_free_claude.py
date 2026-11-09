"""Zoaholic channel for the ID-verified Claude lane of OpenCode Zen exo-free."""

import asyncio
import copy
import json
import logging
import os
from pathlib import Path
import re
import secrets

import httpx

FAMILY = "claude"
CHANNEL_ID = "exo-free-" + FAMILY
PLUGIN_NAME = "exo_free_" + FAMILY
PREFIXES = {"gpt": "resp_", "claude": "msg_"}
MAX_ATTEMPTS = 6
UPSTREAM_URL = "https://opencode.ai/zen/v1/chat/completions"
USER_AGENT = "opencode/latest/1.18.35/cli"
logger = logging.getLogger(PLUGIN_NAME)

PLUGIN_INFO = {
    "name": PLUGIN_NAME,
    "version": "1.0.0",
    "description": f"exo-free {FAMILY.upper()} route: sticky session, raw ID verification, fail closed",
    "author": "云影",
    "dependencies": [],
    "metadata": {"category": "channel", "tags": ["opencode", "exo-free", FAMILY]},
}
EXTENSIONS = ["channels:" + CHANNEL_ID]


class ZenError(Exception):
    def __init__(self, message, code="zen_route_error", status=502):
        super().__init__(message)
        self.code = code
        self.status = status

    def as_dict(self):
        return {"error": {"message": str(self), "type": "upstream_error", "code": self.code},
                "status_code": self.status}


def prepare_payload(payload):
    payload = copy.deepcopy(payload)
    payload["model"] = "exo-free"
    payload["stream"] = True
    payload["stream_options"] = {**(payload.get("stream_options") or {}), "include_usage": True}
    tools = payload.get("tools") or []
    existing = {tool.get("function", {}).get("name") for tool in tools}
    for name in ("shell", "read"):
        if name not in existing:
            tools.append({"type": "function", "function": {
                "name": name,
                "description": "Internal OpenCode free-tier compatibility marker; do not call.",
                "parameters": {"type": "object", "properties": {}},
            }})
    payload["tools"] = tools
    if payload.get("tool_choice") is None:
        payload["tool_choice"] = "auto"
    return payload


async def sse_chunks(response):
    data = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif line == "" and data:
            raw = "\n".join(data)
            data = []
            if raw == "[DONE]":
                yield None
                return
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ZenError("Zen returned malformed SSE JSON", "zen_invalid_sse") from exc
            if not isinstance(chunk, dict):
                raise ZenError("Zen returned a non-object SSE event", "zen_invalid_sse")
            yield chunk
    raise ZenError("Zen stream ended without [DONE]", "zen_incomplete_stream")


class StickyRoute:
    def __init__(self, state_path):
        self.state_path = Path(state_path)
        self.lock = asyncio.Lock()
        self.session = None
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            session = state.get("session")
            if state.get("family") != FAMILY or (session is not None and
                    not re.fullmatch(r"ses_[0-9a-f]{26}", session)):
                raise ValueError(f"Invalid route state: {self.state_path}")
            self.session = session

    def save_session(self, session):
        if self.session == session:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_name(self.state_path.name + "." + secrets.token_hex(6))
        try:
            with temp.open("x", encoding="utf-8") as output:
                os.chmod(temp, 0o600)
                json.dump({"family": FAMILY, "session": session}, output)
            os.replace(temp, self.state_path)
            self.session = session
        finally:
            temp.unlink(missing_ok=True)

    async def chunks(self, client, payload, model, timeout):
        payload = prepare_payload(payload)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            await self.lock.acquire()
            locked = True
            mismatch = False
            try:
                session = self.session or "ses_" + secrets.token_hex(13)
                headers = {
                    "Content-Type": "application/json", "User-Agent": USER_AGENT,
                    "x-opencode-project": "global", "x-opencode-client": "cli",
                    "x-opencode-session": session,
                    "x-session-affinity": session, "x-session-id": session,
                    "x-opencode-request": "msg_" + secrets.token_hex(13),
                }
                async with client.stream("POST", UPSTREAM_URL, headers=headers,
                                         json=payload, timeout=timeout) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode("utf-8", errors="replace")[:600]
                        raise ZenError(f"Zen HTTP {response.status_code}: {body}",
                                       "zen_upstream_http", response.status_code)
                    verified = False
                    finished = False
                    pending = []
                    pending_size = 0
                    async for chunk in sse_chunks(response):
                        if chunk is None:
                            if not verified or not finished:
                                raise ZenError("Zen stream has no verified ID or finish_reason",
                                               "zen_incomplete_stream")
                            yield None
                            return
                        if "error" in chunk:
                            raise ZenError(f"Zen SSE error: {str(chunk['error'])[:600]}",
                                           "zen_upstream_sse")
                        raw_id = chunk.get("id")
                        if raw_id:
                            actual = next((family for family, prefix in PREFIXES.items()
                                           if isinstance(raw_id, str) and raw_id.startswith(prefix)), None)
                            if actual is None:
                                raise ZenError(f"Unrecognized upstream ID: {raw_id}", "zen_unknown_id")
                            if actual != FAMILY:
                                if self.session == session:
                                    self.save_session(None)
                                logger.info("Rejected Zen route: wanted=%s actual=%s id=%s attempt=%s",
                                            FAMILY, actual, raw_id, attempt)
                                if verified:
                                    raise ZenError("Zen changed route after output began", "zen_route_changed")
                                mismatch = True
                                break
                            if not verified:
                                self.save_session(session)
                                verified = True
                                self.lock.release()
                                locked = False
                                logger.info("Accepted Zen route: family=%s id=%s attempt=%s",
                                            FAMILY, raw_id, attempt)
                                for buffered in pending:
                                    if "model" in buffered:
                                        buffered["model"] = model
                                    yield buffered
                                pending.clear()
                        if any(choice.get("finish_reason") is not None
                               for choice in chunk.get("choices", [])):
                            finished = True
                        if not verified:
                            pending_size += len(json.dumps(chunk))
                            if pending_size > 65536:
                                raise ZenError("Zen sent too much data without an ID", "zen_missing_id")
                            pending.append(chunk)
                            continue
                        if "model" in chunk:
                            chunk["model"] = model
                        yield chunk
                    if not mismatch:
                        raise ZenError("Zen stream ended prematurely", "zen_incomplete_stream")
            finally:
                if locked:
                    self.lock.release()
        raise ZenError(f"No {FAMILY.upper()} route after {MAX_ATTEMPTS} attempts; no output released",
                       "zen_route_exhausted")


ROUTE = StickyRoute(Path("/home/data") / (PLUGIN_NAME + "_session.json"))


async def request_adapter(request, engine, provider, api_key=None):
    from core.channels.openai_channel import get_gpt_payload
    _, _, payload = await get_gpt_payload(request, engine, provider, api_key=None)
    if request.max_tokens is not None:
        payload["max_tokens"] = payload.pop("max_completion_tokens", request.max_tokens)
    return UPSTREAM_URL, {"Content-Type": "application/json"}, prepare_payload(payload)


async def stream_adapter(client, url, headers, payload, model, timeout):
    try:
        async for chunk in ROUTE.chunks(client, payload, model, timeout):
            yield "data: [DONE]\n\n" if chunk is None else "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
    except (ZenError, httpx.HTTPError, OSError) as exc:
        error = exc if isinstance(exc, ZenError) else ZenError(str(exc))
        yield error.as_dict()


def merge_chunk(result, choices, chunk):
    for field in ("id", "created", "system_fingerprint", "usage"):
        if chunk.get(field) is not None:
            result[field] = chunk[field]
    for choice in chunk.get("choices", []):
        index = choice.get("index", 0)
        target = choices.setdefault(index, {"index": index, "message": {"role": "assistant", "content": ""},
                                            "finish_reason": None})
        message = target["message"]
        for field, value in choice.get("delta", {}).items():
            if field in ("content", "reasoning_content", "refusal"):
                if value is not None:
                    message[field] = (message.get(field) or "") + value
            elif field == "tool_calls":
                calls = message.setdefault("tool_calls", {})
                for fragment in value:
                    call = calls.setdefault(fragment.get("index", 0), {"function": {"name": "", "arguments": ""}})
                    for key in ("id", "type"):
                        if fragment.get(key):
                            call[key] = fragment[key]
                    for key in ("name", "arguments"):
                        call["function"][key] += fragment.get("function", {}).get(key) or ""
            else:
                message[field] = value
        if choice.get("finish_reason") is not None:
            target["finish_reason"] = choice["finish_reason"]
        if choice.get("logprobs"):
            logprobs = target.setdefault("logprobs", {})
            for key, values in choice["logprobs"].items():
                if isinstance(values, list):
                    logprobs.setdefault(key, []).extend(values)
                else:
                    logprobs[key] = values


async def response_adapter(client, url, headers, payload, model, timeout):
    result = {"object": "chat.completion", "model": model}
    choices = {}
    try:
        async for chunk in ROUTE.chunks(client, payload, model, timeout):
            if chunk is not None:
                merge_chunk(result, choices, chunk)
        for choice in choices.values():
            message = choice["message"]
            if "tool_calls" in message:
                message["tool_calls"] = [call for _, call in sorted(message["tool_calls"].items())]
                if not message["content"]:
                    message["content"] = None
        result["choices"] = [choice for _, choice in sorted(choices.items())]
        yield result
    except (ZenError, httpx.HTTPError, OSError) as exc:
        error = exc if isinstance(exc, ZenError) else ZenError(str(exc))
        yield error.as_dict()


class ChannelAdapter:
    id = CHANNEL_ID
    type_name = "zen-family"
    request_adapter = staticmethod(request_adapter)
    stream_adapter = staticmethod(stream_adapter)
    response_adapter = staticmethod(response_adapter)


def setup(manager):
    from core.channels.registry import register_channel
    manager.register_extension("channels", CHANNEL_ID, ChannelAdapter, priority=100,
                               metadata={"supported_features": ["chat", "stream", "vision", "tools"]},
                               plugin_name=PLUGIN_NAME)
    # The custom type keeps generic passthrough from bypassing ID verification.
    register_channel(id=CHANNEL_ID, type_name="zen-family", default_base_url=UPSTREAM_URL,
                     description=PLUGIN_INFO["description"], request_adapter=request_adapter,
                     stream_adapter=stream_adapter, response_adapter=response_adapter, overwrite=True)


def teardown(manager):
    from core.channels.registry import unregister_channel
    manager.unregister_extension("channels", CHANNEL_ID)
    unregister_channel(CHANNEL_ID)
