"""
Gemini 工具调用参数真流式插件 (gemini_stream_function_args)

解决问题：
1. Zoaholic 原生 core/channels/gemini_channel.py 只处理完整的 functionCall.name + functionCall.args，
   忽略了 Gemini / Vertex 的 partialArgs、willContinue 与 id，导致开启工具参数流式时首帧发 "null"、后续增量帧全部丢失。
2. 本插件在请求侧自动为带 tools 的流式请求注入 toolConfig.functionCallingConfig.streamFunctionCallArguments = True
   （对 /v1/chat/completions 默认启用；对 /v1beta 原生透传仅在检测到 emit_complete_response_* / dispatch_stream_payload_* 等防截断合成工具时启用），
   并在响应侧用增量 JSON 状态机实时将 partialArgs (如 $.content 的 stringValue 分片) 转换为 OpenAI delta.tool_calls[].function.arguments 流式帧。
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from core.channels import get_channel, register_channel
from core.json_utils import json_dumps_text, json_loads
from core.stream_utils import aiter_decoded_lines
from core.log_config import logger
from core.plugins import (
    register_request_interceptor,
    unregister_request_interceptor,
)
from core.response_context import mark_adapter_metrics_managed, mark_content_start, merge_usage
from core.response import check_response
from core.utils import end_of_line, generate_sse_response, safe_get
from core.usage import extract_cache_usage

PLUGIN_NAME = "gemini_stream_function_args"
_INTERCEPTOR_ID = "gemini_stream_function_args_request"

GEMINI_ENGINES = ("gemini", "firebaseVertex", "vertex-gemini", "gemini-cli", "antigravity")
ZOA_TUNNEL_TOOL_NAME = "emit_complete_response_zoa_tunnel"
SYNTHETIC_TOOL_PREFIXES = (
    "emit_complete_response_",
    "emit_test",
    "archive_record_commit_",
    "emit_dual_stage_output_",
    "dispatch_stream_payload_",
    "dispatch_stream_",
    "render_stage_script_",
    "commit_pipeline_output_",
)
TUNNEL_TOOL_NAMES = (ZOA_TUNNEL_TOOL_NAME,)

PLUGIN_INFO = {
    "name": PLUGIN_NAME,
    "version": "1.0.0",
    "description": "Gemini 工具调用参数真流式插件：注入 streamFunctionCallArguments 并将 partialArgs 实时转换为 OpenAI tool_calls arguments 增量流",
    "author": "Felin & molilili",
    "dependencies": [],
    "metadata": {
        "category": "interceptors",
        "tags": ["gemini", "streaming", "tool_calls", "partialArgs"],
    },
}

EXTENSIONS = [
    f"interceptors:{_INTERCEPTOR_ID}",
]

# 记录不支持 streamFunctionCallArguments 的 (provider_name, model)
_UNSUPPORTED_STREAM_ARGS: set[Tuple[str, str]] = set()
_ORIGINAL_STREAM_ADAPTERS: Dict[str, Any] = {}


class IncrementalToolCallState:
    """单个正在流式传输的 Gemini functionCall 状态机。"""

    def __init__(self, call_id: str, name: str, tc_index: int, thought_signature: Optional[str] = None):
        self.call_id = call_id
        self.name = name
        self.tc_index = tc_index
        self.thought_signature = thought_signature
        self.header_sent = False
        self.object_opened = False
        self.closed = False
        self.active_string_key: Optional[str] = None
        self.emitted_keys: List[str] = []
        self.fallback_tree: Dict[str, Any] = {}

    def feed_partial_args(self, partial_args: List[Dict[str, Any]]) -> str:
        """喂入一个 chunk 的 partialArgs 列表，返回本帧可立即输出的 JSON 增量字符串。"""
        out: List[str] = []
        for item in partial_args:
            if not isinstance(item, dict):
                continue
            path = str(item.get("jsonPath") or "").strip()
            if not path.startswith("$."):
                continue
            key = path[2:]
            # 简单顶层字段（无嵌套点号或数组下标）走零缓冲增量流式
            if "." not in key and "[" not in key and key:
                if "stringValue" in item:
                    s_val = item.get("stringValue")
                    if not isinstance(s_val, str):
                        s_val = str(s_val or "")
                    will_cont = bool(item.get("willContinue"))

                    # 若当前未打开该 string key，先打开它
                    if self.active_string_key != key:
                        if self.active_string_key is not None:
                            out.append('"')
                            self.active_string_key = None
                        if not self.object_opened:
                            out.append("{")
                            self.object_opened = True
                            prefix_comma = ""
                        else:
                            prefix_comma = "," if self.emitted_keys else ""
                        out.append(f'{prefix_comma}{json.dumps(key, ensure_ascii=False)}:"')
                        self.active_string_key = key
                        if key not in self.emitted_keys:
                            self.emitted_keys.append(key)

                    if s_val:
                        # 去掉 json.dumps 外层双引号，直接追加转义后的内容
                        escaped = json.dumps(s_val, ensure_ascii=False)[1:-1]
                        out.append(escaped)

                    # Google 结束当前字符串字段的标记：stringValue=="" 且无 willContinue: True，或显式 willContinue=False
                    if not s_val and not will_cont:
                        if self.active_string_key == key:
                            out.append('"')
                            self.active_string_key = None
                elif "numberValue" in item or "boolValue" in item or "nullValue" in item:
                    if self.active_string_key is not None:
                        out.append('"')
                        self.active_string_key = None
                    if not self.object_opened:
                        out.append("{")
                        self.object_opened = True
                        prefix_comma = ""
                    else:
                        prefix_comma = "," if self.emitted_keys else ""
                    if "numberValue" in item:
                        raw_v = json.dumps(item["numberValue"], ensure_ascii=False)
                    elif "boolValue" in item:
                        raw_v = "true" if item["boolValue"] else "false"
                    else:
                        raw_v = "null"
                    out.append(f"{prefix_comma}{json.dumps(key, ensure_ascii=False)}:{raw_v}")
                    if key not in self.emitted_keys:
                        self.emitted_keys.append(key)
                else:
                    # 仅有 {"jsonPath": "$.content"}（无值键），代表该字段流式结束
                    if self.active_string_key == key:
                        out.append('"')
                        self.active_string_key = None
            else:
                # 复杂路径记入 fallback_tree
                if "stringValue" in item:
                    self.fallback_tree[key] = self.fallback_tree.get(key, "") + str(item.get("stringValue") or "")
                elif "numberValue" in item:
                    self.fallback_tree[key] = item["numberValue"]
                elif "boolValue" in item:
                    self.fallback_tree[key] = item["boolValue"]

        return "".join(out)

    def finish(self) -> str:
        """闭合当前工具调用的 JSON 参数对象。"""
        if self.closed:
            return ""
        self.closed = True
        out: List[str] = []
        if self.active_string_key is not None:
            out.append('"')
            self.active_string_key = None
        if self.fallback_tree:
            if not self.object_opened:
                out.append("{")
                self.object_opened = True
            for k, v in self.fallback_tree.items():
                if k in self.emitted_keys:
                    continue
                sep = "," if self.emitted_keys else ""
                out.append(f"{sep}{json.dumps(k, ensure_ascii=False)}:{json.dumps(v, ensure_ascii=False)}")
                self.emitted_keys.append(k)
        if not self.object_opened:
            return "{}"
        out.append("}")
        return "".join(out)



def _is_tunnel_tool(name: Optional[str]) -> bool:
    if not name or not isinstance(name, str):
        return False
    return name in TUNNEL_TOOL_NAMES

def _extract_tunnel_partial_text(partial_args: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    for item in partial_args:
        if not isinstance(item, dict):
            continue
        if "stringValue" in item and item.get("stringValue"):
            out.append(str(item["stringValue"]))
    return "".join(out)

def _has_function_declarations(payload: Dict[str, Any]) -> Tuple[bool, bool]:
    """返回 (是否有函数声明, 是否含防截断合成工具前缀)。"""
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return False, False
    has_decl = False
    has_synthetic = False
    for t in tools:
        if not isinstance(t, dict):
            continue
        decls = t.get("functionDeclarations") or t.get("function_declarations")
        if isinstance(decls, list) and decls:
            has_decl = True
            for d in decls:
                if isinstance(d, dict):
                    fn_name = str(d.get("name") or "")
                    if fn_name.startswith(SYNTHETIC_TOOL_PREFIXES):
                        has_synthetic = True
    return has_decl, has_synthetic


async def gemini_stream_function_args_request_interceptor(
    request: Any,
    engine: str,
    provider: Dict[str, Any],
    api_key: Optional[str],
    url: str,
    headers: Dict[str, Any],
    payload: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """在发往 Gemini 上游前注入 streamFunctionCallArguments: True。"""
    try:
        _install_channel_wrappers()
        if engine not in (*GEMINI_ENGINES, "vertex-express") or not isinstance(payload, dict):
            return url, headers, payload

        is_stream = "streamGenerateContent" in str(url) or bool(getattr(request, "stream", False))
        if not is_stream:
            return url, headers, payload

        prov_name = str(provider.get("provider") or engine)
        model_name = str(getattr(request, "model", None) or "")
        url_lower = str(url or "").lower()
        if (
            (prov_name, model_name) in _UNSUPPORTED_STREAM_ARGS
            or "argon-ultra-web" in prov_name.lower()
            or ":8080" in url_lower
            or "generativelanguage.googleapis.com" in url_lower
        ):
            return url, headers, payload

        # 1) 严格检查当前渠道是否在 enabled_plugins 中启用了本插件
        prefs = provider.get("preferences") if isinstance(provider.get("preferences"), dict) else {}
        enabled_list = prefs.get("enabled_plugins") if isinstance(prefs.get("enabled_plugins"), list) else []
        is_plugin_enabled = any(
            isinstance(item, str) and item.split(":", 1)[0].strip() == PLUGIN_NAME
            for item in enabled_list
        )
        if not is_plugin_enabled:
            return url, headers, payload

        has_decl, has_synthetic = _has_function_declarations(payload)
        if not has_decl:
            return url, headers, payload

        # 2) streamFunctionCallArguments 的 partialArgs 仅对单字符串合成隧道工具 (has_synthetic) 安全；
        #    对于客户端原生复杂嵌套数组工具（如 edit.hunks[].oldContent），开启会导致二级网关整包缓冲或嵌套数组组装残缺。
        if not has_synthetic:
            return url, headers, payload

        # 判断是否为 /v1beta 原生透传请求（检查调用栈是否来自 passthrough.py）
        import sys
        is_passthrough = isinstance(request, dict) or not hasattr(request, "messages")
        if not is_passthrough:
            f = sys._getframe(1)
            while f is not None:
                if f.f_code.co_name == "process_passthrough_request" or "passthrough.py" in f.f_code.co_filename:
                    is_passthrough = True
                    break
                f = f.f_back
        if is_passthrough and not has_synthetic:
            # 原生 /v1beta 默认保留原样，除非匹配防截断合成工具前缀
            return url, headers, payload

        tc = payload.get("toolConfig") or payload.get("tool_config")
        if not isinstance(tc, dict):
            tc = {}
        fcc = tc.get("functionCallingConfig") or tc.get("function_calling_config")
        if not isinstance(fcc, dict):
            fcc = {}

        # 仅当末尾存在 model/system 预填（UUAU/UUAS）时才把 ANY 降级为 VALIDATED 防死锁；
        # 若末尾是正常 user（如 U->A->U 尾锁），保留 ANY 以 100% 强锁工具调用！
        contents_list = payload.get("contents") or []
        last_role = str((contents_list[-1] if contents_list and isinstance(contents_list[-1], dict) else {}).get("role") or "user").lower()
        has_tail_prefill = (
            last_role in ("model", "assistant", "system")
            or (
                len(contents_list) >= 2
                and last_role == "user"
                and any(isinstance(p, dict) and p.get("thought") is True for p in (contents_list[-1].get("parts") or []))
            )
        )
        if (has_synthetic and fcc.get("mode") in (None, "AUTO")) or (fcc.get("mode") == "ANY" and has_tail_prefill):
            fcc["mode"] = "VALIDATED"
            fcc.pop("allowedFunctionNames", None)
            fcc.pop("allowed_function_names", None)

        if "streamFunctionCallArguments" not in fcc and "stream_function_call_arguments" not in fcc:
            fcc["streamFunctionCallArguments"] = True

        tc.pop("function_calling_config", None)
        tc["functionCallingConfig"] = fcc
        payload["toolConfig"] = tc
        payload.pop("tool_config", None)
        logger.info(f"[{PLUGIN_NAME}] Injected mode={fcc.get('mode')}, streamFunctionCallArguments=True (passthrough={is_passthrough})")
    except Exception as e:
        logger.error(f"[{PLUGIN_NAME}] request interceptor error: {e}", exc_info=True)

    return url, headers, payload


async def _stream_gemini_with_partial_args(client, url, headers, payload, model, timeout):
    """支持 partialArgs 增量流式与原子 args 双模的 Gemini 流式响应处理器。"""
    from core.channels.gemini_channel import (
        _extract_gemini_block_message,
        _is_image_model,
        _normalize_gemini_http_error,
    )

    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await asyncio.to_thread(json_dumps_text, payload)

    async with client.stream("POST", url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_gemini_response_stream")
        if error_message:
            # 若因不支持 streamFunctionCallArguments 报 400，剥离后自动重试一次
            err_str = str(error_message)
            if "streamFunctionCallArguments" in err_str and isinstance(payload, dict):
                tc = payload.get("toolConfig")
                if isinstance(tc, dict) and isinstance(tc.get("functionCallingConfig"), dict):
                    tc["functionCallingConfig"].pop("streamFunctionCallArguments", None)
                    logger.warning(f"[{PLUGIN_NAME}] Upstream rejected streamFunctionCallArguments, retrying without it")
                    async for retry_chunk in _stream_gemini_with_partial_args(client, url, headers, payload, model, timeout):
                        yield retry_chunk
                    return
            yield _normalize_gemini_http_error(error_message)
            return

        mark_adapter_metrics_managed()
        promptTokenCount = 0
        candidatesTokenCount = 0
        totalTokenCount = 0
        cachedContentTokenCount = 0
        parts_json = ""

        has_content = False
        has_image = False
        has_function_call = False
        has_reasoning = False
        stream_finished_normally = False
        tc_index = 0
        active_call: Optional[IncrementalToolCallState] = None

        async for line in aiter_decoded_lines(response.aiter_bytes()):
            if not line:
                continue

            if line.startswith("data:"):
                parts_json = line[5:].strip()
                try:
                    response_json = json_loads(parts_json)
                except json.JSONDecodeError:
                    continue
            else:
                parts_json += line
                parts_json = parts_json.lstrip("[,")
                try:
                    response_json = json_loads(parts_json)
                except json.JSONDecodeError:
                    continue

            json_data = safe_get(response_json, "candidates", 0, "content", default=None)
            finishReason = safe_get(response_json, "candidates", 0, "finishReason", default=None)
            blockReason = safe_get(response_json, "promptFeedback", "blockReason", default=None)
            if not blockReason:
                blockReason = safe_get(response_json, "candidates", 0, "blockReason", default=None)

            usage_meta = safe_get(response_json, "usageMetadata", default={}) or {}
            p_tok = safe_get(usage_meta, "promptTokenCount", default=0)
            c_tok = safe_get(usage_meta, "candidatesTokenCount", default=0)
            t_tok = safe_get(usage_meta, "totalTokenCount", default=0)
            if p_tok:
                promptTokenCount = p_tok
            if c_tok:
                candidatesTokenCount = c_tok
            if t_tok:
                totalTokenCount = t_tok
            if totalTokenCount > 0 or promptTokenCount > 0 or candidatesTokenCount > 0:
                _cache_usage = extract_cache_usage(usage_meta)
                if _cache_usage["cached_tokens"]:
                    cachedContentTokenCount = _cache_usage["cached_tokens"]
                merge_usage(
                    prompt_tokens=promptTokenCount,
                    completion_tokens=candidatesTokenCount,
                    total_tokens=totalTokenCount,
                    cached_tokens=cachedContentTokenCount,
                    cache_creation_tokens=_cache_usage["cache_creation_tokens"],
                )

            parts_data = safe_get(json_data, "parts", default=[])
            for part in parts_data:
                if not isinstance(part, dict):
                    continue
                sig = part.get("thoughtSignature")

                # 1. 思考内容
                if part.get("thought") is True:
                    t_text = part.get("text", "")
                    if t_text:
                        has_reasoning = True
                        mark_content_start()
                        yield await generate_sse_response(
                            timestamp, model, reasoning_content=t_text, thought_signature=sig
                        )
                    continue

                # 2. 函数调用（支持 partialArgs 增量流式 + 传统完整 args）
                if "functionCall" in part and isinstance(part["functionCall"], dict):
                    fc = part["functionCall"]
                    fc_name = fc.get("name")
                    fc_id = fc.get("id")
                    fc_args = fc.get("args")
                    partial_args = fc.get("partialArgs")
                    will_continue = bool(fc.get("willContinue"))

                    # 情况 0：Zoaholic 网关内置透明工具隧道 (emit_complete_response_zoa_tunnel) -> 直接拆包为普通文本 delta.content
                    if _is_tunnel_tool(fc_name) or (active_call is not None and _is_tunnel_tool(active_call.name)):
                        if _is_tunnel_tool(fc_name):
                            active_call = IncrementalToolCallState("tunnel", fc_name, 0, sig)
                        if fc_args is not None and not will_continue and not partial_args:
                            c_val = ""
                            if isinstance(fc_args, dict):
                                c_val = str(fc_args.get("content") or "")
                            elif isinstance(fc_args, str):
                                try:
                                    _obj = json_loads(fc_args)
                                    c_val = str(_obj.get("content") or "")
                                except Exception:
                                    c_val = fc_args
                            if c_val:
                                has_content = True
                                mark_content_start()
                                yield await generate_sse_response(timestamp, model, content=c_val, thought_signature=sig)
                        elif isinstance(partial_args, list) and partial_args:
                            c_val = _extract_tunnel_partial_text(partial_args)
                            if c_val:
                                has_content = True
                                mark_content_start()
                                yield await generate_sse_response(timestamp, model, content=c_val, thought_signature=sig)
                        if not will_continue:
                            active_call.closed = True
                            active_call = None
                        continue

                    # 情况 A：新工具调用开启（带 name）
                    if fc_name:
                        # 若上一个流式工具调用未闭合，先闭合它
                        if active_call is not None and not active_call.closed:
                            tail_str = active_call.finish()
                            if tail_str:
                                yield await generate_sse_response(
                                    timestamp, model, content=None, tools_id=active_call.call_id,
                                    function_call_name=None, function_call_content=tail_str,
                                    tool_call_index=active_call.tc_index,
                                )
                            active_call = None

                        has_function_call = True
                        cur_idx = tc_index
                        tc_index += 1
                        call_id = f"call_{fc_id}" if fc_id else f"call_{uuid4().hex[:24]}"

                        # 先发工具调用头帧（含 name 和 thoughtSignature）
                        mark_content_start()
                        yield await generate_sse_response(
                            timestamp, model, content=None, tools_id=call_id,
                            function_call_name=fc_name, thought_signature=sig,
                            tool_call_index=cur_idx,
                        )

                        # 若是传统原子调用（直接带完整 args 且无 willContinue）
                        if fc_args is not None and not will_continue and not partial_args:
                            args_str = fc_args if isinstance(fc_args, str) else json_dumps_text(fc_args, ensure_ascii=False)
                            yield await generate_sse_response(
                                timestamp, model, content=None, tools_id=call_id,
                                function_call_name=None, function_call_content=args_str,
                                tool_call_index=cur_idx,
                            )
                        else:
                            # 进入增量流式模式
                            active_call = IncrementalToolCallState(call_id, fc_name, cur_idx, sig)
                            if isinstance(partial_args, list) and partial_args:
                                delta_str = active_call.feed_partial_args(partial_args)
                                if delta_str:
                                    yield await generate_sse_response(
                                        timestamp, model, content=None, tools_id=active_call.call_id,
                                        function_call_name=None, function_call_content=delta_str,
                                        tool_call_index=active_call.tc_index,
                                    )
                            if not will_continue:
                                tail_str = active_call.finish()
                                if tail_str:
                                    yield await generate_sse_response(
                                        timestamp, model, content=None, tools_id=active_call.call_id,
                                        function_call_name=None, function_call_content=tail_str,
                                        tool_call_index=active_call.tc_index,
                                    )
                                active_call = None

                    # 情况 B：增量后续帧（无 name，属于当前 active_call）
                    elif active_call is not None:
                        if isinstance(partial_args, list) and partial_args:
                            delta_str = active_call.feed_partial_args(partial_args)
                            if delta_str:
                                mark_content_start()
                                yield await generate_sse_response(
                                    timestamp, model, content=None, tools_id=active_call.call_id,
                                    function_call_name=None, function_call_content=delta_str,
                                    tool_call_index=active_call.tc_index,
                                )
                        if not will_continue:
                            tail_str = active_call.finish()
                            if tail_str:
                                mark_content_start()
                                yield await generate_sse_response(
                                    timestamp, model, content=None, tools_id=active_call.call_id,
                                    function_call_name=None, function_call_content=tail_str,
                                    tool_call_index=active_call.tc_index,
                                )
                            active_call = None
                    continue

                # 3. 普通文本
                if "text" in part and not part.get("thought"):
                    c_text = part.get("text", "")
                    if c_text:
                        has_content = True
                        mark_content_start()
                        yield await generate_sse_response(
                            timestamp, model, content=c_text, thought_signature=sig
                        )

                # 4. 内联图片
                if "inlineData" in part:
                    b64_json = safe_get(part, "inlineData", "data", default="")
                    if b64_json and _is_image_model(model):
                        has_image = True
                        mark_content_start()
                        yield ": streaming inline image\n\n"
                        image_content_item = [{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_json}"}
                        }]
                        yield await generate_sse_response(
                            timestamp, model, content=image_content_item, thought_signature=sig
                        )

            if parts_json == "[]" or (blockReason and blockReason != "STOP"):
                msg = _extract_gemini_block_message(response_json) or (blockReason or "Empty Response")
                mark_content_start()
                yield await generate_sse_response(
                    timestamp, model, content=f"\n[Gemini Safety Block] {msg} (Reason: {blockReason or 'SAFETY'})\n", stop="content_filter"
                )
                err_payload = {
                    "error": {
                        "message": f"This request was blocked by Gemini safety filters: {msg} (Reason: {blockReason or 'SAFETY'})",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "content_filter",
                    }
                }
                yield f"data: {json_dumps_text(err_payload, ensure_ascii=False)}\n\n"
                return
            elif finishReason and finishReason not in ["STOP", "MAX_TOKENS"]:
                mark_content_start()
                yield await generate_sse_response(
                    timestamp, model, content=f"\n[Gemini Safety Block] Restricted generation due to {finishReason}\n", stop="content_filter"
                )
                err_payload = {
                    "error": {
                        "message": f"Gemini generation stopped by safety filter. Finish Reason: {finishReason}",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "content_filter",
                    }
                }
                yield f"data: {json_dumps_text(err_payload, ensure_ascii=False)}\n\n"
                return
            elif finishReason:
                if active_call is not None and not active_call.closed:
                    tail_str = active_call.finish()
                    if tail_str:
                        yield await generate_sse_response(
                            timestamp, model, content=None, tools_id=active_call.call_id,
                            function_call_name=None, function_call_content=tail_str,
                            tool_call_index=active_call.tc_index,
                        )
                    active_call = None

                stream_finished_normally = True
                if has_content or has_reasoning or has_function_call or has_image:
                    stop_reason = "tool_calls" if has_function_call and finishReason == "STOP" else "stop"
                    yield await generate_sse_response(timestamp, model, stop=stop_reason)
                break

            parts_json = ""

        if active_call is not None and not active_call.closed:
            tail_str = active_call.finish()
            if tail_str:
                yield await generate_sse_response(
                    timestamp, model, content=None, tools_id=active_call.call_id,
                    function_call_name=None, function_call_content=tail_str,
                    tool_call_index=active_call.tc_index,
                )
            active_call = None

        is_image_model = _is_image_model(model)
        if is_image_model and not has_image:
            yield {
                "error": "Gemini image generation failed: no image was generated",
                "status_code": 502,
                "details": "Gemini image generation failed: no image was generated",
            }
            return

        if not is_image_model and not has_content and not has_reasoning and not has_function_call:
            yield {
                "error": "Gemini returned empty response",
                "status_code": 502,
                "details": "Gemini returned empty response",
            }
            return

        if not stream_finished_normally and (has_content or has_reasoning or has_function_call or has_image):
            stop_reason = "tool_calls" if has_function_call else "stop"
            yield await generate_sse_response(timestamp, model, stop=stop_reason)

        if totalTokenCount > 0:
            yield await generate_sse_response(
                timestamp, model, None, None, None, None, None,
                totalTokenCount, promptTokenCount, candidatesTokenCount,
                cached_tokens=cachedContentTokenCount, cache_creation_tokens=0,
            )

    yield "data: [DONE]" + end_of_line


_stream_gemini_with_partial_args._gemini_stream_fn_args_wrapper = True


def _install_channel_wrappers():
    for ch_id in GEMINI_ENGINES:
        ch = get_channel(ch_id)
        if not ch or not ch.stream_adapter:
            continue
        if getattr(ch.stream_adapter, "_gemini_stream_fn_args_wrapper", False):
            continue
        _ORIGINAL_STREAM_ADAPTERS[ch_id] = ch.stream_adapter
        register_channel(
            id=ch.id,
            type_name=ch.type_name,
            default_base_url=ch.default_base_url,
            auth_header=ch.auth_header,
            description=ch.description,
            request_adapter=ch.request_adapter,
            passthrough_adapter=ch.passthrough_adapter,
            passthrough_payload_adapter=getattr(ch, "passthrough_payload_adapter", None),
            passthrough_stream_adapter=getattr(ch, "passthrough_stream_adapter", None),
            passthrough_response_adapter=getattr(ch, "passthrough_response_adapter", None),
            stream_adapter=_stream_gemini_with_partial_args,
            response_adapter=ch.response_adapter,
            models_adapter=ch.models_adapter,
            source=ch.source,
            overwrite=True,
        )
        logger.info(f"[{PLUGIN_NAME}] Wrapped stream_adapter for channel '{ch_id}'")


def _uninstall_channel_wrappers():
    for ch_id, orig_adapter in list(_ORIGINAL_STREAM_ADAPTERS.items()):
        ch = get_channel(ch_id)
        if not ch:
            continue
        register_channel(
            id=ch.id,
            type_name=ch.type_name,
            default_base_url=ch.default_base_url,
            auth_header=ch.auth_header,
            description=ch.description,
            request_adapter=ch.request_adapter,
            passthrough_adapter=ch.passthrough_adapter,
            passthrough_payload_adapter=getattr(ch, "passthrough_payload_adapter", None),
            passthrough_stream_adapter=getattr(ch, "passthrough_stream_adapter", None),
            passthrough_response_adapter=getattr(ch, "passthrough_response_adapter", None),
            stream_adapter=orig_adapter,
            response_adapter=ch.response_adapter,
            models_adapter=ch.models_adapter,
            source=ch.source,
            overwrite=True,
        )
    _ORIGINAL_STREAM_ADAPTERS.clear()


def setup(manager=None):
    try:
        unregister_request_interceptor(_INTERCEPTOR_ID)
    except Exception:
        pass
    register_request_interceptor(
        interceptor_id=_INTERCEPTOR_ID,
        callback=gemini_stream_function_args_request_interceptor,
        priority=180,
        plugin_name=PLUGIN_NAME,
        metadata={"description": "Inject streamFunctionCallArguments for Gemini streaming tool calls"},
    )
    _install_channel_wrappers()
    logger.info(f"[{PLUGIN_NAME}] v{PLUGIN_INFO['version']} initialized")


def teardown(manager=None):
    try:
        unregister_request_interceptor(_INTERCEPTOR_ID)
    except Exception:
        pass
    _uninstall_channel_wrappers()
    logger.info(f"[{PLUGIN_NAME}] torn down")


# 模块导入时自动挂载（与 gemini_prefill 等插件保持一致）
setup()
