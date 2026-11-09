"""
Moonshot AI（月之暗面 / Kimi）渠道插件

对照官方文档 https://platform.kimi.ai/docs/api/chat 实现，
为 Zoaholic 新增一个 Moonshot OpenAI 兼容模式专用渠道，
避免把 Kimi 当普通 OpenAI 渠道接入时出现的格式错误。

修正要点（全部来自官方文档）：
1. base_url 以后台配置为准（支持自部署/代理网关，如 https://api.otto114514.com/v1），
   仅在后台留空或仍填官方域名时兜底为 https://api.moonshot.ai/v1，
   自动拼接 /chat/completions，防止后台误填不完整端点。
2. reasoning_effort 仅对 kimi-k3 生效（官方：low / high / max，默认 max）；
   其他模型传该字段会报错，必须移除。
3. kimi-k2.6 / kimi-k2.7-code 的 thinking 开关是 thinking.type / thinking.keep，
   不属于顶层 reasoning_effort（k2.7-code 恒为 enabled 且 thinking.keep 固定 "all"）。
4. stream_options（官方支持 {"include_usage": true}）、logprobs、top_logprobs、
   prediction、stop 官方都支持，**不要删**，否则流式 / 计费统计会出错。
5. tool_choice 官方支持 auto / none / required / 指定函数对象，不限模型，
   不做模型级降级（旧版 k2.x 不支持 required 的说法未见于官方文档）。
6. response_format 增加 json_schema（官方 Structured Output 推荐用法）。
7. 多模态 content 官方支持 text / image_url / video_url，相应的 video_url 保留。
8. 请求体仅清理真正不被官方支持的字段：
   functions（已由 tools 取代）、store、metadata、service_tier、modalities、audio、
   parallel_tool_calls、user、seed、logit_bias、logprobs 之外的非官方扩展字段。
9. max_tokens / max_completion_tokens 官方都接受（max_tokens 已 deprecated 但仍兼容）；
   为稳妥把 max_tokens 规范化改写成 max_completion_tokens，
   防止旧前端发出的 max_tokens 被未来版本拒绝。
10. 响应端复用 core 内置 OpenAI 解析（流式/非流式/模型列表/usage，
    reasoning_content 透传给前端）。

配置示例（后台添加渠道时）：
  渠道类型: moonshot
  base_url: https://api.moonshot.ai/v1 （默认即可，国内可改 https://api.moonshot.cn/v1）
  api: sk-xxxxxxxxxxxxxxxx
  model:
    - kimi-k3
    - kimi-k2.7-code
    - kimi-k2.6
    - kimi-k2.5
    - moonshot-v1
"""

import copy
from typing import TYPE_CHECKING

from core.channels.openai_channel import (
    get_gpt_payload,
    fetch_openai_response,
    fetch_gpt_response_stream,
    fetch_openai_models,
)

if TYPE_CHECKING:
    from core.plugins import PluginManager

# ==================== 插件元信息 ====================

PLUGIN_INFO = {
    "name": "moonshot_channel",
    "version": "1.0.2",
    "description": "Moonshot AI（月之暗面 / Kimi）OpenAI 兼容模式专用渠道（对照官方文档）",
    "author": "Mita",
    "dependencies": [],
    "metadata": {
        "category": "channel",
        "tags": ["moonshot", "kimi", "月之暗面", "openai-compatible"],
    },
}

EXTENSIONS = ["channels:moonshot"]

MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
MOONSHOT_BASE_URL_CN = "https://api.moonshot.cn/v1"

# Kimi Coding Plan 专属端点标记：该端点要求白名单 Coding-Agent User-Agent，否则 401/403
KIMI_CODING_MARKERS = ("api.kimi.com/coding",)


def _kimi_coding_headers(base_url: str) -> dict:
    """识别 Kimi Coding 端点并返回必须携带的 User-Agent 头。"""
    if base_url and any(m in base_url for m in KIMI_CODING_MARKERS):
        return {"User-Agent": "KimiCLI/1.0"}
    return {}


# 官方文档未列出或明确废弃/不兼容的顶层字段
# （官方支持：stream_options、logprobs、top_logprobs、prediction、stop、
#   prompt_cache_key、safety_identifier、temperature、top_p、n、
#   stream、tools、tool_choice、reasoning_effort、response_format）
_UNSUPPORTED_FIELDS = (
    "functions",          # 官方：已由 tools 取代，明确不支持
    "store",
    "metadata",
    "service_tier",
    "modalities",
    "audio",              # 顶层 audio 字段非官方 chat completion schema
    "parallel_tool_calls",
    "user",               # 官方用 safety_identifier 替代
    "seed",
    "logit_bias",
)

# kimi-k3 顶层 reasoning_effort 仅接受这三个取值
ALLOWED_REASONING_EFFORT = {"low", "high", "max"}
# 只有 kimi-k3 支持顶层 reasoning_effort
# （k2.6 / k2.7-code 用 thinking.type / thinking.keep，传 reasoning_effort 会报错）
REASONING_EFFORT_MODELS = {"kimi-k3"}

ALLOWED_TOOL_CHOICE = {"auto", "none", "required"}
ALLOWED_RESPONSE_FORMAT_TYPES = {"text", "json_object", "json_schema"}
# 官方 content 数组只接受这三种 type
ALLOWED_CONTENT_TYPES = {"text", "image_url", "video_url"}


def sanitize_payload(payload: dict) -> dict:
    """
    按官方文档清理发往 Moonshot 的请求体：
    只移除官方明确不支持/不会接受的字段，官方写明的字段一律保留。
    """
    if not isinstance(payload, dict):
        return payload

    payload = copy.deepcopy(payload)
    model = payload.get("model", "")

    # 顶层不支持/已废弃字段直接移除
    for field in _UNSUPPORTED_FIELDS:
        payload.pop(field, None)

    # reasoning_effort 只对 kimi-k3 生效；其他模型传了会报错
    reasoning_effort = payload.get("reasoning_effort")
    if reasoning_effort is not None:
        if model != "kimi-k3" or reasoning_effort not in ALLOWED_REASONING_EFFORT:
            payload.pop("reasoning_effort", None)

    # max_tokens 已 deprecated，官方推荐 max_completion_tokens；
    # 为兼容未来版本统一改写（只在没有显式 max_completion_tokens 时改写）
    if "max_tokens" in payload and "max_completion_tokens" not in payload:
        payload["max_completion_tokens"] = payload.pop("max_tokens")
    else:
        payload.pop("max_tokens", None)

    # tool_choice：auto / none / required / {type: function, function: {...}}
    tool_choice = payload.get("tool_choice")
    if tool_choice is not None:
        if isinstance(tool_choice, str):
            if tool_choice not in ALLOWED_TOOL_CHOICE:
                payload.pop("tool_choice", None)
        elif isinstance(tool_choice, dict):
            if tool_choice.get("type") != "function":
                payload.pop("tool_choice", None)
        else:
            payload.pop("tool_choice", None)

    # response_format：官方支持 text / json_object / json_schema
    response_format = payload.get("response_format")
    if response_format is not None:
        fmt_type = response_format.get("type") if isinstance(response_format, dict) else None
        if fmt_type not in ALLOWED_RESPONSE_FORMAT_TYPES:
            payload.pop("response_format", None)

    # 多模态 content 仅保留官方认可的三种类型，其他（如 input_audio）移除
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                cleaned = [
                    item for item in content
                    if not isinstance(item, dict) or item.get("type") in ALLOWED_CONTENT_TYPES
                ]
                msg["content"] = cleaned if cleaned else ""

    return payload


# ==================== 渠道适配器实现 ====================

async def get_moonshot_payload(request, engine, provider, api_key=None):
    """
    RequestAdapter：清理 payload，base_url 尊重后台配置（支持自部署/代理）。
    """
    provider = dict(provider)
    base = (provider.get("base_url") or "").rstrip("/")
    # 仅在后台没填 base_url、或填的仍是 moonshot 官方域名（可能缺路径）时兜底官方端点；
    # 其他情况（自部署/代理网关，如 api.otto114514.com）一律尊重后台配置
    if not base:
        provider["base_url"] = MOONSHOT_BASE_URL
    elif base in ("https://api.moonshot.ai", "https://api.moonshot.cn"):
        # 官方域名但缺 /v1 路径，补齐
        provider["base_url"] = base + "/v1"

    url, headers, payload = await get_gpt_payload(request, engine, provider, api_key=api_key)
    headers.update(_kimi_coding_headers(provider["base_url"]))
    payload = sanitize_payload(payload)

    return url, headers, payload


async def fetch_moonshot_models(client, provider):
    """拉取模型列表：Kimi Coding 端点需携带白名单 UA，其余走核心 OpenAI 逻辑。"""
    provider = dict(provider)
    base = (provider.get("base_url") or "").rstrip("/")
    if not base:
        provider["base_url"] = MOONSHOT_BASE_URL
    elif base in ("https://api.moonshot.ai", "https://api.moonshot.cn"):
        provider["base_url"] = base + "/v1"

    if not _kimi_coding_headers(provider["base_url"]):
        return await fetch_openai_models(client, provider)

    api_key = provider.get("api") or provider.get("api_key")
    if isinstance(api_key, list):
        api_key = api_key[0] if api_key else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(_kimi_coding_headers(provider["base_url"]))

    response = await client.get(provider["base_url"] + "/models", headers=headers)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and "data" in data:
        return [m.get("id") for m in data["data"] if m.get("id")]
    if isinstance(data, list):
        return [m.get("id") if isinstance(m, dict) else m for m in data]
    return []


class MoonshotChannelAdapter:
    """Moonshot AI 渠道适配器"""

    id = "moonshot"
    type_name = "openai"  # Moonshot 完全兼容 OpenAI 方言/透传格式

    request_adapter = staticmethod(get_moonshot_payload)
    stream_adapter = staticmethod(fetch_gpt_response_stream)
    response_adapter = staticmethod(fetch_openai_response)
    models_adapter = staticmethod(fetch_moonshot_models)


# ==================== 插件生命周期 ====================

def setup(manager: "PluginManager"):
    """插件初始化：注册 Moonshot 渠道"""

    manager.register_extension(
        extension_point="channels",
        extension_id="moonshot",
        implementation=MoonshotChannelAdapter,
        priority=100,
        metadata={
            "description": "Moonshot AI（月之暗面 / Kimi）OpenAI 兼容模式渠道适配器",
            "supported_features": ["chat", "stream", "vision", "video", "tools", "reasoning", "json_schema"],
        },
        plugin_name=PLUGIN_INFO["name"],
    )

    from core.channels.registry import register_channel

    try:
        register_channel(
            id=MoonshotChannelAdapter.id,
            type_name=MoonshotChannelAdapter.type_name,
            default_base_url=MOONSHOT_BASE_URL,
            auth_header="Authorization: Bearer {api_key}",
            description="Moonshot AI 月之暗面 Kimi（OpenAI 兼容模式）",
            request_adapter=MoonshotChannelAdapter.request_adapter,
            stream_adapter=MoonshotChannelAdapter.stream_adapter,
            response_adapter=MoonshotChannelAdapter.response_adapter,
            models_adapter=MoonshotChannelAdapter.models_adapter,
            source="plugin",
        )
        print(f"[{PLUGIN_INFO['name']}] Channel 'moonshot' registered successfully!")
    except ValueError as e:
        print(f"[{PLUGIN_INFO['name']}] Channel registration skipped: {e}")


def teardown(manager: "PluginManager"):
    """插件清理：注销 Moonshot 渠道"""
    manager.unregister_extension("channels", "moonshot")

    from core.channels.registry import unregister_channel
    unregister_channel("moonshot")

    print(f"[{PLUGIN_INFO['name']}] Channel 'moonshot' unregistered!")


def unload():
    """插件卸载回调"""
    print(f"[{PLUGIN_INFO['name']}] Plugin unloading...")


if __name__ == "__main__":
    print(f"Plugin: {PLUGIN_INFO['name']} v{PLUGIN_INFO['version']}")
    print(f"Description: {PLUGIN_INFO['description']}")
    print(f"Extensions: {EXTENSIONS}")
