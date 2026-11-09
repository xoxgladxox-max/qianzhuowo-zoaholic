"""
Gemini 工具 Schema 清洗插件

用途：
    修复 Gemini 后端 400 报错：
    "Unknown name \"type\" ... Proto field is not repeating, cannot start list."

    Gemini 的 function_declarations 使用 proto Schema，type 字段只接受
    单个字符串，不接受 ["string", "null"] 这类数组写法，也不支持 anyOf。
    本插件在请求发往上游前，递归清洗 tools 里的 JSON Schema：
      - "type": ["string", "null"]  ->  "type": "string", "nullable": true
      - anyOf: [...]                ->  拍平为第一个非 null 类型（保留 nullable）

启用方式：
    在渠道的插件配置中勾选 gemini_schema_fix。
    注意：对「上游最终会转成 Gemini」的 OpenAI 兼容渠道也要勾选，
    因为脏 schema 是在本层以 OpenAI 格式携带的，洗好后上游转换才干净。
"""

from core.plugins.interceptors import register_request_interceptor
from core.log_config import logger

PLUGIN_INFO = {
    "name": "gemini_schema_fix",
    "version": "1.1.0",
    "description": "Gemini 工具 Schema 清洗 - 压平 type 数组 / anyOf，修复 Proto field is not repeating 400 报错",
    "author": "Praxis",
}


def _sanitize_schema(node, stats):
    """递归清洗 JSON Schema 节点，stats 用于统计修改次数"""
    if isinstance(node, dict):
        t = node.get("type")
        if isinstance(t, list):
            non_null = [x for x in t if x != "null"]
            if "null" in t:
                node["nullable"] = True
            node["type"] = non_null[0] if non_null else "string"
            stats["type_array"] += 1

        any_of = node.get("anyOf")
        if isinstance(any_of, list):
            merged = next(
                (s for s in any_of if isinstance(s, dict) and s.get("type") not in (None, "null")),
                {"type": "string"},
            )
            if any(isinstance(s, dict) and s.get("type") == "null" for s in any_of):
                merged = {**merged, "nullable": True}
            node.update(merged)
            del node["anyOf"]
            stats["any_of"] += 1

        for v in node.values():
            _sanitize_schema(v, stats)
    elif isinstance(node, list):
        for item in node:
            _sanitize_schema(item, stats)


def _fix_tools(payload, stats):
    """同时处理 OpenAI 形态和 Gemini 原生形态的 tools 字段"""
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        # OpenAI 形态：tools[].function.parameters
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("parameters"), dict):
            _sanitize_schema(fn["parameters"], stats)

        # Gemini 原生形态：tools[].function_declarations[].parameters
        decls = tool.get("function_declarations")
        if isinstance(decls, list):
            for decl in decls:
                if isinstance(decl, dict) and isinstance(decl.get("parameters"), dict):
                    _sanitize_schema(decl["parameters"], stats)


async def gemini_schema_fix(request, engine, provider, api_key, url, headers, payload):
    """请求拦截器：在发往上游前清洗 tools schema（不限引擎，勾了哪个渠道就洗哪个）"""
    stats = {"type_array": 0, "any_of": 0}

    if isinstance(payload, dict) and isinstance(payload.get("tools"), list):
        _fix_tools(payload, stats)

    # 必打日志：用于确认插件被调用、以及实际清洗了几处
    logger.info(
        f"gemini_schema_fix: engine={engine!r}, "
        f"has_tools={isinstance(payload, dict) and 'tools' in payload}, "
        f"fixed type_array={stats['type_array']}, anyOf={stats['any_of']}"
    )

    return url, headers, payload


register_request_interceptor(
    "gemini_schema_fix",
    gemini_schema_fix,
    priority=40,
    plugin_name="gemini_schema_fix",
)
