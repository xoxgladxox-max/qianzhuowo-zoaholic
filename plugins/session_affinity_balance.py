"""渠道可选的 IP + 原始 system UTF-8 字节数亲和调度。

TTL=3600 秒，每次成功选 Key 续期；新会话按最近 60 秒分配次数、
有效绑定数、独立轮转顺序选择。只统计分配，不估算 Token/缓存成本。
只读解码结果保存长度，不保存正文，不输出 Key；无预热或模型调用。
通过 setup/unload 安装/撤销运行时覆盖，不修改核心文件。
"""

import asyncio
from collections import Counter, deque
from time import time

from fastapi import HTTPException
from core import middleware
from core.key_pool import ThreadSafeCircularList, provider_api_circular_list
from core.plugins.interceptors import (
    is_plugin_enabled,
    register_channel_inbound_interceptor,
    unregister_channel_inbound_interceptor,
)

NAME = "session_affinity_balance"
TTL = 3600
WINDOW = 60
LENGTH_FIELD = "_session_affinity_system_bytes"
STATE_ATTR = "_session_affinity_balance_state_v1"
PLUGIN_INFO = {
    "name": NAME,
    "version": "1.0.0",
    "description": "IP+原始system字节数亲和；新会话按近期分配量分流（3600s/60s）",
    "author": "local",
}
_installation = None


def _text_bytes(value):
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, list):
        return sum(
            _text_bytes(block.get("text"))
            for block in value
            if isinstance(block, dict)
            and block.get("type") in (None, "text", "input_text")
        )
    return 0


def _message_bytes(messages):
    if not isinstance(messages, list):
        return 0
    return sum(
        _text_bytes(message.get("content"))
        for message in messages
        if isinstance(message, dict) and message.get("role") in ("system", "developer")
    )


def _system_bytes(payload, endpoint=""):
    """解码后原文逐段 UTF-8 求和：保留空白，不添加分隔符，不计 JSON 转义。

    协议明确时只计该协议的字段；无 system 时为 0。不计 cache_control、
    图片及其它非文本块。非法孤立 surrogate 无 UTF-8 表示，由捕获层回退 0。
    """
    if not isinstance(payload, dict):
        return 0
    path = endpoint.split(" ", 1)[-1]
    responses = path.endswith("/responses")
    gemini = "/models/" in path
    anthropic = path.endswith("/messages")
    chat = path.endswith("/chat/completions")
    if responses or (not (gemini or anthropic or chat) and "instructions" in payload):
        return _text_bytes(payload.get("instructions")) + _message_bytes(payload.get("input"))
    if gemini or (not (responses or anthropic or chat)
                  and ("systemInstruction" in payload or "system_instruction" in payload)):
        instruction = payload.get("systemInstruction", payload.get("system_instruction"))
        return _text_bytes(instruction.get("parts")) if isinstance(instruction, dict) else 0
    if anthropic or (not (responses or gemini or chat) and "system" in payload):
        return _text_bytes(payload.get("system"))
    return _message_bytes(payload.get("messages"))


def _capture_decode(original_decode, active, *args, **kwargs):
    # StatsMiddleware 在首次 JSON 解码前已建立 request_info；重试/审查解码不覆盖。
    info = middleware.request_info.get()
    capture = (active and isinstance(info, dict) and info.get("request_id")
               and LENGTH_FIELD not in info)
    if capture:
        info[LENGTH_FIELD] = 0
    payload = original_decode(*args, **kwargs)
    if capture:
        try:
            info[LENGTH_FIELD] = _system_bytes(payload, info.get("endpoint", ""))
        except UnicodeEncodeError:
            pass
    return payload


def _state(pool, channel, now):
    # 挂在池对象上，热重载保留亲和；池被配置重建后自然释放。渠道间不共享绑定。
    states = getattr(pool, STATE_ATTR, None)
    if states is None:
        states = {}
        setattr(pool, STATE_ATTR, states)
    if channel not in states:
        legacy = {}
        if pool.schedule_algorithm == "sticky_ip":
            for ip, (index, expires) in pool._sticky_sessions.items():
                if expires > now and 0 <= index < len(pool.items):
                    legacy[ip] = (pool.items[index], expires)
        states[channel] = {"sessions": {}, "recent": {}, "cursor": 0, "legacy": legacy}
    return states[channel]


async def _balanced_next(pool, info, model):
    now = time()
    channel = info["_provider_cfg"]["provider"]
    state = _state(pool, channel, now)
    sessions, recent = state["sessions"], state["recent"]
    items = pool.items
    pool._request_cleanup_counter += 1
    if pool._request_cleanup_counter % 100 == 0:
        pool._cleanup_stale_request_keys()
        state["sessions"] = sessions = {
            sid: binding for sid, binding in sessions.items()
            if binding[1] > now and binding[0] in items
        }
        state["legacy"] = {
            ip: binding for ip, binding in state["legacy"].items()
            if binding[1] > now and binding[0] in items
        }
    for key in list(recent):
        if key not in items:
            del recent[key]
        else:
            while recent[key] and recent[key][0] <= now - WINDOW:
                recent[key].popleft()
    sid = (info.get("client_ip", ""), info.get(LENGTH_FIELD, 0))
    binding = sessions.get(sid)
    # 旧 sticky_ip 仅能知道 IP：该 IP 首次出现的长度继承原绑定，之后独立分流。
    if binding is None:
        binding = state["legacy"].get(sid[0])
    bound_key = binding[0] if binding and binding[1] > now and binding[0] in items else None
    counts = Counter(key for key, expiry in sessions.values() if expiry > now and key in items)
    counts.update(key for key, expiry in state["legacy"].values() if expiry > now and key in items)
    order = sorted(range(len(items)), key=lambda i: (
        len(recent.get(items[i], ())), counts[items[i]],
        (i - state["cursor"]) % len(items),
    ))
    if bound_key is not None:
        bound_index = items.index(bound_key)
        order.remove(bound_index)
        order.insert(0, bound_index)
    for index in order:
        item = items[index]
        # 保持 after_next_current() 及原 next 的游标语义。
        pool.index = (index + 1) % len(items)
        # 不先做 is_check 探测：沿用原方法，只有实际可用的 Key 计数一次。
        if not await pool.is_rate_limited(item, model):
            selected_at = time()
            sessions[sid] = (item, selected_at + TTL)
            state["legacy"].pop(sid[0], None)
            recent.setdefault(item, deque()).append(selected_at)
            if item != bound_key:
                state["cursor"] = pool.index
            return item
        # 保留当前核心的持锁遍历让出事件循环补丁。
        await asyncio.sleep(0)
    raise HTTPException(status_code=429, detail="Too many requests")


def _selected(pool, info):
    provider = info.get("_provider_cfg") if isinstance(info, dict) else None
    return (is_plugin_enabled(NAME, provider)
            and provider_api_circular_list.get(provider.get("provider")) is pool)


async def _next(pool, model, original_next, installation):
    info = middleware.request_info.get()
    if installation["active"] and _selected(pool, info) and pool.items:
        async with pool.lock:
            # 禁用/重载可能发生在等待锁期间；退出锁后才委托原方法，避免重入死锁。
            if installation["active"] and _selected(pool, info):
                return await _balanced_next(pool, info, model)
    return await original_next(pool, model)


async def _channel_entry(request_data, request, provider, api_key_info, enabled_plugins):
    # 只用于在 GUI 渠道插件列表中声明能力；调度在此阶段之前已完成。
    return request_data


def setup(manager):
    global _installation
    if _installation is not None:
        return
    original_decode = middleware.json_loads
    original_next = ThreadSafeCircularList.next
    installation = {"active": True}

    # 原方法和状态必须由每次安装的闭包持有，而非可被下一次 setup 改写的全局。
    def decode(*args, **kwargs):
        return _capture_decode(original_decode, installation["active"], *args, **kwargs)

    async def next_key(pool, model=None):
        return await _next(pool, model, original_next, installation)

    register_channel_inbound_interceptor(
        NAME, _channel_entry, plugin_name=NAME,
        metadata={"name": "会话亲和低负载分流", "description": PLUGIN_INFO["description"]},
    )
    _installation = (installation, decode, next_key, original_decode, original_next)
    middleware.json_loads = decode
    ThreadSafeCircularList.next = next_key


def teardown(manager=None):
    global _installation
    if _installation is None:
        return
    installation, decode, next_key, original_decode, original_next = _installation
    installation["active"] = False
    # 不抹掉随后加载的其它插件 wrapper；遗留闭包仅透传到其安装时的原方法。
    if middleware.json_loads is decode:
        middleware.json_loads = original_decode
    if ThreadSafeCircularList.next is next_key:
        ThreadSafeCircularList.next = original_next
    unregister_channel_inbound_interceptor(NAME)
    _installation = None


def unload():
    # 文件覆盖路径直接调用 loader.unload_plugin，不一定经过 manager.teardown。
    teardown()
