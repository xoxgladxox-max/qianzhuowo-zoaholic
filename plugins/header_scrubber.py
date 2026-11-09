"""上游请求头清洗插件（header_scrubber）

删除客户端入站残留的「真实 IP / 地理位置 / 链路指纹 / 隐私 / hop-by-hop」头，
使出站请求尽量接近官方 SDK 直连的形态。

为什么需要它
    nginx 用 $proxy_add_x_forwarded_for 给入站请求打上
    `x-forwarded-for: <客户端真实IP>, <CF节点IP>`，CF 另外附带
    cf-connecting-ip / cf-ipcountry 等一整套地理与链路头。
    而 passthrough 路径的 _filter_passthrough_headers() 只丢弃 7 个头
    （authorization / x-api-key / api-key / x-goog-api-key / host /
    content-length / accept-encoding），其余**全量透传给上游**。
    本插件在 request_interceptors 的最后一棒把它们清掉。

匹配顺序（命中即停）
    ① NEVER_STRIP        删了必挂的头，任何参数都突破不了
    ② strip=             用户显式指定删除
    ③ keep= / PROTECTED / preferences.headers 里出现过的 key   → 保留
    ④ STRIP              精确 + 前缀匹配，删除
    ⑤ 默认放行           未知头一律保留（保守，新渠道功能头不受影响）

③ 里自动纳入 provider.preferences.headers 的 key，是因为 apply_custom_headers()
在本拦截器**之前**执行 —— 管理员显式配置的 Host/Cookie/X-Title 会混进 headers，
不保护就会被静默删掉，等于插件覆盖了管理员的明确意图。

参数（enabled_plugins 里写 "header_scrubber:k=v,k=v"）
    (空)          推荐，无需任何参数
    keep=a;b;c    额外保护（分号分隔，因为逗号已被插件参数格式占用）
    strip=a;b;c   额外删除（可突破 PROTECTED，但突破不了 NEVER_STRIP）

本插件**不碰 user-agent**。UA 是渠道身份的一部分（换额度的凭据），
且本网关 httpx client 的默认 UA 被硬编码为 curl/7.68.0 —— 删掉 UA 只会
退化成一个 2020 年的旧版 curl，比任何客户端 UA 都醒目。
要调整 UA 请用渠道的 preferences.headers（框架原生支持设置与删除）。

逃生舱
    Cookie 认证渠道 → keep=cookie ｜ 域前置 → keep=host
    连 OS/架构一起藏 → strip=x-stainless-os;x-stainless-arch

设计论证与逐条取舍见同目录 README.md。

安全约束：fail-open（异常只记日志），只删不加，不改 payload / url / user-agent。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

from core.log_config import logger
from core.plugins import register_request_interceptor, unregister_request_interceptor
from core.plugins.interceptors import get_plugin_options


PLUGIN_NAME = "header_scrubber"
_INTERCEPTOR_ID = "header_scrubber_request"

PLUGIN_INFO = {
    "name": PLUGIN_NAME,
    "version": "3.0.0",
    "description": "上游请求头清洗 - 删除客户端真实 IP、地理位置、链路指纹与隐私头（不碰 user-agent）",
    "author": "Zoaholic Team",
    "dependencies": [],
    "metadata": {
        "category": "interceptors",
        "tags": ["privacy", "headers", "sanitizer", "fingerprint", "geo"],
        "params_hint": (
            "参数格式：header_scrubber[:k=v,k=v]\n"
            "  (空)         = 推荐，无需参数\n"
            "  keep=a;b;c   = 额外保护这些头（分号分隔）\n"
            "  strip=a;b;c  = 额外删除这些头\n"
            "本插件不碰 user-agent。要调 UA 请用渠道 preferences.headers。\n"
            "preferences.headers 里配置的头会自动受保护。\n"
            "常见：keep=cookie（Cookie 认证渠道）/ keep=host（域前置）"
        ),
    },
}

EXTENSIONS = ["interceptors:header_scrubber_request"]


# 该渠道在本拦截器之后会把出站头裁剪到自己的极小白名单，本插件在它上面是
# 纯粹的无用开销，且它对请求头极其敏感。直接跳过。
SKIP_ENGINES = frozenset({"antigravity"})


# ==================== ① NEVER_STRIP：任何参数都突破不了 ====================
# 删掉必然导致请求失败的头。strip= 也不能删它们 —— 这是防手滑的硬底线。

NEVER_STRIP = frozenset({
    "authorization", "x-api-key", "api-key", "x-goog-api-key", "x-auth-token",
    "content-type", "user-agent",
})


# ==================== ② PROTECTED：命中即保留 ====================

PROTECTED_EXACT = frozenset({
    "accept",
    # Codex / 各家 CLI 的身份头
    "session_id", "originator", "x-session-id", "x-client-name", "x-client-version",
    "x-request-timeout", "x-portkey-provider",
})

PROTECTED_PREFIXES = (
    "x-amz-",         # AWS SigV4 签名集合 —— 删任何一个都会 403
    "x-goog-",        # Vertex / Google
    "x-ms-", "azure-",
    "anthropic-",     # anthropic-version / -beta / -dangerous-direct-browser-access
    "openai-",        # openai-beta / -organization / -project
    "x-stainless-",   # 官方 SDK 自洽伴生头；删了会让 UA 与伴生头不匹配，反而更可疑
    "cf-aig-",        # CF AI Gateway 功能头（会被下面 cf- 命中，靠本行救回）
    "grpc-",
)


# ==================== ③ STRIP：删除规则 ====================

STRIP_EXACT = frozenset({
    # --- 客户端真实 IP（各家 CDN / LB / 应用框架的变体） ---
    "via", "forwarded", "x-forwarded", "cdn-loop",
    "true-client-ip", "fastly-client-ip", "client-ip",
    "x-client-ip", "x-cluster-client-ip", "x-originating-ip",
    "proxy-client-ip", "wl-proxy-client-ip",   # Java / WebLogic 系
    "x-proxyuser-ip",                          # Google
    "x-remote-addr", "remote-addr", "x-coming-from", "x-from-ip",
    "x-host", "x-scheme",
    # --- 地理位置（主人重点：CF 会把国家/城市/经纬度都塞进来） ---
    "x-country-code", "x-timezone",
    "accept-language",   # "zh-CN" —— 最强的地区信号，比 IP 还直白
    # --- 个人隐私 / 反代域名泄露 ---
    "cookie",   # 可能含面板 session
    "origin",   # Web 客户端强制携带 = 反代域名
    "referer",  # 反代域名 + 完整路径。实测已泄露到上游，危害不低于 IP
    "from",
    # --- 浏览器指纹（暴露"这是 Web UI 而非 SDK"） ---
    "dnt", "upgrade-insecure-requests", "priority", "pragma",
    "purpose", "sec-purpose", "sec-gpc",
    "device-memory", "viewport-width", "rtt", "downlink", "ect",
    # --- 分布式追踪（可能含内网服务名） ---
    "traceparent", "tracestate", "baggage", "b3",
    "x-request-id", "x-correlation-id", "x-trace-id", "x-span-id",
    "x-cloud-trace-context",   # GCP LB 注入
    # --- OpenRouter 排行榜标识 ---
    # 注意：这两个头 openrouter_channel 也会主动注入。删掉只影响排行榜归属，
    # 不影响功能；想保留归属用 keep=http-referer;x-title。
    "http-referer", "x-title",
    # --- 稳健性：必须让 httpx 重算/重设 ---
    # host：残留的入站 Host（= 面板域名）既泄露反代域名，又造成上游 SNI/路由错配。
    #       已确认 AWS SigV4 签名不依赖入站 host（canonical host 由 httpx 按 URL 重设）。
    # content-length：payload 被前置插件改写后，旧长度会导致请求体截断或连接挂起。
    # 三者 passthrough 已先删一次，这里是兜底（普通转换路径同样受益）。
    "host", "content-length", "transfer-encoding", "accept-encoding",
    # --- hop-by-hop 残留（RFC 9110 §7.6.1 规定不得转发） ---
    # 故意不删 connection / keep-alive：值只有 keep-alive|close，零隐私零指纹价值，
    # 而部分逆向渠道的出站白名单里保留了 connection，可能靠它匹配真实客户端。
    # 收益 0 而风险 >0 → 放行。需要时用 strip=connection 手动删。
    "proxy-connection", "proxy-authorization",
    "te", "trailer", "upgrade", "expect",
})

STRIP_PREFIXES = (
    "x-forwarded-",   # -for / -host / -proto / -port / -server / 未来变体
    "x-original-",
    "x-real-",
    # cf-*：connecting-ip / ray / visitor / pseudo-ipv4，以及一整套地理头
    # （ipcountry / ipcity / region / timezone / iplatitude / iplongitude）。
    # 前缀匹配让未来新增的 cf 地理头自动被覆盖。cf-aig-* 由 PROTECTED 救回。
    "cf-",
    "cloudfront-",    # AWS CloudFront：viewer-country / -city / -latitude / -longitude
    "x-azure-",       # Azure Front Door：clientip / socketip（注意与受保护的 azure- 不同）
    "x-akamai-",      # Akamai edgescape 含 country / region / lat / long
    "fly-",           # Fly.io：client-ip / region / forwarded-*
    "x-geo-",
    "sec-ch-",        # sec-ch-ua* 客户端提示
    "sec-fetch-",     # sec-fetch-site / -mode / -dest / -user
    "x-envoy-",
    "x-b3-",
    "x-datadog-",
    "x-newrelic",
    "x-amzn-trace",   # ALB 追踪。与受保护的 x-amz- 不冲突（第 5 字符 'n' ≠ '-'）
    "x-appengine-",   # GAE 注入，含 -country / -city，直接泄露地理位置
    "x-vercel-",      # 含 x-vercel-ip-country / -city / -latitude
    "x-nf-", "x-render-", "x-railway-",
)


def _split_list(value: Optional[str]) -> Set[str]:
    """解析 "a;b;c" 为小写集合。用分号是因为逗号已被插件参数格式占用。"""
    if not value:
        return set()
    return {p.strip().lower() for p in value.split(";") if p.strip()}


def _parse_options(options: Any) -> Dict[str, str]:
    """解析 "k=v,k=v"。无 "=" 的裸词按 flag 处理。

    enabled_plugins 条目可以是字符串，也可以是 dict / list 形式，
    三种都要处理 —— 否则参数被静默丢弃，用户以为配了其实没生效。
    """
    if isinstance(options, dict):
        return {str(k).strip().lower(): str(v).strip() for k, v in options.items()}
    if isinstance(options, (list, tuple)):
        options = ",".join(str(x) for x in options)
    if not isinstance(options, str) or not options.strip():
        return {}

    result: Dict[str, str] = {}
    for part in options.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip().lower()] = v.strip()
        else:
            result[part.lower()] = "true"
    return result


def _preference_header_keys(provider: Any) -> Set[str]:
    """provider.preferences.headers 里管理员显式配置的 key（小写）。

    apply_custom_headers() 在本拦截器之前执行，这些头已经混进 headers。
    它们是管理员的明确意图，不能被通用 STRIP 规则静默覆盖。

    例外：值为 "null" 是框架约定的**删除**指令，那种 key 的意图恰好相反，
    不能纳入保护 —— 否则会挡住本插件的兜底删除。
    """
    try:
        prefs = (provider or {}).get("preferences") or {}
        custom = prefs.get("headers") or {}
        if not isinstance(custom, dict):
            return set()
        return {
            str(k).lower()
            for k, v in custom.items()
            if not (isinstance(v, str) and v.strip().lower() == "null")
        }
    except Exception:
        return set()


def _should_strip(name_lower: str, keep: Set[str], strip: Set[str]) -> bool:
    """五层匹配，命中即停。顺序见模块 docstring。"""
    if name_lower in NEVER_STRIP:
        return False
    if name_lower in strip:
        return True
    if name_lower in keep:
        return False
    if name_lower in PROTECTED_EXACT or name_lower.startswith(PROTECTED_PREFIXES):
        return False
    return name_lower in STRIP_EXACT or name_lower.startswith(STRIP_PREFIXES)


async def header_scrubber_request(
    request: Any,
    engine: str,
    provider: Dict[str, Any],
    api_key: Optional[str],
    url: str,
    headers: Dict[str, Any],
    payload: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """原地清洗 headers 并显式回传（与其余插件的返回惯例一致）。"""
    try:
        if not isinstance(headers, dict) or not headers or engine in SKIP_ENGINES:
            return url, headers, payload

        opts = _parse_options(get_plugin_options(PLUGIN_NAME, provider))
        keep = _split_list(opts.get("keep")) | _preference_header_keys(provider)
        strip = _split_list(opts.get("strip"))

        # 遍历删除，同时天然覆盖同名不同大小写的重复 key
        removed = []
        for name in list(headers.keys()):
            if _should_strip(str(name).lower(), keep, strip):
                del headers[name]
                removed.append(str(name))

        if removed:
            logger.debug(
                f"[{PLUGIN_NAME}] {engine}: stripped {len(removed)}: {', '.join(removed)}"
            )
    except Exception as e:
        logger.error(f"[{PLUGIN_NAME}] error (fail-open): {e}", exc_info=True)
    return url, headers, payload


# ==================== 注册（幂等） ====================
try:
    unregister_request_interceptor(_INTERCEPTOR_ID)
except Exception:
    pass

register_request_interceptor(
    _INTERCEPTOR_ID,
    header_scrubber_request,
    # 升序执行：数值越小越先跑。现存拦截器区间 3~999，本插件排在最后，
    # 确保无论前面哪个环节合并了入站头都会被清掉。
    # 注意：这之后仍有 httpx client 默认头合并与部分渠道的出站白名单裁剪，
    # 所以「最后一个拦截器」≠「最后一个动 headers 的环节」。
    priority=1000,
    plugin_name=PLUGIN_NAME,
    metadata={"stage": "request_interceptors"},
)

logger.info(f"[{PLUGIN_NAME}] v{PLUGIN_INFO['version']} loaded")
