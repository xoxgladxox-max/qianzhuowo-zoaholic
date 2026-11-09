"""Zoaholic display adapter for the external Ask Sage maintainer.

This plugin reads a public, token-free snapshot. Credential refresh and gateway
configuration changes happen only in the separate maintainer service.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


PLUGIN_INFO = {
    "name": "asksage_maintainer",
    "version": "1.0.0",
    "description": "显示已接管 Ask Sage 账号的推理额度和登录有效期",
    "author": "Local",
    "dependencies": [],
    "metadata": {"category": "balance", "tags": ["asksage", "quota"]},
}
EXTENSIONS = ["balance_enrichers:asksage_maintainer_balance"]
STATUS_PATH = Path(os.environ.get("ASKSAGE_MAINTAINER_STATUS_PATH", "/home/data/asksage-maintainer/status.json"))
PROVIDER = "anthropic"
ENGINE = "claude"
BASE_URL = "https://api.asksage.ai/server/anthropic/v1"
ENRICHER_ID = "asksage_maintainer_balance"
SLOT_ID = "asksage_maintainer.quota_display"


QUOTA_DISPLAY = """
export default function render(ctx) {
    const { el, data, context } = ctx || {};
    if (!el) return;
    const value = data?.asksage_maintainer ?? context?.balance?.asksage_maintainer;
    if (!value) { el.textContent = ''; return; }
    const parts = [];
    if (value.status === 'unmanaged') {
        parts.push('未接管');
    } else if (value.status === 'ok' || value.status === 'stale') {
        if (typeof value.available === 'number' && typeof value.total === 'number') {
            parts.push(`Ask Sage 推理额度 ${value.available.toLocaleString()}/${value.total.toLocaleString()}`);
        } else {
            parts.push('Ask Sage 推理额度未知');
        }
        if (value.status === 'stale') parts.push('旧数据');
    } else {
        parts.push('Ask Sage 推理额度未知');
    }
    if (value.token_expires_at) {
        const when = new Date(value.token_expires_at);
        if (!Number.isNaN(when.getTime())) parts.push(`登录令牌到期 ${when.toLocaleString()}`);
    }
    if (value.needs_credentials) parts.push('缺续期凭据');
    if (value.availability === 'unavailable') parts.push('不可用');
    el.textContent = parts.join(' · ');
    el.title = value.error_category ? `状态: ${value.error_category}` : '';
}
""".strip()


def _key_from_provider(provider: dict[str, Any]) -> str | None:
    value = provider.get("api") or provider.get("api_key")
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    if isinstance(value, dict) and len(value) == 1:
        only = next(iter(value.keys()))
        return only if isinstance(only, str) else None
    return None


def _snapshot() -> dict[str, Any] | None:
    try:
        with STATUS_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("provider") != PROVIDER or type(data.get("managed_count")) is not int or data["managed_count"] < 0 or not isinstance(data.get("accounts"), dict):
        return None
    return data


async def _balance_enricher(result: dict[str, Any], engine: str, provider: dict[str, Any]) -> dict[str, Any]:
    base_url = provider.get("base_url")
    if engine != ENGINE or provider.get("provider") != PROVIDER or not isinstance(base_url, str) or base_url.rstrip("/") != BASE_URL:
        return result
    token = _key_from_provider(provider)
    if not token:
        result["asksage_maintainer"] = {"status": "unknown", "error_category": "key_not_selected"}
        return result
    snapshot = _snapshot()
    if snapshot is None:
        result["asksage_maintainer"] = {"status": "unknown", "error_category": "status_unavailable"}
        return result
    hashed = hashlib.sha256(token.encode("utf-8")).hexdigest()
    matching = [entry for entry in snapshot["accounts"].values() if isinstance(entry, dict) and entry.get("key_sha256") == hashed]
    if len(matching) != 1:
        result["asksage_maintainer"] = {"status": "unmanaged"}
        return result
    entry = matching[0]
    # Explicit allowlist means corrupt or unexpected fields never enter the UI.
    result["asksage_maintainer"] = {
        "status": entry.get("quota_status") if entry.get("quota_status") in {"ok", "stale", "unknown", "unsupported"} else "unknown",
        "total": entry.get("total") if type(entry.get("total")) in {int, float} else None,
        "used": entry.get("used") if type(entry.get("used")) in {int, float} else None,
        "available": entry.get("available") if type(entry.get("available")) in {int, float} else None,
        "token_expires_at": entry.get("token_expires_at") if isinstance(entry.get("token_expires_at"), str) else None,
        "needs_credentials": entry.get("needs_credentials") is True,
        "availability": entry.get("availability") if entry.get("availability") in {"usable", "exhausted", "unavailable", "unknown", "retired", "externally_removed"} else "unknown",
        "error_category": entry.get("error_category") if isinstance(entry.get("error_category"), str) else None,
        "unit": "Ask Sage inference tokens",
    }
    return result


def setup(manager: Any) -> None:
    from core.plugins import register_balance_enricher
    from core.ui_slots.registry import register_ui_slot

    register_balance_enricher(
        enricher_id=ENRICHER_ID,
        callback=_balance_enricher,
        priority=100,
        plugin_name=PLUGIN_INFO["name"],
        overwrite=True,
    )
    register_ui_slot(
        slot_id=SLOT_ID,
        slot="quota_display",
        script=QUOTA_DISPLAY,
        source=PLUGIN_INFO["name"],
        priority=50,
        mode="replace",
        engines=[ENGINE],
        auth_types=["api_key"],
        enabled_plugin=PLUGIN_INFO["name"],
    )


def teardown(manager: Any) -> None:
    from core.plugins import unregister_balance_enricher
    from core.ui_slots.registry import unregister_ui_slot

    unregister_balance_enricher(ENRICHER_ID)
    unregister_ui_slot(SLOT_ID)


def unload() -> None:
    teardown(None)
