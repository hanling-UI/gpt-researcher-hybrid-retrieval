"""访问口令 + 每日研究次数上限。

目的：演示站点是公开的，但每次研究都会调用付费模型接口。陌生人或爬虫
只要知道网址，就能刷爆你的 API 额度。这里加两道闸：

1. **访问口令**（`ACCESS_PASSWORD`）：不知道口令的人连首页都看不到，
   更别说发请求。口令只给需要演示的人。
2. **每日次数上限**（`DAILY_RESEARCH_LIMIT`）：就算口令泄露，一天最多
   只能跑 N 次研究，成本有天花板。

设计取舍：

* 口令用 cookie 里的 SHA-256 令牌，不存明文；比较用 `hmac.compare_digest`
  防时序攻击。
* 次数按 UTC 日期计数，落在 `data/usage.json`。免费实例重启会丢，
  这可以接受——宁可重置也不能因为写盘失败把服务卡住。
* **未设置 `ACCESS_PASSWORD` 时整个门禁关闭**：本地开发和 CI 不受影响。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path

COOKIE_NAME = "gptr_access"
COOKIE_MAX_AGE = 7 * 24 * 3600  # 7 天
_USAGE_FILE = Path("data/usage.json")
_TOKEN_PREFIX = "gptr-access-v1:"


def gate_enabled() -> bool:
    """只有显式配置了口令才启用门禁，避免影响本地开发。"""
    return bool((os.getenv("ACCESS_PASSWORD") or "").strip())


def daily_limit() -> int:
    """每日研究次数上限；<=0 表示不限制。"""
    raw = (os.getenv("DAILY_RESEARCH_LIMIT") or "10").strip()
    try:
        return int(raw)
    except ValueError:
        return 10


def token_for(password: str) -> str:
    return hashlib.sha256((_TOKEN_PREFIX + password).encode("utf-8")).hexdigest()


def expected_token() -> str:
    password = (os.getenv("ACCESS_PASSWORD") or "").strip()
    return token_for(password) if password else ""


def is_valid_token(token: str | None) -> bool:
    if not gate_enabled():
        return True
    if not token:
        return False
    return hmac.compare_digest(token, expected_token())


def check_password(candidate: str | None) -> bool:
    if not gate_enabled():
        return True
    if not candidate:
        return False
    return hmac.compare_digest(candidate.strip(), (os.getenv("ACCESS_PASSWORD") or "").strip())


# ---------------------------------------------------------------------------
# 每日额度
# ---------------------------------------------------------------------------


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_usage() -> dict:
    try:
        data = json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - 文件不存在/损坏都按"没用过"处理
        return {}


def _save_usage(data: dict) -> None:
    try:
        _USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _USAGE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001 - 写盘失败不应让请求失败
        pass


def used_today() -> int:
    entry = _load_usage().get(_today())
    if isinstance(entry, dict):
        try:
            return int(entry.get("count", 0))
        except (TypeError, ValueError):
            return 0
    return 0


def remaining() -> int:
    """剩余可用次数；门禁关闭或不限额时返回一个很大的数。"""
    if not gate_enabled() or daily_limit() <= 0:
        return 10**9
    return max(0, daily_limit() - used_today())


def consume() -> tuple[bool, int]:
    """尝试占用一次研究额度。

    Returns:
        (是否允许, 剩余次数)。被拒绝时剩余次数为 0。
    """
    if not gate_enabled() or daily_limit() <= 0:
        return True, 10**9

    usage = _load_usage()
    today = _today()
    entry = usage.get(today)
    count = int(entry.get("count", 0)) if isinstance(entry, dict) else 0

    if count >= daily_limit():
        return False, 0

    usage[today] = {"count": count + 1}
    _save_usage(usage)
    return True, max(0, daily_limit() - (count + 1))


LIMIT_MESSAGE = (
    "今日演示次数已用完（上限 {limit} 次）。这是为了控制 API 成本——"
    "每次研究都会调用付费模型接口。明天会自动重置，或联系我获取更多额度。"
)


def limit_message() -> str:
    return LIMIT_MESSAGE.format(limit=daily_limit())


# ---------------------------------------------------------------------------
# 登录页
# ---------------------------------------------------------------------------

_LOGIN_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>访问验证</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(45deg, #1a1a2e, #0f2351); color: #e5e7eb;
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
  }}
  .card {{
    width: min(92vw, 380px); padding: 32px 28px; border-radius: 16px;
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1);
    box-shadow: 0 20px 50px rgba(0,0,0,.45);
  }}
  h1 {{ font-size: 18px; margin: 0 0 6px; }}
  p.hint {{ font-size: 13px; color: #9ca3af; margin: 0 0 22px; line-height: 1.6; }}
  input {{
    width: 100%; padding: 11px 13px; border-radius: 10px; font-size: 14px;
    border: 1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.05); color: #e5e7eb;
  }}
  input:focus {{ outline: 2px solid rgba(20,184,166,.5); border-color: transparent; }}
  button {{
    width: 100%; margin-top: 14px; padding: 11px; border: none; border-radius: 10px;
    background: linear-gradient(to right, #0d9488, #14b8a6); color: #fff;
    font-size: 15px; font-weight: 600; cursor: pointer;
  }}
  button:hover {{ opacity: .9; }}
  .error {{
    margin: 0 0 14px; padding: 9px 12px; border-radius: 8px; font-size: 13px;
    background: rgba(239,68,68,.15); border: 1px solid rgba(239,68,68,.35); color: #fca5a5;
  }}
</style>
</head>
<body>
  <form class="card" method="post" action="/login">
    <h1>访问验证</h1>
    <p class="hint">这是个人作品演示站，每次研究都会产生 API 费用，因此需要口令。请联系站点主人获取。</p>
    {error}
    <input type="password" name="password" placeholder="请输入访问口令" autofocus autocomplete="current-password">
    <button type="submit">进入</button>
  </form>
</body>
</html>"""


def login_page(error: str = "") -> str:
    error_html = f'<p class="error">{error}</p>' if error else ""
    return _LOGIN_HTML.format(error=error_html)
