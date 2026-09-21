"""Application-owned function tools.

The model is allowed to request only tools registered in this module.  The
server validates and executes the request, so API keys and private business
logic never need to be sent to the Android client.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone as dt_timezone
import json
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class ToolExecutionError(RuntimeError):
    """A safe, user-facing tool execution failure."""


async def get_current_time(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the current time for an IANA timezone."""
    timezone_name = str(arguments.get("timezone") or "Asia/Shanghai")
    try:
        timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        # Windows installations do not always ship an IANA tz database. Keep
        # the default project timezone usable even before tzdata is installed.
        fixed_offsets = {
            "UTC": dt_timezone.utc,
            "Asia/Shanghai": dt_timezone(timedelta(hours=8), name="Asia/Shanghai"),
        }
        timezone = fixed_offsets.get(timezone_name)
        if timezone is None:
            raise ToolExecutionError(f"不支持的时区：{timezone_name}")
    now = datetime.now(timezone)
    return {
        "timezone": timezone_name,
        "datetime": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
    }


# Chat Completions-compatible tool definitions.  Realtime uses the same
# logical definitions but removes the nested `function` object; see
# realtime_tool_definitions() below.
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "查询指定时区的当前日期和时间。用户询问现在几点、今天日期时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA 时区名称，例如 Asia/Shanghai、UTC。未提供时使用 Asia/Shanghai。",
                    }
                },
                "additionalProperties": False,
            },
        },
    }
]

TOOL_HANDLERS: dict[str, ToolHandler] = {
    "get_current_time": get_current_time,
}


def realtime_tool_definitions() -> list[dict[str, Any]]:
    """Return the flat tool shape expected by the Realtime API."""
    return [item["function"] | {"type": item["type"]} for item in TOOL_DEFINITIONS]


def tool_names() -> list[str]:
    return list(TOOL_HANDLERS)


async def execute_tool(name: str, arguments: Any, timeout_seconds: float = 10.0) -> dict[str, Any]:
    """Validate, execute, and normalize a model-requested tool call."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "error": f"未知工具：{name}"}

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "error": "工具参数不是有效的 JSON"}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "工具参数必须是 JSON 对象"}

    try:
        result = await asyncio.wait_for(handler(arguments), timeout=timeout_seconds)
        if not isinstance(result, dict):
            result = {"result": result}
        return {"ok": True, **result}
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"工具执行超时（{timeout_seconds:g} 秒）"}
    except ToolExecutionError as error:
        return {"ok": False, "error": str(error)}
    except Exception as error:  # Keep provider protocol stable on handler bugs.
        return {"ok": False, "error": f"工具执行失败：{type(error).__name__}"}
