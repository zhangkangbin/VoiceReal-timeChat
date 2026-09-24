"""应用侧拥有的函数工具及其统一执行入口。

模型只能请求本模块注册的工具。服务器负责校验并执行请求，因此 API 密钥和
私有业务逻辑无需发送到 Android 客户端。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone as dt_timezone
import json
import os
import re
import time
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from .memory import memory_delete_tool, memory_list_tool, memory_save_tool


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class ToolExecutionError(RuntimeError):
    """可安全展示给用户的工具执行失败。"""


_NETWORK_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_TROY_OUNCE_GRAMS = 31.1034768


def _network_timeout() -> float:
    """读取网络工具超时；非法配置回退到安全的短超时。"""
    try:
        return max(1.0, float(os.getenv("NETWORK_TIMEOUT_SECONDS", "8")))
    except ValueError:
        return 8.0


def _network_cache_seconds() -> float:
    """读取网络工具缓存时长；设为 0 可关闭缓存。"""
    try:
        return max(0.0, float(os.getenv("NETWORK_CACHE_SECONDS", "300")))
    except ValueError:
        return 300.0


def _cached_result(key: str) -> dict[str, Any] | None:
    ttl = _network_cache_seconds()
    item = _NETWORK_CACHE.get(key)
    if ttl <= 0 or item is None:
        return None
    timestamp, value = item
    if time.monotonic() - timestamp >= ttl:
        _NETWORK_CACHE.pop(key, None)
        return None
    return dict(value)


def _store_cached_result(key: str, value: dict[str, Any]) -> dict[str, Any]:
    if _network_cache_seconds() > 0:
        _NETWORK_CACHE[key] = (time.monotonic(), dict(value))
    return value


async def _request_json(method: str, url: str, *, params: dict[str, Any] | None = None,
                        json_body: dict[str, Any] | None = None,
                        headers: dict[str, str] | None = None) -> dict[str, Any]:
    """执行受限 JSON 请求，并把底层网络异常转换为可朗读的工具错误。"""
    if not url.startswith(("https://", "http://")):
        raise ToolExecutionError("网络接口地址必须使用 HTTP 或 HTTPS")
    try:
        async with httpx.AsyncClient(timeout=_network_timeout(), trust_env=False) as client:
            response = await client.request(method, url, params=params, json=json_body, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException as error:
        raise ToolExecutionError("网络查询超时，请稍后再试") from error
    except httpx.HTTPStatusError as error:
        raise ToolExecutionError(f"网络服务返回 HTTP {error.response.status_code}") from error
    except (httpx.RequestError, ValueError) as error:
        raise ToolExecutionError("网络查询暂时不可用，请稍后再试") from error
    if not isinstance(payload, dict):
        raise ToolExecutionError("网络服务返回格式无效")
    return payload


def _number(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ToolExecutionError(f"网络结果缺少有效的{name}") from error


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else None


def _first(value: Any, name: str) -> Any:
    if not isinstance(value, list) or not value:
        raise ToolExecutionError(f"网络结果缺少{name}")
    return value[0]


async def get_current_time(arguments: dict[str, Any]) -> dict[str, Any]:
    """按 IANA 时区返回当前日期、时间和星期，并提供 Windows 回退时区。"""
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


_WEATHER_CODES = {
    0: "晴",
    1: "基本晴朗",
    2: "局部多云",
    3: "阴天",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "大毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "大阵雪",
    95: "雷雨",
    96: "雷雨伴小冰雹",
    99: "雷雨伴大冰雹",
}


async def get_weather(arguments: dict[str, Any]) -> dict[str, Any]:
    """查询城市今天的天气，默认使用 Open-Meteo 的地理编码和预报接口。"""
    city = str(arguments.get("city") or os.getenv("DEFAULT_WEATHER_CITY", "")).strip()
    if not city:
        raise ToolExecutionError("请提供要查询天气的城市")
    cache_key = f"weather:{city.casefold()}"
    cached = _cached_result(cache_key)
    if cached is not None:
        return cached

    geocoding_url = os.getenv("WEATHER_GEOCODING_URL", "https://geocoding-api.open-meteo.com/v1/search")
    geocoding = await _request_json("GET", geocoding_url, params={
        "name": city,
        "count": 1,
        "language": "zh",
        "format": "json",
    })
    locations = geocoding.get("results")
    location = _first(locations, "城市")
    if not isinstance(location, dict):
        raise ToolExecutionError("地理编码结果格式无效")
    latitude = _number(location.get("latitude"), "纬度")
    longitude = _number(location.get("longitude"), "经度")

    forecast_url = os.getenv("WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast")
    forecast = await _request_json("GET", forecast_url, params={
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset",
        "forecast_days": 1,
        "timezone": "auto",
    })
    current = forecast.get("current")
    daily = forecast.get("daily")
    if not isinstance(current, dict) or not isinstance(daily, dict):
        raise ToolExecutionError("天气服务返回格式无效")
    date = _first(daily.get("time"), "日期")
    weather_code = int(_number(current.get("weather_code"), "天气代码"))
    result = {
        "city": str(location.get("name") or city),
        "country": str(location.get("country") or ""),
        "date": str(date),
        "condition": _WEATHER_CODES.get(weather_code, "未知天气"),
        "temperature_c": _number(current.get("temperature_2m"), "当前温度"),
        "feels_like_c": _optional_number(current.get("apparent_temperature")),
        "humidity_pct": _optional_number(current.get("relative_humidity_2m")),
        "wind_speed_kmh": _optional_number(current.get("wind_speed_10m")),
        "high_c": _number(_first(daily.get("temperature_2m_max"), "最高温度"), "最高温度"),
        "low_c": _number(_first(daily.get("temperature_2m_min"), "最低温度"), "最低温度"),
        "precipitation_probability_pct": _optional_number(_optional_first(daily.get("precipitation_probability_max"))),
        "sunrise": str(_optional_first(daily.get("sunrise")) or ""),
        "sunset": str(_optional_first(daily.get("sunset")) or ""),
        "source": "Open-Meteo",
        "source_url": forecast_url,
        "retrieved_at": datetime.now(dt_timezone.utc).isoformat(timespec="seconds"),
    }
    # Keep short aliases for the model prompt while retaining explicit units
    # for clients that consume the structured tool result.
    result["temperature"] = result["temperature_c"]
    result["high"] = result["high_c"]
    result["low"] = result["low_c"]
    return _store_cached_result(cache_key, result)


async def get_gold_price(arguments: dict[str, Any]) -> dict[str, Any]:
    """查询一份现货黄金参考价，默认返回人民币/克的 24K 价格。"""
    currency = str(arguments.get("currency") or "CNY").strip().upper()
    unit = str(arguments.get("unit") or "gram").strip().lower()
    if not re.fullmatch(r"[A-Z]{3}", currency):
        raise ToolExecutionError("币种必须是三位字母，例如 CNY 或 USD")
    unit_aliases = {"g": "gram", "克": "gram", "oz": "oz", "ounce": "oz", "盎司": "oz", "kg": "kg"}
    unit = unit_aliases.get(unit, unit)
    if unit not in {"gram", "oz", "kg"}:
        raise ToolExecutionError("单位必须是 gram、kg 或 oz")
    cache_key = f"gold:{currency}:{unit}"
    cached = _cached_result(cache_key)
    if cached is not None:
        return cached
    url_template = os.getenv("GOLD_PRICE_API_URL", "https://api.gold-api.com/price/XAU/{currency}")
    try:
        url = url_template.format(currency=currency)
    except (KeyError, ValueError):
        raise ToolExecutionError("金价接口地址格式无效")
    payload = await _request_json("GET", url)
    price_per_ounce = _number(payload.get("price"), "金价")
    price = {
        "oz": price_per_ounce,
        "gram": price_per_ounce / _TROY_OUNCE_GRAMS,
        "kg": price_per_ounce * 1000 / _TROY_OUNCE_GRAMS,
    }[unit]
    result = {
        "price": price,
        "currency": currency,
        "unit": unit,
        "metal": "XAU",
        "purity": "24K spot reference",
        "price_per_troy_ounce": price_per_ounce,
        "timestamp": str(payload.get("updatedAt") or payload.get("updated_at") or payload.get("timestamp") or datetime.now(dt_timezone.utc).isoformat(timespec="seconds")),
        "source": "Gold API",
        "source_url": url,
    }
    result["updated_at"] = result["timestamp"]
    return _store_cached_result(cache_key, result)


async def web_search(arguments: dict[str, Any]) -> dict[str, Any]:
    """执行服务端网络搜索；默认适配 Brave Search API，也支持兼容的自定义端点。"""
    query = str(arguments.get("query") or "").strip()
    freshness = str(arguments.get("freshness") or "").strip().lower()
    if not query:
        raise ToolExecutionError("请提供要搜索的问题")
    if len(query) > 500:
        raise ToolExecutionError("搜索内容过长")
    cache_key = f"search:{query.casefold()}:{freshness}"
    cached = _cached_result(cache_key)
    if cached is not None:
        return cached
    configured_url = os.getenv("SEARCH_API_URL", "https://api.search.brave.com/res/v1/web/search").strip()
    api_key = os.getenv("SEARCH_API_KEY", "").strip()
    if not api_key:
        raise ToolExecutionError("未配置 SEARCH_API_KEY，暂时无法执行网络搜索")
    freshness_param = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}.get(freshness)
    params: dict[str, Any] = {
        "q": query,
        "count": 5,
        "country": "CN",
        "search_lang": "zh",
        "ui_lang": "zh-CN",
    }
    if freshness_param:
        params["freshness"] = freshness_param
    payload = await _request_json("GET", configured_url, params=params, headers={
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
    })
    items = ((payload.get("web") or {}).get("results") or [])
    if not isinstance(items, list):
        raise ToolExecutionError("搜索服务返回格式无效")
    results = [{
        "title": str(item.get("title") or ""),
        "url": str(item.get("url") or ""),
        "snippet": str(item.get("description") or "")[:600],
        "published": str(item.get("age") or item.get("published") or ""),
    } for item in items if isinstance(item, dict)]
    source = "Brave Search"
    if not results:
        raise ToolExecutionError("没有找到相关网络结果")
    return _store_cached_result(cache_key, {
        "query": query,
        "freshness": freshness or "any",
        "source": source,
        "results": results,
        "retrieved_at": datetime.now(dt_timezone.utc).isoformat(timespec="seconds"),
    })


# Chat Completions-compatible tool definitions.  Realtime uses the same
# logical definitions but removes the nested `function` object; see
# realtime_tool_definitions() below.
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "request_camera_capture",
            "description": "当用户要求描述眼前、面前或相机看到的景象时，请求手机后置相机拍照。只在用户明确需要看现场画面时调用。拍照完成后服务端会继续分析照片并用语音回答。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "拍照原因，例如 describe_scene。",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市今天的天气、温度、体感温度、风速和降水概率。用户询问今天或当前天气时使用；没有城市时先询问城市。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，例如上海、北京或 Tokyo。"},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gold_price",
            "description": "查询当前24K现货黄金参考价。默认返回人民币每克；用户询问今天金价、黄金价格时使用，并说明这是现货参考价及数据时间。",
            "parameters": {
                "type": "object",
                "properties": {
                    "currency": {"type": "string", "description": "三位货币代码，例如 CNY 或 USD；默认 CNY。"},
                    "unit": {"type": "string", "enum": ["gram", "kg", "oz"], "description": "计价单位；默认 gram。"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索近期网络信息，用于天气和金价以外的事实查询、新闻或需要最新资料的问题。不要用它代替 get_weather 或 get_gold_price。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜索的完整问题或关键词。"},
                    "freshness": {"type": "string", "enum": ["day", "week", "month", "year"], "description": "结果时效偏好，可不填。"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_save",
            "description": "保存用户明确表达的长期偏好、习惯或事实。只有用户明确说‘记住’、表达稳定喜好或习惯时才调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["preference", "habit", "fact", "constraint"]},
                    "key": {"type": "string", "description": "稳定的记忆键，例如 likes、response_style。"},
                    "value": {"type": "string", "description": "要保存的简短记忆内容。"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "importance": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["type", "key", "value"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "删除用户明确要求忘记的记忆。清除全部记忆前必须确认用户确实要求清除全部。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要删除的记忆关键词。"},
                    "clear_all": {"type": "boolean", "description": "是否清除全部长期记忆。"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_list",
            "description": "查看当前用户已保存的长期记忆。用户询问‘你记得我什么’时调用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]

TOOL_HANDLERS: dict[str, ToolHandler] = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "get_gold_price": get_gold_price,
    "web_search": web_search,
    "memory_save": memory_save_tool,
    "memory_delete": memory_delete_tool,
    "memory_list": memory_list_tool,
}


def realtime_tool_definitions() -> list[dict[str, Any]]:
    """转换为 Realtime API 所需的扁平工具定义结构。"""
    return [item["function"] | {"type": item["type"]} for item in TOOL_DEFINITIONS]


def tool_names() -> list[str]:
    """返回已注册工具名称，供校验或日志使用。"""
    return list(TOOL_HANDLERS)


async def execute_tool(name: str, arguments: Any, timeout_seconds: float | None = None,
                       context: dict[str, Any] | None = None) -> dict[str, Any]:
    """校验模型参数、执行工具，并把成功与失败统一为字典结果。"""
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
        effective_timeout = timeout_seconds
        if effective_timeout is None:
            effective_timeout = max(10.0, _network_timeout() * 3) if name in {"get_weather", "get_gold_price", "web_search"} else 10.0
        handler_arguments = dict(arguments)
        if context:
            # Internal context is added only after model JSON validation and
            # is never exposed in the public tool schema.
            handler_arguments["__context"] = context
        result = await asyncio.wait_for(handler(handler_arguments), timeout=effective_timeout)
        if not isinstance(result, dict):
            result = {"result": result}
        return {"ok": True, **result}
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"工具执行超时（{effective_timeout:g} 秒）"}
    except ToolExecutionError as error:
        return {"ok": False, "error": str(error)}
    except Exception as error:  # Keep provider protocol stable on handler bugs.
        return {"ok": False, "error": f"工具执行失败：{type(error).__name__}"}
