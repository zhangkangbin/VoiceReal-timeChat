"""网络工具的离线回归测试。

所有 HTTP 请求都由 fake client 拦截，测试不会依赖真实天气、金价或搜索服务。
"""

import os
import unittest
from unittest.mock import patch

import httpx

from app import tools


class FakeResponse:
    """提供网络工具所需的最小 HTTP 响应接口。"""

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeHttpClient:
    """按顺序返回预置响应，并记录请求参数。"""

    responses = []
    requests = []
    error = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)


class NetworkToolTests(unittest.IsolatedAsyncioTestCase):
    """验证工具注册、正常结果归一化和网络错误处理。"""

    def setUp(self):
        FakeHttpClient.responses = []
        FakeHttpClient.requests = []
        FakeHttpClient.error = None
        # 每个测试都独立访问 fake 服务，避免模块缓存掩盖请求行为。
        tools._NETWORK_CACHE.clear()

    def _patch_client(self):
        return patch.object(tools.httpx, "AsyncClient", return_value=FakeHttpClient())

    def test_network_tools_are_registered_and_exposed_to_model(self):
        names = tools.tool_names()
        self.assertIn("get_weather", names)
        self.assertIn("get_gold_price", names)
        self.assertIn("web_search", names)
        definition_names = {
            item["function"]["name"] for item in tools.TOOL_DEFINITIONS
        }
        self.assertTrue({"get_weather", "get_gold_price", "web_search"} <= definition_names)

    async def test_weather_uses_geocoding_then_forecast_and_normalizes_result(self):
        FakeHttpClient.responses = [
            FakeResponse({
                "results": [{"name": "Shanghai", "latitude": 31.23, "longitude": 121.47}],
            }),
            FakeResponse({
                "current": {
                    "temperature_2m": 26.4,
                    "apparent_temperature": 27.0,
                    "relative_humidity_2m": 65,
                    "wind_speed_10m": 10.2,
                    "weather_code": 1,
                },
                "daily": {
                    "time": ["2026-09-25"],
                    "temperature_2m_max": [29.0],
                    "temperature_2m_min": [22.0],
                    "precipitation_probability_max": [20],
                    "weather_code": [1],
                    "sunrise": ["2026-09-25T05:30"],
                    "sunset": ["2026-09-25T18:00"],
                },
            }),
        ]
        with patch.dict(os.environ, {"NETWORK_CACHE_SECONDS": "0"}, clear=False), self._patch_client():
            result = await tools.execute_tool("get_weather", {"city": "上海"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["city"], "Shanghai")
        self.assertEqual(result["date"], "2026-09-25")
        self.assertEqual(result["temperature_c"], 26.4)
        self.assertEqual(result["high_c"], 29.0)
        self.assertEqual(result["low_c"], 22.0)
        self.assertIn("source", result)
        self.assertEqual(len(FakeHttpClient.requests), 2)
        self.assertIn("geocoding-api.open-meteo.com", FakeHttpClient.requests[0][1])
        self.assertIn("api.open-meteo.com", FakeHttpClient.requests[1][1])

    async def test_gold_price_converts_troy_ounce_to_gram(self):
        FakeHttpClient.responses = [
            FakeResponse({
                "price": 2480.0,
                "updatedAt": "2026-09-25T08:00:00Z",
            }),
        ]
        with patch.dict(os.environ, {"NETWORK_CACHE_SECONDS": "0"}, clear=False), self._patch_client():
            result = await tools.execute_tool(
                "get_gold_price", {"currency": "USD", "unit": "gram"}
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["unit"], "gram")
        self.assertAlmostEqual(result["price"], 2480.0 / 31.1034768, places=6)
        self.assertEqual(result["timestamp"], "2026-09-25T08:00:00Z")
        self.assertAlmostEqual(result["price_per_troy_ounce"], 2480.0)
        self.assertIn("gold-api.com", FakeHttpClient.requests[0][1])

    async def test_web_search_uses_brave_api_and_returns_results(self):
        FakeHttpClient.responses = [
            FakeResponse({
                "web": {"results": [{
                    "title": "天气预报",
                    "description": "今天晴朗。",
                    "url": "https://example.test/weather",
                    "published": "2026-09-25",
                }]},
            }),
        ]
        env = {
            "SEARCH_API_URL": "https://search.example.test/v1/search",
            "SEARCH_API_KEY": "test-key",
            "NETWORK_CACHE_SECONDS": "0",
        }
        with patch.dict(os.environ, env, clear=False), self._patch_client():
            result = await tools.execute_tool(
                "web_search", {"query": "今天的天气", "freshness": "day"}
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["query"], "今天的天气")
        self.assertEqual(result["results"][0]["title"], "天气预报")
        self.assertEqual(result["results"][0]["url"], "https://example.test/weather")
        method, url, request_kwargs = FakeHttpClient.requests[0]
        self.assertEqual(method, "GET")
        self.assertEqual(url, "https://search.example.test/v1/search")
        self.assertEqual(request_kwargs["headers"]["X-Subscription-Token"], "test-key")
        self.assertEqual(request_kwargs["params"]["freshness"], "pd")

    async def test_network_timeout_is_returned_as_safe_tool_error(self):
        FakeHttpClient.error = httpx.ReadTimeout("simulated timeout")
        with patch.dict(os.environ, {"NETWORK_CACHE_SECONDS": "0"}, clear=False), self._patch_client():
            result = await tools.execute_tool("get_gold_price", {"currency": "USD"})

        self.assertFalse(result["ok"])
        self.assertIn("超时", result["error"])


if __name__ == "__main__":
    unittest.main()
