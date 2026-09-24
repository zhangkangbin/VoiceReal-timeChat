"""无需启动模型服务的函数调用回归测试，覆盖事件归一化和工具执行链。"""
import asyncio
import unittest
from unittest.mock import patch

from app import main
from app.tools import execute_tool


class FakeResponse:
    """模拟 HTTP 响应，提供客户端测试所需的最小接口。"""
    def __init__(self, payload):
        """保存预置 JSON 负载，供测试读取。"""
        self.payload = payload

    def raise_for_status(self):
        """模拟成功 HTTP 响应的状态检查。"""
        pass

    def json(self):
        """返回预置响应负载。"""
        return self.payload


class FakeHttpClient:
    """按预设响应记录请求的异步 HTTP 客户端替身。"""
    responses = []
    requests = []

    async def __aenter__(self):
        """进入异步客户端上下文并返回自身。"""
        return self

    async def __aexit__(self, *args):
        """离开异步客户端上下文，不抑制异常。"""
        return False

    async def post(self, url, json):
        """记录请求体并按顺序弹出模拟响应。"""
        self.requests.append(json)
        return self.responses.pop(0)


class FunctionCallTests(unittest.IsolatedAsyncioTestCase):
    """验证 Realtime 事件、工具错误处理及 LM Studio 多轮工具调用。"""
    def test_realtime_function_call_events_are_normalized(self):
        """确认两种 Realtime 事件形态都归一化为同一三元组。"""
        self.assertEqual(
            main.realtime_function_call({
                "type": "response.function_call_arguments.done",
                "call_id": "call-1",
                "name": "get_current_time",
                "arguments": "{}",
            }),
            ("call-1", "get_current_time", "{}"),
        )
        self.assertEqual(
            main.realtime_function_call({
                "type": "response.output_item.done",
                "item": {"type": "function_call", "call_id": "call-2", "name": "get_current_time", "arguments": "{}"},
            }),
            ("call-2", "get_current_time", "{}"),
        )

    async def test_current_time_tool_returns_structured_result(self):
        """确认时钟工具返回成功标志、时区和结构化日期时间字段。"""
        result = await execute_tool("get_current_time", {"timezone": "UTC"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["timezone"], "UTC")
        self.assertIn("datetime", result)

    async def test_unknown_tool_is_rejected(self):
        """确认未注册工具被拒绝，并返回可识别的中文错误。"""
        result = await execute_tool("delete_everything", {})
        self.assertFalse(result["ok"])
        self.assertIn("未知工具", result["error"])

    async def test_lmstudio_executes_tool_then_continues_response(self):
        """确认模型先请求工具，再携带 tool 消息继续生成最终回答。"""
        FakeHttpClient.responses = [
            FakeResponse({"choices": [{"message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "get_current_time", "arguments": "{\"timezone\":\"UTC\"}"},
                }],
            }}]}),
            FakeResponse({"choices": [{"message": {
                "role": "assistant",
                "content": "现在是 UTC 时间。",
            }}]}),
        ]
        FakeHttpClient.requests = []
        with patch.object(main.httpx, "AsyncClient", return_value=FakeHttpClient()), \
             patch.object(main, "FUNCTION_CALLS_ENABLED", True):
            answer = await main.ask_lmstudio("现在几点？")

        self.assertEqual(answer, "现在是 UTC 时间。")
        self.assertEqual(len(FakeHttpClient.requests), 2)
        self.assertIn("tools", FakeHttpClient.requests[0])
        second_messages = FakeHttpClient.requests[1]["messages"]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["tool_call_id"], "call-1")


if __name__ == "__main__":
    unittest.main()
