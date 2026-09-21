"""Function-call regressions that do not require a running model server."""
import asyncio
import unittest
from unittest.mock import patch

from app import main
from app.tools import execute_tool


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeHttpClient:
    responses = []
    requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json):
        self.requests.append(json)
        return self.responses.pop(0)


class FunctionCallTests(unittest.IsolatedAsyncioTestCase):
    def test_realtime_function_call_events_are_normalized(self):
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
        result = await execute_tool("get_current_time", {"timezone": "UTC"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["timezone"], "UTC")
        self.assertIn("datetime", result)

    async def test_unknown_tool_is_rejected(self):
        result = await execute_tool("delete_everything", {})
        self.assertFalse(result["ok"])
        self.assertIn("未知工具", result["error"])

    async def test_lmstudio_executes_tool_then_continues_response(self):
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
