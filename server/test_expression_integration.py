"""表情词模块接入 REST、语音回合和取消流程的回归测试。

全部模型、语音与记忆依赖均使用替身，不访问外部服务或下载权重。
"""

import asyncio
import json
import unittest
from contextlib import ExitStack
from copy import deepcopy
from unittest.mock import AsyncMock, patch

import httpx

from app import main
from app.realtime_session import LocalRealtimeSession


class RecordingSocket:
    def __init__(self):
        self.events = []

    async def send_json(self, event):
        self.events.append(event)


class FakeModelResponse:
    def __init__(self, message):
        self.message = message

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": self.message}]}


class FakeModelClient:
    def __init__(self, messages):
        self.responses = [FakeModelResponse(message) for message in messages]
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json):
        # 后续工具回合会追加 messages，保留每次请求当时的独立快照。
        self.requests.append(deepcopy(json))
        return self.responses.pop(0)


class ExpressionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(main, "EXPRESSIONS_ENABLED", True))
        self.stack.enter_context(patch.object(main, "MEMORY_ENABLED", False))
        self.stack.enter_context(patch.object(main, "transcribe_pcm", return_value="今天终于完成任务了"))
        self.tts = self.stack.enter_context(patch.object(main, "synthesize_audio", return_value=b"\0" * 4800))

    async def run_turn(self, raw_reply, *, history=None, turn_id="expression-turn", request_camera=None):
        socket = RecordingSocket()
        history = [] if history is None else history
        with patch.object(main, "ask_lmstudio", new=AsyncMock(return_value=raw_reply)):
            await main.local_turn(
                socket, asyncio.Lock(), b"\0" * 256, turn_id, history,
                user_id="expression-test-user", request_camera=request_camera,
            )
        return socket.events, history

    def text_event(self, events):
        matches = [event for event in events if event["type"] == "response.text.done"]
        self.assertEqual(len(matches), 1, events)
        return matches[0]

    async def test_http_catalog_and_matching_endpoints(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            response = await client.get("/api/expressions")
            self.assertEqual(response.status_code, 200)
            catalog = response.json()
            self.assertEqual(len(catalog["expressions"]), 60)
            self.assertEqual(len({item["id"] for item in catalog["expressions"]}), 60)
            response = await client.post("/api/expressions/match", json={"word": "狂喜"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "MATCH")
            self.assertEqual(response.json()["expression_id"], "happy")
            for payload in ({}, {"word": []}, {"word": "开" * 65}):
                with self.subTest(payload=payload):
                    response = await client.post("/api/expressions/match", json=payload)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["status"], "INVALID_INPUT")

    async def test_rest_unwraps_expression_without_leaking_json_into_reply(self):
        raw = json.dumps({"text": "真为你高兴！", "expression": "狂喜"}, ensure_ascii=False)
        with patch.object(main, "ask_lmstudio", new=AsyncMock(return_value=raw)):
            response = await main.local_chat({"prompt": "我完成任务了"})
        self.assertEqual(response["reply"], "真为你高兴！")
        self.assertEqual(response["expression_word"], "狂喜")
        self.assertEqual(response["expression"]["expression_id"], "happy")

    async def test_voice_turn_strips_metadata_before_history_and_speech(self):
        raw = json.dumps({"text": "太好了，恭喜你！", "expression": "狂喜"}, ensure_ascii=False)
        events, history = await self.run_turn(raw)
        event = self.text_event(events)
        self.assertEqual(event["text"], "太好了，恭喜你！")
        self.assertEqual(event["expression_word"], "狂喜")
        self.assertEqual(event["expression"]["expression_id"], "happy")
        self.assertTrue(all(item["turn_id"] == "expression-turn" for item in events))
        self.tts.assert_called_once_with("太好了，恭喜你！")
        self.assertEqual(history[-1], {"role": "assistant", "content": "太好了，恭喜你！"})
        self.assertEqual(events[-1]["type"], "response.audio.done")

    async def test_unknown_word_preserves_reply_but_does_not_invent_expression(self):
        raw = json.dumps({"text": "我听到了。", "expression": "天气"}, ensure_ascii=False)
        events, history = await self.run_turn(raw)
        event = self.text_event(events)
        self.assertEqual(event["expression_word"], "天气")
        self.assertEqual(event["expression"]["status"], "NO_MATCH")
        self.assertIsNone(event["expression"]["expression_id"])
        self.tts.assert_called_once_with("我听到了。")
        self.assertEqual(history[-1]["content"], "我听到了。")

    async def test_legacy_plain_text_is_not_reclassified_from_its_contents(self):
        # 回复文本带“开心”，也不能被当成上游表情词。
        events, _ = await self.run_turn("祝你每天开心。")
        event = self.text_event(events)
        self.assertEqual(event["text"], "祝你每天开心。")
        self.assertIsNone(event["expression_word"])
        self.assertEqual(event["expression"]["status"], "NO_MATCH")
        self.tts.assert_called_once_with("祝你每天开心。")

    async def test_malformed_envelope_uses_safe_spoken_fallback(self):
        raw = '{"text": "你好", "expression": "开心"'
        events, history = await self.run_turn(raw)
        event = self.text_event(events)
        fallback = "抱歉，回复格式异常，请再试一次。"
        self.assertEqual(event["text"], fallback)
        self.assertEqual(event["expression"]["status"], "NO_MATCH")
        self.assertIsNone(event["expression_word"])
        self.assertEqual(history[-1]["content"], fallback)
        self.tts.assert_called_once_with(fallback)

    async def test_disabling_feature_retains_old_payload_and_speech(self):
        with patch.object(main, "EXPRESSIONS_ENABLED", False):
            events, _ = await self.run_turn("普通回答。")
            with patch.object(main, "ask_lmstudio", new=AsyncMock(return_value="普通回答。")):
                rest = await main.local_chat({"prompt": "你好"})
        event = self.text_event(events)
        self.assertNotIn("expression", event)
        self.assertNotIn("expression_word", event)
        self.assertEqual(rest, {"reply": "普通回答。"})
        self.tts.assert_called_once_with("普通回答。")

    async def test_camera_final_reply_is_unwrapped_like_normal_reply(self):
        photo_request = AsyncMock(return_value={"bytes": b"jpeg", "mime_type": "image/jpeg"})
        raw = json.dumps({"text": "我看到了一张桌子。", "expression": "好奇"}, ensure_ascii=False)
        with patch.object(main, "analyze_image", new=AsyncMock(return_value=raw)) as analyze:
            events, history = await self.run_turn(main.CameraCaptureIntent(), request_camera=photo_request)
        photo_request.assert_awaited_once_with("expression-turn", "describe_scene")
        analyze.assert_awaited_once_with("今天终于完成任务了", b"jpeg", "image/jpeg")
        event = self.text_event(events)
        self.assertEqual(event["expression"]["expression_id"], "curious")
        self.assertEqual(event["text"], "我看到了一张桌子。")
        self.tts.assert_called_once_with("我看到了一张桌子。")
        self.assertEqual(history[-1]["content"], "我看到了一张桌子。")

    async def test_memory_uses_original_user_text_and_clean_reply(self):
        raw = json.dumps({"text": "我记得你喜欢蓝色。", "expression": "微笑"}, ensure_ascii=False)
        with patch.object(main, "MEMORY_ENABLED", True), \
             patch.object(main.memory_service, "handle_explicit_command", new=AsyncMock(return_value=None)), \
             patch.object(main.memory_service, "relevant", new=AsyncMock(return_value=[])), \
             patch.object(main.memory_service, "extract_explicit", new=AsyncMock()) as extract:
            events, history = await self.run_turn(raw)
        extract.assert_awaited_once_with("expression-test-user", "今天终于完成任务了", "expression-turn")
        self.assertEqual(history[-1]["content"], "我记得你喜欢蓝色。")
        self.assertEqual(self.text_event(events)["expression"]["expression_id"], "smile")

    async def test_handled_memory_command_does_not_call_model_or_guess_expression(self):
        with patch.object(main, "MEMORY_ENABLED", True), \
             patch.object(main.memory_service, "handle_explicit_command", new=AsyncMock(return_value={
                 "handled": True, "reply": "已经记住你喜欢开心的表情。",
             })), \
             patch.object(main, "ask_lmstudio", new=AsyncMock()) as model:
            socket = RecordingSocket()
            await main.local_turn(socket, asyncio.Lock(), b"voice", "memory-turn", [])
        model.assert_not_awaited()
        event = self.text_event(socket.events)
        self.assertEqual(event["text"], "已经记住你喜欢开心的表情。")
        self.assertEqual(event["expression"]["status"], "NO_MATCH")
        self.assertIsNone(event["expression_word"])

    async def test_tool_round_returns_final_envelope_and_keeps_tool_protocol(self):
        raw = json.dumps({"text": "现在是十点。", "expression": "平静"}, ensure_ascii=False)
        client = FakeModelClient([
            {"content": None, "tool_calls": [{"id": "call-time", "type": "function", "function": {
                "name": "get_current_time", "arguments": "{}",
            }}]},
            {"content": raw},
        ])
        with patch.object(main.httpx, "AsyncClient", return_value=client), \
             patch.object(main, "FUNCTION_CALLS_ENABLED", True), \
             patch.object(main, "execute_tool", new=AsyncMock(return_value={"ok": True, "time": "10:00"})):
            result = await main.ask_lmstudio("现在几点")
        self.assertEqual(result, raw)
        self.assertEqual(len(client.requests), 2)
        self.assertIn("expression", client.requests[0]["messages"][0]["content"])
        self.assertEqual(client.requests[1]["messages"][-1]["role"], "tool")
        self.assertEqual(client.requests[1]["messages"][-1]["tool_call_id"], "call-time")

    async def test_disabled_model_request_does_not_include_expression_contract(self):
        client = FakeModelClient([{"content": "普通回答。"}])
        with patch.object(main.httpx, "AsyncClient", return_value=client), \
             patch.object(main, "EXPRESSIONS_ENABLED", False), \
             patch.object(main, "FUNCTION_CALLS_ENABLED", False):
            result = await main.ask_lmstudio("你好")
        self.assertEqual(result, "普通回答。")
        self.assertNotIn("expression", client.requests[0]["messages"][0]["content"])

    async def test_cancelled_turn_never_emits_expression_and_next_turn_recovers(self):
        socket = RecordingSocket()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def pending_reply(*args):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        async def handle(pcm, turn_id, history, user_id, request_camera):
            await main.local_turn(socket, asyncio.Lock(), pcm, turn_id, history, user_id, request_camera)

        session = LocalRealtimeSession(
            object(), socket.send_json, handle, default_user_id="expression-test-user", max_history_turns=3,
        )
        try:
            with patch.object(main, "ask_lmstudio", new=AsyncMock(side_effect=pending_reply)):
                await session.commit("old")
                await asyncio.wait_for(started.wait(), 2)
                await session.cancel_current_turn()
            self.assertTrue(cancelled.is_set())
            self.assertFalse(any(event["type"] == "response.text.done" for event in socket.events))
            self.assertEqual(session.history, [])
            self.tts.assert_not_called()
            raw = json.dumps({"text": "新的回答。", "expression": "哈哈"}, ensure_ascii=False)
            with patch.object(main, "ask_lmstudio", new=AsyncMock(return_value=raw)):
                await session.commit("new")
                await asyncio.wait_for(session.current_turn_task, 2)
            event = self.text_event(socket.events)
            self.assertEqual(event["turn_id"], "new")
            self.assertEqual(event["expression"]["expression_id"], "happy")
        finally:
            await session.cancel_current_turn(notify=False)


if __name__ == "__main__":
    unittest.main()
