"""相机意图、异步拍照协议和视觉请求的回归测试。"""

import asyncio
import base64
import unittest
from unittest.mock import patch

from app import main
from app.realtime_session import CameraCaptureError, LocalRealtimeSession


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeHttpClient:
    def __init__(self, response):
        self.response = response
        self.payloads = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json):
        self.payloads.append(json)
        return self.response


class CameraTests(unittest.IsolatedAsyncioTestCase):
    async def test_camera_tool_call_becomes_intent_marker(self):
        response = FakeResponse({"choices": [{"message": {
            "tool_calls": [{"id": "camera-1", "type": "function", "function": {
                "name": "request_camera_capture", "arguments": '{"reason":"describe_scene"}',
            }}],
        }}]})
        client = FakeHttpClient(response)
        token = main._turn_context.set({"camera_intent_enabled": True})
        try:
            with patch.object(main.httpx, "AsyncClient", return_value=client):
                result = await main.ask_lmstudio("描述眼前景象")
        finally:
            main._turn_context.reset(token)
        self.assertIsInstance(result, main.CameraCaptureIntent)
        self.assertEqual(result.reason, "describe_scene")
        self.assertEqual(client.payloads[0]["tools"][0]["function"]["name"], "request_camera_capture")

    async def test_vision_request_contains_text_and_data_url(self):
        client = FakeHttpClient(FakeResponse({"choices": [{"message": {"content": "看到了桌子。"}}]}))
        with patch.object(main.httpx, "AsyncClient", return_value=client):
            result = await main.analyze_image("请描述眼前景象", b"jpeg-bytes")
        self.assertEqual(result, "看到了桌子。")
        content = client.payloads[0]["messages"][1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "请描述眼前景象"})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    async def test_session_rejects_stale_photo_and_accepts_matching_photo(self):
        sent = []
        handler_started = asyncio.Event()
        handler_finished = asyncio.Event()

        async def send(event):
            sent.append(event)

        async def handler(pcm, turn_id, history, user_id, request_camera):
            handler_started.set()
            photo = await request_camera(turn_id, "describe_scene")
            self.assertEqual(photo["bytes"], b"jpeg")
            handler_finished.set()

        session = LocalRealtimeSession(
            object(), send, handler, default_user_id="test", max_history_turns=3,
            camera_timeout_seconds=2,
        )
        await session.commit("turn-1")
        await asyncio.wait_for(handler_started.wait(), 1)
        request = sent[-1]
        await session.handle_event({
            "type": "camera.capture.result", "turn_id": "old", "request_id": request["request_id"],
            "mime_type": "image/jpeg", "image_base64": base64.b64encode(b"stale").decode(),
        })
        self.assertFalse(handler_finished.is_set())
        await session.handle_event({
            "type": "camera.capture.result", "turn_id": "turn-1", "request_id": request["request_id"],
            "mime_type": "image/jpeg", "image_base64": base64.b64encode(b"jpeg").decode(),
        })
        await asyncio.wait_for(handler_finished.wait(), 1)
        await session.cancel_current_turn(notify=False)

    async def test_invalid_photo_is_reported_to_waiting_turn(self):
        sent = []

        async def send(event):
            sent.append(event)

        async def handler(pcm, turn_id, history, user_id, request_camera):
            with self.assertRaises(CameraCaptureError):
                await request_camera(turn_id, "describe_scene")

        session = LocalRealtimeSession(
            object(), send, handler, default_user_id="test", max_history_turns=3,
            camera_timeout_seconds=2,
        )
        await session.commit("turn-2")
        while not sent:
            await asyncio.sleep(0)
        request = sent[-1]
        await session.handle_event({
            "type": "camera.capture.result", "turn_id": "turn-2", "request_id": request["request_id"],
            "mime_type": "image/jpeg", "image_base64": "%%%not-base64%%%",
        })
        task = session.current_turn_task
        if task is not None:
            await asyncio.wait_for(task, 1)
        await session.cancel_current_turn(notify=False)


if __name__ == "__main__":
    unittest.main()
