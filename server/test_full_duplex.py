"""Deterministic full-duplex protocol regressions (no model download required).

Run from server: .venv/Scripts/python.exe -m unittest -v test_full_duplex
"""
import asyncio
import base64
import json
import threading
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import WebSocketDisconnect
from app import main


class FakeSocket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()

    async def accept(self):
        pass

    async def receive_text(self):
        event = await self.incoming.get()
        if event is None:
            raise WebSocketDisconnect()
        return json.dumps(event)

    async def send_json(self, event):
        await self.outgoing.put(event)

    async def send(self, event_type, **values):
        await self.incoming.put({"type": event_type, **values})

    async def next(self):
        return await asyncio.wait_for(self.outgoing.get(), 3)

    async def until(self, event_type):
        events = []
        while True:
            event = await self.next()
            events.append(event)
            if event["type"] == event_type:
                return events


class FullDuplexTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patches = [
            patch.object(main, "provider_name", return_value="lmstudio"),
            patch.object(main, "transcribe_pcm", return_value="测试语音"),
            patch.object(main, "ask_lmstudio", new=AsyncMock(return_value="本地回答")),
            patch.object(main, "synthesize_sapi", return_value=b"\0" * 9600),
        ]
        for item in self.patches:
            item.start()
        self.ws = FakeSocket()
        self.task = asyncio.create_task(main.realtime(self.ws))
        self.assertEqual((await self.ws.next())["type"], "server.ready")

    async def asyncTearDown(self):
        await self.ws.incoming.put(None)
        try:
            await asyncio.wait_for(self.task, 3)
        finally:
            for item in reversed(self.patches):
                item.stop()

    async def commit(self, turn_id, pcm=b"\0" * 1024):
        await self.ws.send("input_audio_buffer.append", audio=base64.b64encode(pcm).decode())
        await self.ws.send("input_audio_buffer.commit", turn_id=turn_id)

    async def test_normal_response_echoes_client_turn_id(self):
        await self.commit("client-1")
        events = await self.ws.until("response.audio.done")
        self.assertEqual([e["type"] for e in events], [
            "conversation.transcript", "response.text.done", "response.audio.delta",
            "response.audio.delta", "response.audio.done"])
        self.assertTrue(all(e["turn_id"] == "client-1" for e in events))

    async def test_barge_in_during_llm_then_next_turn_completes(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def slow_llm(text):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        with patch.object(main, "ask_lmstudio", side_effect=slow_llm):
            await self.commit("old")
            await asyncio.wait_for(started.wait(), 3)
            await self.ws.send("input_audio_buffer.speech_started")
            events = await self.ws.until("response.cancelled")
            self.assertEqual(events[-1]["turn_id"], "old")
            self.assertTrue(cancelled.is_set())
            self.assertFalse(any(e["type"] == "response.audio.delta" for e in events))
        await self.commit("new")
        events = await self.ws.until("response.audio.done")
        self.assertTrue(all(e["turn_id"] == "new" for e in events))

    async def test_cancel_ack_even_when_audio_already_sent(self):
        await self.commit("buffered-on-phone")
        await self.ws.until("response.audio.done")
        await self.ws.send("input_audio_buffer.speech_started")
        self.assertEqual(await self.ws.next(), {"type": "response.cancelled", "turn_id": "buffered-on-phone"})

    async def test_new_speech_and_discard_clear_uncommitted_audio(self):
        await self.ws.send("input_audio_buffer.append", audio=base64.b64encode(b"stale").decode())
        await self.ws.send("input_audio_buffer.speech_started")
        await self.commit("fresh", b"fresh")
        await self.ws.until("response.audio.done")
        main.transcribe_pcm.assert_called_once_with(b"fresh")
        await self.ws.send("input_audio_buffer.append", audio=base64.b64encode(b"noise").decode())
        await self.ws.send("input_audio_buffer.clear")
        await self.commit("after-discard", b"voice")
        await self.ws.until("response.audio.done")
        main.transcribe_pcm.assert_called_with(b"voice")

    async def test_cancelled_stt_does_not_overlap_new_gpu_job(self):
        loop = asyncio.get_running_loop()
        first_started = asyncio.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        active = 0
        peak = 0
        lock = threading.Lock()

        def stt(pcm):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if pcm == b"first":
                    loop.call_soon_threadsafe(first_started.set)
                    release_first.wait(3)
                else:
                    second_started.set()
                return "测试"
            finally:
                with lock:
                    active -= 1

        with patch.object(main, "transcribe_pcm", side_effect=stt):
            try:
                await self.commit("first", b"first")
                await asyncio.wait_for(first_started.wait(), 3)
                await self.ws.send("input_audio_buffer.speech_started")
                self.assertEqual((await self.ws.next())["type"], "response.cancelled")
                await self.commit("second", b"second")
                # A session update proves receive stays responsive while STT runs.
                await self.ws.send("session.update")
                self.assertEqual((await self.ws.next())["type"], "session.updated")
                await asyncio.sleep(0.05)
                self.assertFalse(second_started.is_set())
            finally:
                release_first.set()
            events = await self.ws.until("response.audio.done")
        self.assertEqual(peak, 1)
        self.assertTrue(all(e["turn_id"] == "second" for e in events))

    async def test_tts_finishing_after_cancel_sends_no_old_audio(self):
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        finished = asyncio.Event()

        def delayed_tts(text):
            loop.call_soon_threadsafe(started.set)
            release.wait(3)
            loop.call_soon_threadsafe(finished.set)
            return b"\0" * 9600

        with patch.object(main, "synthesize_sapi", side_effect=delayed_tts):
            try:
                await self.commit("cancelled-tts")
                await asyncio.wait_for(started.wait(), 3)
                await self.ws.send("input_audio_buffer.speech_started")
                events = await self.ws.until("response.cancelled")
                self.assertFalse(any(e["type"] == "response.audio.delta" for e in events))
            finally:
                release.set()
            await asyncio.wait_for(finished.wait(), 3)
            await asyncio.sleep(0.02)
        await self.commit("next")
        events = await self.ws.until("response.audio.done")
        self.assertTrue(all(e["turn_id"] == "next" for e in events))

    async def test_error_is_tagged_and_next_turn_recovers(self):
        with patch.object(main, "ask_lmstudio", side_effect=RuntimeError("offline")):
            await self.commit("failed")
            events = await self.ws.until("error")
            self.assertEqual(events[-1]["turn_id"], "failed")
        await self.commit("recovered")
        events = await self.ws.until("response.audio.done")
        self.assertTrue(all(e["turn_id"] == "recovered" for e in events))

    async def test_disconnect_cancels_pending_turn(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def pending(text):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        with patch.object(main, "ask_lmstudio", side_effect=pending):
            await self.commit("disconnected")
            await asyncio.wait_for(started.wait(), 3)
            await self.ws.incoming.put(None)
            await asyncio.wait_for(self.task, 3)
            self.assertTrue(cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
