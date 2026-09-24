"""全双工 WebSocket 协议回归测试。

本模块用假的 WebSocket 和 mock 后端组件覆盖实时语音会话的并发边界，
不下载模型也不依赖显卡，因此可以稳定验证回合 ID、打断、取消和断线清理。
测试从 server 目录运行：``.venv/Scripts/python.exe -m unittest -v test_full_duplex``。
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
    """用两个 asyncio 队列模拟 WebSocket 的收发方向。

    测试代码通过 ``send`` 把客户端事件放入 incoming，应用通过 ``send_json``
    把服务端事件放入 outgoing；``None`` 是专门约定的断开哨兵。
    """

    def __init__(self):
        """初始化客户端→服务端和服务端→客户端两个独立队列。"""
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()

    async def accept(self):
        """满足 FastAPI WebSocket 接口的握手方法；假连接无需实际握手。"""
        pass

    async def receive_text(self):
        """取出一个客户端事件并编码成应用期望的 JSON 文本。

        ``None`` 模拟真实连接中的 WebSocketDisconnect，使 realtime 协程进入清理路径。
        """
        event = await self.incoming.get()
        if event is None:
            raise WebSocketDisconnect()
        return json.dumps(event)

    async def send_json(self, event):
        """记录应用发出的事件，测试随后用 next/until 按顺序读取。"""
        await self.outgoing.put(event)

    async def send(self, event_type, **values):
        """构造并注入一个客户端协议事件。"""
        await self.incoming.put({"type": event_type, **values})

    async def next(self):
        """等待下一个服务端事件；超时可避免协议回归测试永久挂起。"""
        return await asyncio.wait_for(self.outgoing.get(), 3)

    async def until(self, event_type):
        """持续收集事件直到出现指定类型，并返回包含终止事件的完整序列。"""
        events = []
        while True:
            event = await self.next()
            events.append(event)
            if event["type"] == event_type:
                return events


class FullDuplexTests(unittest.IsolatedAsyncioTestCase):
    """覆盖 realtime 协程的正常回合和并发取消语义。"""

    async def asyncSetUp(self):
        """为每个用例替换外部依赖，并启动一个独立的 realtime 会话。

        provider、STT、LLM 和 TTS 都是确定性 mock，使断言只关注协议顺序与任务生命周期。
        ``server.ready`` 作为服务端已完成初始化的同步点，避免测试抢跑。
        """
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
        """向假连接发送断开哨兵，等待 realtime 收尾后恢复所有 mock。"""
        await self.ws.incoming.put(None)
        try:
            await asyncio.wait_for(self.task, 3)
        finally:
            for item in reversed(self.patches):
                item.stop()

    async def commit(self, turn_id, pcm=b"\0" * 1024):
        """按真实客户端顺序追加一段 Base64 PCM，再提交带 ID 的回合。"""
        await self.ws.send("input_audio_buffer.append", audio=base64.b64encode(pcm).decode())
        await self.ws.send("input_audio_buffer.commit", turn_id=turn_id)

    async def test_normal_response_echoes_client_turn_id(self):
        """正常响应应包含固定事件顺序，并在每个事件中回显客户端 turn_id。"""
        await self.commit("client-1")
        events = await self.ws.until("response.audio.done")
        self.assertEqual([e["type"] for e in events], [
            "conversation.transcript", "response.text.done", "response.audio.delta",
            "response.audio.delta", "response.audio.done"])
        self.assertTrue(all(e["turn_id"] == "client-1" for e in events))

    async def test_barge_in_during_llm_then_next_turn_completes(self):
        """LLM 阶段收到新语音时取消旧回合，下一回合仍可完整完成。"""
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def slow_llm(text):
            """让模型请求永久等待，以观测 realtime 对任务取消的传播。"""
            # 永不自行结束，只有 realtime 取消任务时 finally 才会释放 cancelled。
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        with patch.object(main, "ask_lmstudio", side_effect=slow_llm):
            await self.commit("old")
            await asyncio.wait_for(started.wait(), 3)
            # speech_started 代表用户抢话；response.cancelled 必须明确标记被打断的旧回合。
            await self.ws.send("input_audio_buffer.speech_started")
            events = await self.ws.until("response.cancelled")
            self.assertEqual(events[-1]["turn_id"], "old")
            self.assertTrue(cancelled.is_set())
            self.assertFalse(any(e["type"] == "response.audio.delta" for e in events))
        await self.commit("new")
        events = await self.ws.until("response.audio.done")
        self.assertTrue(all(e["turn_id"] == "new" for e in events))

    async def test_cancel_ack_even_when_audio_already_sent(self):
        """即使旧回合已经发送过音频，抢话也必须立即收到取消确认。"""
        await self.commit("buffered-on-phone")
        await self.ws.until("response.audio.done")
        await self.ws.send("input_audio_buffer.speech_started")
        self.assertEqual(await self.ws.next(), {"type": "response.cancelled", "turn_id": "buffered-on-phone"})

    async def test_new_speech_and_discard_clear_uncommitted_audio(self):
        """新语音开始和显式 clear 都应丢弃尚未提交的旧音频，避免串入下一回合。"""
        await self.ws.send("input_audio_buffer.append", audio=base64.b64encode(b"stale").decode())
        await self.ws.send("input_audio_buffer.speech_started")
        await self.commit("fresh", b"fresh")
        await self.ws.until("response.audio.done")
        # 只有 fresh 被提交；若 stale 未清除，此断言会观察到错误的 STT 输入。
        main.transcribe_pcm.assert_called_once_with(b"fresh")
        await self.ws.send("input_audio_buffer.append", audio=base64.b64encode(b"noise").decode())
        await self.ws.send("input_audio_buffer.clear")
        await self.commit("after-discard", b"voice")
        await self.ws.until("response.audio.done")
        main.transcribe_pcm.assert_called_with(b"voice")

    async def test_cancelled_stt_does_not_overlap_new_gpu_job(self):
        """STT 在工作线程中被取消后，后续回合必须等待旧任务退出，不能并发占用 GPU。"""
        loop = asyncio.get_running_loop()
        first_started = asyncio.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        active = 0
        peak = 0
        lock = threading.Lock()

        def stt(pcm):
            """记录工作线程中的并发数，并让第一段 STT 可控地阻塞。"""
            # active/peak 是并发探针；peak 应保持 1，证明两个 STT 没有重叠。
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
                # session.update 能在 STT 阻塞时得到响应，证明 receive 循环没有被同步任务卡死。
                await self.ws.send("session.update")
                self.assertEqual((await self.ws.next())["type"], "session.updated")
                await asyncio.sleep(0.05)
                self.assertFalse(second_started.is_set())
            finally:
                # 释放第一个同步 STT；realtime 应在它真正退出后才启动 second。
                release_first.set()
            events = await self.ws.until("response.audio.done")
        self.assertEqual(peak, 1)
        self.assertTrue(all(e["turn_id"] == "second" for e in events))

    async def test_tts_finishing_after_cancel_sends_no_old_audio(self):
        """TTS 已进入同步函数后才被取消时，迟到的旧音频也不能泄漏到客户端。"""
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        finished = asyncio.Event()

        def delayed_tts(text):
            """把合成阻塞到测试明确释放，模拟取消发生在 TTS 已启动之后。"""
            # 通过线程事件把 TTS 的“已开始/已完成”与测试时序解耦。
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
                # 旧回合即使最终返回字节，也必须被 turn 状态检查拦截，不能发送 delta。
                self.assertFalse(any(e["type"] == "response.audio.delta" for e in events))
            finally:
                # 让后台 TTS 收尾，随后再验证新回合仍可正常播放。
                release.set()
            await asyncio.wait_for(finished.wait(), 3)
            await asyncio.sleep(0.02)
        await self.commit("next")
        events = await self.ws.until("response.audio.done")
        self.assertTrue(all(e["turn_id"] == "next" for e in events))

    async def test_error_is_tagged_and_next_turn_recovers(self):
        """LLM 异常应带上失败回合 ID；错误后会话仍能处理下一回合。"""
        with patch.object(main, "ask_lmstudio", side_effect=RuntimeError("offline")):
            await self.commit("failed")
            events = await self.ws.until("error")
            self.assertEqual(events[-1]["turn_id"], "failed")
        await self.commit("recovered")
        events = await self.ws.until("response.audio.done")
        self.assertTrue(all(e["turn_id"] == "recovered" for e in events))

    async def test_disconnect_cancels_pending_turn(self):
        """客户端断线时，尚未完成的 LLM 任务必须收到取消并释放资源。"""
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def pending(text):
            """模拟永不结束的模型请求，用于验证断线清理。"""
            # 用永不完成的 Future 模拟长期请求；finally 是取消是否传递到任务的探针。
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
