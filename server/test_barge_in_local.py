"""通过局域网 WebSocket 验证真实本地 ASR/LLM/TTS 链路及抢话。

脚本使用本机 SAPI 生成固定的中文输入，经过采样率转换后分片上传，
先重复三次在转写完成后打断旧回合，再执行一次完整回合，适合手动检查真实模型的
取消延迟、旧事件隔离和最终音频输出。它不是 unittest，用断言失败直接报告协议问题。
"""
import argparse
import asyncio
import audioop
import base64
import json
import time

import websockets
from app.main import synthesize_sapi


async def main(url):
    """连接指定的实时端点，执行三轮可取消回合和一轮完整回合。

    ``turn_id`` 会随服务端的 transcript/text/audio/cancel 事件回传；最终回合的
    每个事件都必须仍属于 ``final``，这正是检测“旧回合迟到事件”是否被丢弃的关键断言。
    """
    # 用本机 SAPI 生成确定内容，避免依赖麦克风；to_thread 防止同步 TTS 阻塞事件循环。
    pcm = await asyncio.to_thread(synthesize_sapi, "请回复本地测试成功")
    # SAPI 返回 24 kHz、16-bit 数据；ratecv 将其转换到 Whisper 端点使用的 16 kHz。
    pcm, _ = audioop.ratecv(pcm, 2, 1, 24000, 16000, None)
    async with websockets.connect(url) as ws:
        # 只有收到 server.ready 才开始发送音频，确保服务端会话已初始化。
        assert json.loads(await ws.recv())["type"] == "server.ready"

        async def utterance(turn_id):
            """按 4800 字节分片追加 PCM，并用给定 ID 提交一个回合。"""
            # 分片边界与浏览器测试页保持一致；Base64 只是 WebSocket JSON 的传输编码。
            for offset in range(0, len(pcm), 4800):
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[offset:offset + 4800]).decode()}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "turn_id": turn_id}))

        async def receive():
            """接收一个带超时的事件，并把服务端 error 转成测试异常。"""
            event = json.loads(await asyncio.wait_for(ws.recv(), 120))
            if event["type"] == "error":
                raise RuntimeError(event)
            return event

        for index in range(3):
            # 每轮都等到真实 ASR 完成，再模拟用户开始说下一句话（barge-in）。
            turn_id = f"interrupted-{index}"
            await utterance(turn_id)
            # Cancel as soon as real ASR finishes and local LLM is starting.
            while (await receive())["type"] != "conversation.transcript":
                pass
            started = time.perf_counter()
            await ws.send(json.dumps({"type": "input_audio_buffer.speech_started"}))
            while True:
                event = await receive()
                if event["type"] == "response.cancelled":
                    # 取消确认必须指向刚被抢话的旧回合，不能误报新回合或缺少 ID。
                    assert event["turn_id"] == turn_id
                    break
            print(json.dumps({"turn": turn_id, "cancel_ack_ms": round((time.perf_counter() - started) * 1000, 1)}), flush=True)

        started = time.perf_counter()
        await utterance("final")
        # 最后一轮不打断，收集完整事件并统计服务端返回的音频字节数。
        audio_bytes = 0
        kinds = set()
        while True:
            event = await receive()
            # 若取消回合的事件迟到，这个断言会立即失败，暴露旧事件未丢弃的问题。
            assert event["turn_id"] == "final", f"Late cancelled event: {event['type']}"
            kinds.add(event["type"])
            if event["type"] == "response.audio.delta":
                audio_bytes += len(base64.b64decode(event["delta"]))
            if event["type"] == "response.audio.done":
                break
        # 完整回合至少要有转写、文本和音频增量；音频非空说明 TTS 真正产出了数据。
        assert {"conversation.transcript", "response.text.done", "response.audio.delta"}.issubset(kinds)
        assert audio_bytes > 0
        print(json.dumps({"ok": True, "audio_bytes": audio_bytes, "final_turn_seconds": round(time.perf_counter() - started, 2)}), flush=True)


if __name__ == "__main__":
    # 提供默认局域网地址，同时允许通过 --url 指定其他部署实例。
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://192.168.0.2:8000/ws/realtime")
    asyncio.run(main(parser.parse_args().url))
