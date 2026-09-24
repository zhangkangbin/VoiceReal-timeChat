"""最小化的本地语音端到端冒烟脚本。

脚本不模拟服务端组件，而是连接本机已启动的 WebSocket，使用 SAPI 生成输入，
验证服务至少能完成一次音频上传、转写/回答并返回音频或错误事件；输出前 500 个字符
便于在命令行快速观察事件内容。
"""

import asyncio
import audioop
import base64
import json

import websockets

from app.main import synthesize_sapi


async def main():
    """生成固定 PCM，分片提交到本地实时端点并打印直到回合结束的事件。"""
    # 同步 SAPI 调用只执行一次；脚本是人工冒烟测试，不需要把它放入线程池。
    pcm = synthesize_sapi("请回复本地测试成功")
    # 服务端输入约定 16 kHz 单声道 PCM；SAPI 返回 24 kHz、16-bit 数据。
    pcm, _ = audioop.ratecv(pcm, 2, 1, 24000, 16000, None)
    print("input_pcm", len(pcm))
    async with websockets.connect("ws://127.0.0.1:8000/ws/realtime") as ws:
        # 打印握手事件，确认连接已准备好接收 append/commit。
        print("ready", await ws.recv())
        # 以固定大小的 Base64 分片上传，避免一次 JSON 消息过大。
        for offset in range(0, len(pcm), 4800):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[offset:offset + 4800]).decode(),
            }))
        # 未显式提供 turn_id 时由服务端生成/管理回合标识，后续事件仍会带有统一上下文。
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        while True:
            # 120 秒上限防止模型或服务异常时脚本无限等待。
            message = await asyncio.wait_for(ws.recv(), 120)
            print(message[:500])
            # audio.done 表示正常回合的音频流结束；error 表示服务端已明确失败。
            if "response.audio.done" in message or '"type": "error"' in message:
                break


asyncio.run(main())
