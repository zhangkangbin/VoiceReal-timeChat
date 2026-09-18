import asyncio
import audioop
import base64
import json

import websockets

from app.main import synthesize_sapi


async def main():
    pcm = synthesize_sapi("请回复本地测试成功")
    pcm, _ = audioop.ratecv(pcm, 2, 1, 24000, 16000, None)
    print("input_pcm", len(pcm))
    async with websockets.connect("ws://127.0.0.1:8000/ws/realtime") as ws:
        print("ready", await ws.recv())
        for offset in range(0, len(pcm), 4800):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[offset:offset + 4800]).decode(),
            }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        while True:
            message = await asyncio.wait_for(ws.recv(), 120)
            print(message[:500])
            if "response.audio.done" in message or '"type": "error"' in message:
                break


asyncio.run(main())
