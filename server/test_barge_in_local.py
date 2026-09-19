"""Exercise real local ASR/LLM/TTS through a LAN WebSocket, including barge-in."""
import argparse
import asyncio
import audioop
import base64
import json
import time

import websockets
from app.main import synthesize_sapi


async def main(url):
    pcm = await asyncio.to_thread(synthesize_sapi, "请回复本地测试成功")
    pcm, _ = audioop.ratecv(pcm, 2, 1, 24000, 16000, None)
    async with websockets.connect(url) as ws:
        assert json.loads(await ws.recv())["type"] == "server.ready"

        async def utterance(turn_id):
            for offset in range(0, len(pcm), 4800):
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[offset:offset + 4800]).decode()}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "turn_id": turn_id}))

        async def receive():
            event = json.loads(await asyncio.wait_for(ws.recv(), 120))
            if event["type"] == "error":
                raise RuntimeError(event)
            return event

        for index in range(3):
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
                    assert event["turn_id"] == turn_id
                    break
            print(json.dumps({"turn": turn_id, "cancel_ack_ms": round((time.perf_counter() - started) * 1000, 1)}), flush=True)

        started = time.perf_counter()
        await utterance("final")
        audio_bytes = 0
        kinds = set()
        while True:
            event = await receive()
            assert event["turn_id"] == "final", f"Late cancelled event: {event['type']}"
            kinds.add(event["type"])
            if event["type"] == "response.audio.delta":
                audio_bytes += len(base64.b64decode(event["delta"]))
            if event["type"] == "response.audio.done":
                break
        assert {"conversation.transcript", "response.text.done", "response.audio.delta"}.issubset(kinds)
        assert audio_bytes > 0
        print(json.dumps({"ok": True, "audio_bytes": audio_bytes, "final_turn_seconds": round(time.perf_counter() - started, 2)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://192.168.0.2:8000/ws/realtime")
    asyncio.run(main(parser.parse_args().url))
