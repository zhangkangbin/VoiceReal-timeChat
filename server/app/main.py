import asyncio
import audioop
import base64
import json
import os
import subprocess
import time
import tempfile
import wave
from contextlib import suppress

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

load_dotenv()
app = FastAPI(title="Local Realtime Voice Chat Server", version="0.2.0")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
_whisper = None
_stt_lock = asyncio.Lock()


def provider_name() -> str:
    return os.getenv("AI_PROVIDER", "lmstudio").lower()


@app.get("/health")
async def health():
    return {"ok": True, "provider": provider_name(), "lmstudio_url": os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1"), "lmstudio_model": os.getenv("LMSTUDIO_MODEL", "qwen2.5-coder-14b-instruct"), "whisper_model": WHISPER_MODEL}


@app.get("/", response_class=HTMLResponse)
async def test_page():
    return HTMLResponse("""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>本地大模型测试</title><style>body{font-family:system-ui;max-width:760px;margin:40px auto;padding:0 16px}textarea{width:100%;height:110px;padding:10px}button{padding:10px 18px;margin:10px 0}#out{white-space:pre-wrap;background:#f4f4f4;padding:14px;min-height:80px}.ok{color:green}.err{color:#b00}</style></head>
<body><h1>本地大模型测试</h1><p>入口：FastAPI + LM Studio。本页面不会调用云端服务。</p><p id="health">检查本地服务中…</p>
<textarea id="prompt" placeholder="输入一句话，例如：用一句话介绍你自己"></textarea><br><button onclick="send()">发送给本地模型</button><div id="out"></div>
<script>
async function check(){try{let r=await fetch('/health');let x=await r.json();document.querySelector('#health').textContent='服务正常｜模型：'+x.lmstudio_model+'｜模式：'+x.provider;document.querySelector('#health').className='ok'}catch(e){document.querySelector('#health').textContent='服务不可用';document.querySelector('#health').className='err'}}
async function send(){let out=document.querySelector('#out');out.textContent='本地模型生成中…';try{let r=await fetch('/api/local-chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:document.querySelector('#prompt').value})});let x=await r.json();if(!r.ok)throw new Error(x.detail||x.message||'请求失败');out.textContent=x.reply}catch(e){out.textContent='错误：'+e.message}}
check();
</script></body></html>""")


@app.post("/api/local-chat")
async def local_chat(payload: dict):
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        return {"reply": "请输入测试内容"}
    try:
        return {"reply": await ask_lmstudio(prompt)}
    except httpx.ConnectError:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="LM Studio 本地 API 未启动，请在 LM Studio 中启动 Local Server")
    except Exception as error:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(error))


def transcribe_pcm(pcm: bytes) -> str:
    global _whisper
    if not pcm:
        return ""
    if _whisper is None:
        from faster_whisper import WhisperModel
        try:
            _whisper = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="float16" if WHISPER_DEVICE == "cuda" else "int8")
        except Exception:
            _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    # faster-whisper accepts an audio array or a media file. Write a temporary
    # WAV so the Android PCM format is explicit and reproducible.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
        path = file.name
    try:
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(pcm)
        def run_transcription(model):
            # Android already performs client-side VAD. Skipping a second VAD
            # pass reduces the time from commit to transcript.
            segments, _ = model.transcribe(path, language="zh", beam_size=1, vad_filter=False)
            return "".join(segment.text for segment in segments).strip()
        try:
            return run_transcription(_whisper)
        except Exception:
            if WHISPER_DEVICE != "cuda":
                raise
            _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            return run_transcription(_whisper)
    finally:
        with suppress(OSError):
            os.remove(path)


async def ask_lmstudio(text: str) -> str:
    url = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1").rstrip("/") + "/chat/completions"
    payload = {"model": os.getenv("LMSTUDIO_MODEL", "qwen2.5-coder-14b-instruct"), "messages": [
        {"role": "system", "content": "你是一个自然、简洁、友好的中文语音助手。回答适合直接朗读，不要使用 Markdown。"},
        {"role": "user", "content": text}], "stream": False, "temperature": 0.3, "max_tokens": 128}
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


def synthesize_sapi(text: str) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        wav_path = os.path.join(directory, "reply.wav")
        script = ("Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  "$f=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(24000,[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono); "
                  "$s.SetOutputAudioFormat($f); $s.SetOutputToWaveFile('%s'); $s.Speak('%s'); $s.Dispose()"
                  % (wav_path.replace("'", "''"), text.replace("'", "''")))
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], check=True, capture_output=True)
        with wave.open(wav_path, "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        if channels != 1 or width != 2:
            raise RuntimeError(f"Unsupported local SAPI format: channels={channels}, width={width}, rate={rate}")
        if rate != 24000:
            frames, _ = audioop.ratecv(frames, width, channels, rate, 24000, None)
        return frames


async def local_turn(websocket: WebSocket, pcm: bytes, turn_id: str):
    started = time.perf_counter()
    print(f"local turn start: {len(pcm)} bytes", flush=True)
    try:
        # faster-whisper/CTranslate2 should not receive concurrent GPU jobs
        # from multiple WebSocket clients on this local service.
        async with _stt_lock:
            text = await asyncio.to_thread(transcribe_pcm, pcm)
        print(f"local turn transcript: {text!r} ({time.perf_counter() - started:.2f}s)", flush=True)
        await websocket.send_json({"type": "conversation.transcript", "turn_id": turn_id, "text": text, "final": True})
        if not text:
            await websocket.send_json({"type": "response.audio.done", "turn_id": turn_id})
            return
        reply = await ask_lmstudio(text)
        print(f"local turn llm done ({time.perf_counter() - started:.2f}s)", flush=True)
        await websocket.send_json({"type": "response.text.done", "turn_id": turn_id, "text": reply, "final": True})
        audio = await asyncio.to_thread(synthesize_sapi, reply)
        print(f"local turn tts done: {len(audio)} bytes ({time.perf_counter() - started:.2f}s)", flush=True)
        for offset in range(0, len(audio), 4800):
            chunk = base64.b64encode(audio[offset:offset + 4800]).decode()
            await websocket.send_json({"type": "response.audio.delta", "turn_id": turn_id, "delta": chunk})
        await websocket.send_json({"type": "response.audio.done", "turn_id": turn_id})
    except WebSocketDisconnect:
        # The Android client may close/reconnect while local STT/LLM/TTS is
        # still running. Do not attempt to send a second error frame after
        # the close frame, which only creates noisy server-side task errors.
        return
    except Exception as error:
        with suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(error)})


async def openai_relay(client: WebSocket):
    url = "wss://api.openai.com/v1/realtime?model=" + os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "OpenAI-Beta": "realtime=v1"}
    async with websockets.connect(url, additional_headers=headers, max_size=None) as upstream:
        await upstream.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}, "output": {"format": {"type": "audio/pcm", "rate": 24000}}},
            "output_modalities": ["audio"], "voice": "alloy", "turn_detection": {"type": "server_vad", "silence_duration_ms": 500},
            "instructions": "你是一个简洁、自然、友好的中文语音助手。"}}))
        async def android_to_openai():
            while True:
                await upstream.send(await client.receive_text())
        async def openai_to_android():
            async for message in upstream:
                await client.send_text(message)
        tasks = [asyncio.create_task(android_to_openai()), asyncio.create_task(openai_to_android())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            with suppress(asyncio.CancelledError):
                task.result()


@app.websocket("/ws/realtime")
async def realtime(websocket: WebSocket):
    await websocket.accept()
    mode = provider_name()
    await websocket.send_json({"type": "server.ready", "provider": mode})
    if mode == "openai":
        try:
            await openai_relay(websocket)
        except WebSocketDisconnect:
            pass
        return
    pcm_buffer = bytearray()
    turn_index = 0
    try:
        while True:
            event = json.loads(await websocket.receive_text())
            event_type = event.get("type")
            if event_type == "input_audio_buffer.append":
                pcm_buffer.extend(base64.b64decode(event.get("audio", "")))
            elif event_type == "input_audio_buffer.commit":
                pcm = bytes(pcm_buffer); pcm_buffer.clear()
                turn_index += 1
                turn_id = f"turn-{turn_index}"
                # The Android client is half-duplex. Process one local turn
                # inline so a noisy VAD edge cannot start overlapping STT/LLM/TTS
                # jobs and leave the client waiting for a later audio.done.
                await local_turn(websocket, pcm, turn_id)
            elif event_type == "session.update":
                await websocket.send_json({"type": "session.updated", "provider": "local"})
    except WebSocketDisconnect:
        pass
