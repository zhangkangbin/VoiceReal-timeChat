"""本地实时语音聊天服务的入口模块。

本模块把 Android 客户端（或测试网页）的 WebSocket 音频流连接到本机的
Silero/VAD 之后的语音转写、LM Studio 对话模型和 Windows SAPI 语音合成链路。
默认的 `lmstudio` 模式完全在局域网内工作；配置为 `openai` 时，则把事件转发
给 OpenAI Realtime API。除了 HTTP 健康检查和测试接口外，绝大多数状态都限定在
单条 WebSocket 会话中，以便不同用户的历史记录、记忆和正在播放的回合互不串扰。
"""

import asyncio
import audioop
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from contextlib import suppress

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response

from .memory import DEFAULT_USER_ID, MemoryService, memory_service
from .tools import TOOL_DEFINITIONS, execute_tool, realtime_tool_definitions, tool_names

# 从 server/.env（如果存在）读取本地开发配置；显式设置的环境变量仍由 dotenv
# 的默认行为保留。下面的常量在模块导入时确定，便于每个请求使用一致的配置。
load_dotenv()
app = FastAPI(title="Local Realtime Voice Chat Server", version="0.2.0")
# 测试网页与 API 进程一起提供，避免手机或浏览器测试时还要启动第二个静态服务。
TEST_PAGE = (Path(__file__).with_name("test_page.html")).read_text(encoding="utf-8")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
MAX_HISTORY_TURNS = max(1, int(os.getenv("MAX_HISTORY_TURNS", "30")))
MAX_TOOL_CALL_ROUNDS = max(1, int(os.getenv("MAX_TOOL_CALL_ROUNDS", "3")))
FUNCTION_CALLS_ENABLED = os.getenv("FUNCTION_CALLS_ENABLED", "true").lower() not in {"0", "false", "no", "off"}
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() not in {"0", "false", "no", "off"}
# 当前异步回合的上下文（用户 ID、回合 ID、召回的记忆）。ContextVar 能让并发
# WebSocket 回合各自看到自己的值，不需要把上下文层层添加到所有函数签名中。
_turn_context: ContextVar[dict] = ContextVar("turn_context", default={})
_whisper = None
_cuda_dll_handles = []
# Cancelling an asyncio task cannot stop a running CTranslate2 call. Keep
# serialization in the worker, so a cancelled job never overlaps the next one.
_stt_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-stt")


def configure_cuda_runtime() -> None:
    """把 pip 安装的 NVIDIA CUDA DLL 加入 Windows 进程的动态库搜索路径。

    faster-whisper 通过 CTranslate2 加载 CUDA 运行时。Windows 不一定会自动在
    Python 虚拟环境的 site-packages 中搜索这些 DLL，因此第一次转写前扫描 NVIDIA
    wheel 的 ``bin`` 目录并保存句柄。句柄必须一直保留到进程结束，否则 Python 可能
    过早卸载 DLL；重复调用则通过 ``_cuda_dll_handles`` 直接返回。
    """
    if os.name != "nt" or _cuda_dll_handles:
        return
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia_root.is_dir():
        return
    for dll_dir in sorted(nvidia_root.rglob("bin")):
        if dll_dir.is_dir():
            _cuda_dll_handles.append(os.add_dll_directory(str(dll_dir)))
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")


def provider_name() -> str:
    """返回当前语音服务使用的上游提供方名称（默认是本地 LM Studio）。"""
    return os.getenv("AI_PROVIDER", "lmstudio").lower()


@app.get("/health")
async def health():
    """返回服务和模型加载状态，供启动脚本、测试页和人工排障使用。

    该接口不触发模型加载，只报告 Whisper 是否已经按需初始化、当前 AI 提供方、
    可用工具以及会话历史/记忆等关键开关。
    """
    whisper_engine = getattr(_whisper, "model", None)
    return {"ok": True, "provider": provider_name(), "lmstudio_url": os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1"), "lmstudio_model": os.getenv("LMSTUDIO_MODEL", "google/gemma-3-12b"), "whisper_model": WHISPER_MODEL, "whisper_beam_size": WHISPER_BEAM_SIZE, "max_history_turns": MAX_HISTORY_TURNS, "function_calls_enabled": FUNCTION_CALLS_ENABLED, "memory_enabled": MEMORY_ENABLED, "memory_user_id": DEFAULT_USER_ID, "tools": tool_names(), "whisper_loaded": _whisper is not None, "whisper_device": getattr(whisper_engine, "device", None), "whisper_compute_type": getattr(whisper_engine, "compute_type", None)}


@app.get("/", response_class=HTMLResponse)
async def test_page():
    """返回内置 WebSocket 测试页的 HTML 内容。"""
    return HTMLResponse(TEST_PAGE)


@app.get("/api/test-default-audio")
async def test_default_audio():
    """生成一段固定的中文测试语句并以 24 kHz 单声道 WAV 返回。

    合成过程放到线程中，避免 Windows SAPI 的同步调用阻塞 FastAPI 事件循环；返回
    WAV 头则让浏览器可以直接播放，同时仍与客户端发送的 PCM 采样率约定一致。
    """
    pcm = await asyncio.to_thread(synthesize_sapi, "请介绍一下你自己，并说一句本地测试成功")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    return Response(content=output.getvalue(), media_type="audio/wav")


@app.post("/api/local-chat")
async def local_chat(payload: dict):
    """处理测试网页的非流式文本对话请求。

    请求先检查用户的显式记忆命令，再召回相关记忆并调用 LM Studio；模型调用结束后
    才提取本轮明确陈述的事实。这样“记住/忘记/查看”命令不会被模型误改写，也不会把
    本轮新产生的记忆提前混入同一轮的提示词。
    """
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        return {"reply": "请输入测试内容"}
    try:
        if MEMORY_ENABLED:
            command = await memory_service.handle_explicit_command(DEFAULT_USER_ID, prompt)
            if command and command.get("handled"):
                return {"reply": command["reply"]}
            memories = await memory_service.relevant(DEFAULT_USER_ID, prompt, limit=8)
            token = _turn_context.set({"user_id": DEFAULT_USER_ID, "memory_context": memories})
            try:
                reply = await ask_lmstudio(prompt)
            finally:
                _turn_context.reset(token)
            await memory_service.extract_explicit(DEFAULT_USER_ID, prompt)
            return {"reply": reply}
        return {"reply": await ask_lmstudio(prompt)}
    except httpx.ConnectError:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="LM Studio 本地 API 未启动，请在 LM Studio 中启动 Local Server")
    except Exception as error:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(error))


def transcribe_pcm(pcm: bytes) -> str:
    """把 16 kHz、单声道、16-bit little-endian PCM 转写为中文文本。

    Whisper 模型采用惰性初始化，并优先使用配置的 CUDA；加载或推理失败时自动
    回退到 CPU int8。临时 WAV 明确描述 Android 的原始 PCM 格式，转写完成后无论
    成功或失败都会删除。调用者通过单线程执行器调用本函数，以免 GPU 上出现并发
    CTranslate2 作业。
    """
    global _whisper
    if not pcm:
        return ""
    configure_cuda_runtime()
    if _whisper is None:
        from faster_whisper import WhisperModel
        try:
            _whisper = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="float16" if WHISPER_DEVICE == "cuda" else "int8")
        except Exception as error:
            print(f"Whisper CUDA unavailable; falling back to CPU: {type(error).__name__}: {error}", flush=True)
            _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    # faster-whisper accepts an audio array or a media file. Write a temporary
    # WAV so the Android PCM format is explicit and reproducible.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
        path = file.name
    try:
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(pcm)
        def run_transcription(model):
            """执行一次具体模型转写；参数是已初始化的 Whisper 实例。"""
            # Android already performs client-side VAD. Skipping a second VAD
            # pass reduces the time from commit to transcript.
            segments, _ = model.transcribe(
                path,
                language="zh",
                task="transcribe",
                beam_size=WHISPER_BEAM_SIZE,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt="以下是普通话语音转写，使用简体中文和自然标点。",
                vad_filter=False,
            )
            return "".join(segment.text for segment in segments).strip()
        try:
            return run_transcription(_whisper)
        except Exception as error:
            if WHISPER_DEVICE != "cuda":
                raise
            print(f"Whisper CUDA transcription failed; falling back to CPU: {type(error).__name__}: {error}", flush=True)
            from faster_whisper import WhisperModel
            _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            return run_transcription(_whisper)
    finally:
        with suppress(OSError):
            os.remove(path)


async def ask_lmstudio(text: str, history: list[dict] | None = None) -> str:
    """调用 OpenAI-compatible 的 LM Studio Chat Completions 接口。

    ``history`` 只传入最近的会话轮次；系统提示包含召回的长期记忆。模型可以返回
    一个或多个工具调用，本函数会逐个执行并把 JSON 结果追加回消息列表，直到模型
    产生最终文本或达到 ``MAX_TOOL_CALL_ROUNDS``。部分本地模型虽然暴露兼容接口却
    不支持 ``tools``，收到 400 时会在当前请求中降级为普通聊天。
    """
    url = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1").rstrip("/") + "/chat/completions"
    turn_context = _turn_context.get({})
    system_prompt = "你是一个自然、简洁、友好的中文语音助手。回答适合直接朗读，不要使用 Markdown。请记住本次会话中用户告诉你的信息，并在后续问题中使用这些信息。如果用户询问当前时间，请调用 get_current_time，不要猜测时间。"
    if MEMORY_ENABLED:
        system_prompt += " 如果用户明确要求记住、忘记或查看记忆，请使用 memory_save、memory_delete 或 memory_list。只有明确表达的偏好、习惯或事实才保存。"
        memory_prompt = MemoryService.prompt_context(turn_context.get("memory_context", []))
        if memory_prompt:
            system_prompt += "\n\n" + memory_prompt
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": text})
    tools_enabled = FUNCTION_CALLS_ENABLED
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        for tool_round in range(MAX_TOOL_CALL_ROUNDS + 1):
            payload = {"model": os.getenv("LMSTUDIO_MODEL", "google/gemma-3-12b"), "messages": messages, "stream": False, "temperature": 0.3, "max_tokens": 128}
            if tools_enabled:
                available_tools = [
                    tool for tool in TOOL_DEFINITIONS
                    if MEMORY_ENABLED or not tool["function"]["name"].startswith("memory_")
                ]
                payload["tools"] = available_tools
                payload["tool_choice"] = "auto"
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                # Some local models expose a Chat Completions-compatible API
                # without tool support. Keep ordinary chat usable in that case.
                if tools_enabled and error.response.status_code == 400:
                    print("LM Studio rejected tools; retrying without function calls", flush=True)
                    tools_enabled = False
                    continue
                raise

            message = response.json()["choices"][0]["message"]
            tool_calls = list(message.get("tool_calls") or [])
            legacy_call = message.get("function_call")
            if legacy_call and not tool_calls:
                tool_calls = [{"id": f"legacy-tool-{tool_round}", "type": "function", "function": legacy_call}]
            if not tool_calls:
                return str(message.get("content") or "").strip()
            if tool_round >= MAX_TOOL_CALL_ROUNDS:
                return "抱歉，工具调用次数超过限制，暂时无法完成这个请求。"

            # Preserve the assistant tool-call message before appending each
            # tool result, as required by the Chat Completions protocol.
            messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls})
            for call in tool_calls:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                arguments = function.get("arguments") or "{}"
                result = await execute_tool(name, arguments, context=turn_context)
                print(f"function call: {name} -> {result.get('ok')}", flush=True)
                messages.append({
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or f"tool-{tool_round}"),
                    "content": json.dumps(result, ensure_ascii=False),
                })
        return "抱歉，暂时无法完成这个请求。"


def synthesize_sapi(text: str) -> bytes:
    """使用 Windows SAPI 将文本合成为 24 kHz、单声道、16-bit PCM。

    System.Speech 只能写入 WAV 文件，因此先在临时目录中生成文件，再读取帧并在
    采样率不符合协议时用 ``audioop.ratecv`` 重采样。函数只返回裸 PCM，调用者负责
    按 WebSocket 音频事件分块和 Base64 编码。
    """
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


async def send_event(websocket: WebSocket, send_lock: asyncio.Lock, payload: dict):
    """在统一发送锁内发送一条 JSON 事件。

    STT、LLM/TTS 回合和上游 OpenAI 转发可能同时产生活动；锁保证多个协程不会把
    JSON 帧交错写入同一连接，也保留客户端可依赖的事件顺序。
    """
    async with send_lock:
        await websocket.send_json(payload)


async def local_turn(websocket: WebSocket, send_lock: asyncio.Lock, pcm: bytes, turn_id: str,
                     history: list[dict], user_id: str = DEFAULT_USER_ID):
    """完成本地模式的一整个回合：转写、记忆/模型处理、合成并发送音频。

    该协程由 ``realtime`` 为每次 commit 创建。取消它会停止尚未发送的后续阶段；
    底层 Whisper 原生调用本身无法被 asyncio 立即终止，但单线程执行器会保证旧任务
    完成后才运行下一次转写。所有事件都带同一个 ``turn_id``，手机因此能丢弃迟到的
    已取消回合数据。
    """
    started = time.perf_counter()
    print(f"local turn start: {len(pcm)} bytes", flush=True)
    try:
        # faster-whisper/CTranslate2 should not receive concurrent GPU jobs
        # from multiple WebSocket clients on this local service.
        text = await asyncio.get_running_loop().run_in_executor(_stt_executor, transcribe_pcm, pcm)
        print(f"local turn transcript: {text!r} ({time.perf_counter() - started:.2f}s)", flush=True)
        await send_event(websocket, send_lock, {"type": "conversation.transcript", "turn_id": turn_id, "text": text, "final": True})
        if not text:
            await send_event(websocket, send_lock, {"type": "response.audio.done", "turn_id": turn_id})
            return
        command = await memory_service.handle_explicit_command(user_id, text, turn_id) if MEMORY_ENABLED else None
        if command and command.get("handled"):
            reply = str(command["reply"])
        else:
            memories = await memory_service.relevant(user_id, text, limit=8) if MEMORY_ENABLED else []
            token = _turn_context.set({
                "user_id": user_id,
                "turn_id": turn_id,
                "memory_context": memories,
            })
            try:
                # Keep the first-turn call compatible with simple provider adapters;
                # subsequent turns receive the accumulated conversation history.
                reply = await ask_lmstudio(text, history) if history else await ask_lmstudio(text)
            finally:
                _turn_context.reset(token)
            if MEMORY_ENABLED:
                await memory_service.extract_explicit(user_id, text, turn_id)
        history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": reply}])
        del history[:-(MAX_HISTORY_TURNS * 2)]
        print(f"local turn llm done ({time.perf_counter() - started:.2f}s)", flush=True)
        await send_event(websocket, send_lock, {"type": "response.text.done", "turn_id": turn_id, "text": reply, "final": True})
        audio = await asyncio.to_thread(synthesize_sapi, reply)
        print(f"local turn tts done: {len(audio)} bytes ({time.perf_counter() - started:.2f}s)", flush=True)
        for offset in range(0, len(audio), 4800):
            chunk = base64.b64encode(audio[offset:offset + 4800]).decode()
            await send_event(websocket, send_lock, {"type": "response.audio.delta", "turn_id": turn_id, "delta": chunk})
            # A buffered WebSocket send need not yield; let barge-in be read
            # even while many audio chunks are ready to send.
            await asyncio.sleep(0)
        await send_event(websocket, send_lock, {"type": "response.audio.done", "turn_id": turn_id})
    except asyncio.CancelledError:
        print(f"local turn cancelled: {turn_id}", flush=True)
        raise
    except WebSocketDisconnect:
        # The Android client may close/reconnect while local STT/LLM/TTS is
        # still running. Do not attempt to send a second error frame after
        # the close frame, which only creates noisy server-side task errors.
        return
    except Exception as error:
        with suppress(Exception):
            await send_event(websocket, send_lock, {"type": "error", "turn_id": turn_id, "message": str(error)})


def realtime_function_call(event: dict) -> tuple[str, str, str] | None:
    """从两种 Realtime 事件格式中提取已完成的工具调用。

    不同版本/代理可能发送 ``response.function_call_arguments.done``，或把函数调用
    包装在 ``response.output_item.done`` 中。统一返回调用 ID、函数名和 JSON 参数，
    未识别或缺字段时返回 ``None``，让普通音频/文本事件原样转发。
    """
    if event.get("type") == "response.function_call_arguments.done":
        call_id = str(event.get("call_id") or "")
        name = str(event.get("name") or "")
        arguments = str(event.get("arguments") or "{}")
        return (call_id, name, arguments) if call_id and name else None
    if event.get("type") == "response.output_item.done":
        item = event.get("item") or {}
        if item.get("type") == "function_call":
            call_id = str(item.get("call_id") or "")
            name = str(item.get("name") or "")
            arguments = str(item.get("arguments") or "{}")
            return (call_id, name, arguments) if call_id and name else None
    return None


async def openai_relay(client: WebSocket, send_lock: asyncio.Lock):
    """在客户端 WebSocket 与 OpenAI Realtime WebSocket 之间转发事件。

    上行协程负责把 Android 的音频和会话设置送给 OpenAI，下行协程负责向客户端回传
    音频/文本并拦截函数调用。工具由本服务执行，再以 ``function_call_output`` 回送
    上游；任一方向结束后取消另一方向，避免连接泄漏。
    """
    url = "wss://api.openai.com/v1/realtime?model=" + os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "OpenAI-Beta": "realtime=v1"}
    async with websockets.connect(url, additional_headers=headers, max_size=None) as upstream:
        user_id = DEFAULT_USER_ID
        memory_context = await memory_service.relevant(DEFAULT_USER_ID, "", limit=8) if MEMORY_ENABLED else []
        instructions = "你是一个简洁、自然、友好的中文语音助手。如果用户询问当前时间，请调用 get_current_time，不要猜测时间。"
        if MEMORY_ENABLED:
            instructions += " 如果用户明确要求记住、忘记或查看记忆，请使用 memory_save、memory_delete 或 memory_list。只有明确表达的偏好、习惯或事实才保存。"
            memory_prompt = MemoryService.prompt_context(memory_context)
            if memory_prompt:
                instructions += "\n\n" + memory_prompt
        session = {
            "type": "realtime", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}, "output": {"format": {"type": "audio/pcm", "rate": 24000}}},
            "output_modalities": ["audio"], "voice": "alloy", "turn_detection": {"type": "server_vad", "silence_duration_ms": 500},
            "instructions": instructions}
        if FUNCTION_CALLS_ENABLED:
            session["tools"] = [
                tool for tool in realtime_tool_definitions()
                if MEMORY_ENABLED or not tool["name"].startswith("memory_")
            ]
            session["tool_choice"] = "auto"
        await upstream.send(json.dumps({"type": "session.update", "session": session}))
        async def android_to_openai():
            """读取客户端事件，注入本地用户 ID 的记忆上下文后转发给上游。"""
            nonlocal user_id
            while True:
                raw_event = await client.receive_text()
                try:
                    event = json.loads(raw_event)
                except (TypeError, json.JSONDecodeError):
                    await upstream.send(raw_event)
                    continue
                if event.get("type") == "session.update" and event.get("user_id"):
                    user_id = str(event.pop("user_id"))[:128]
                    # user_id is an app-local field, not a Realtime API field.
                    # Refresh instructions with the memories for this user.
                    if MEMORY_ENABLED:
                        selected = await memory_service.relevant(user_id, "", limit=8)
                        refreshed_instructions = "你是一个简洁、自然、友好的中文语音助手。如果用户询问当前时间，请调用 get_current_time，不要猜测时间。 如果用户明确要求记住、忘记或查看记忆，请使用 memory_save、memory_delete 或 memory_list。只有明确表达的偏好、习惯或事实才保存。"
                        memory_prompt = MemoryService.prompt_context(selected)
                        if memory_prompt:
                            refreshed_instructions += "\n\n" + memory_prompt
                        await upstream.send(json.dumps({"type": "session.update", "session": {"instructions": refreshed_instructions}}))
                    if set(event) == {"type"}:
                        continue
                await upstream.send(json.dumps(event))

        async def openai_to_android():
            """转发上游事件，并对每个函数调用 ID 至多执行一次本地工具。"""
            handled_calls: set[str] = set()
            async for message in upstream:
                try:
                    event = json.loads(message)
                except (TypeError, json.JSONDecodeError):
                    await client.send_text(message)
                    continue

                function_call = realtime_function_call(event) if FUNCTION_CALLS_ENABLED else None
                if function_call is None:
                    await client.send_text(message)
                    continue
                call_id, name, arguments = function_call
                if call_id in handled_calls:
                    continue
                handled_calls.add(call_id)
                await send_event(client, send_lock, {"type": "tool.started", "name": name, "call_id": call_id})
                result = await execute_tool(name, arguments, context={"user_id": user_id, "call_id": call_id})
                await upstream.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": call_id, "output": json.dumps(result, ensure_ascii=False)},
                }))
                await upstream.send(json.dumps({"type": "response.create"}))
                await send_event(client, send_lock, {"type": "tool.completed", "name": name, "call_id": call_id, "ok": result.get("ok", False)})

        tasks = [asyncio.create_task(android_to_openai()), asyncio.create_task(openai_to_android())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            with suppress(asyncio.CancelledError):
                task.result()


@app.websocket("/ws/realtime")
async def realtime(websocket: WebSocket):
    """处理一条实时语音 WebSocket 会话。

    本地模式维护输入 PCM 缓冲区、会话历史和当前任务。客户端检测到说话开始时
    触发 barge-in，立即取消旧回合并清空缓冲；commit 则复制当前 PCM、分配唯一
    回合 ID 并启动 ``local_turn``。会话结束时无论是正常断开还是异常断开，都会
    取消仍在运行的回合，防止后台继续生成或向已关闭连接写入数据。
    """
    await websocket.accept()
    mode = provider_name()
    send_lock = asyncio.Lock()
    await send_event(websocket, send_lock, {"type": "server.ready", "provider": mode})
    if mode == "openai":
        try:
            await openai_relay(websocket, send_lock)
        except WebSocketDisconnect:
            pass
        return
    pcm_buffer = bytearray()
    history: list[dict] = []
    user_id = DEFAULT_USER_ID
    turn_index = 0
    current_turn_task: asyncio.Task | None = None
    current_turn_id: str | None = None

    async def cancel_current_turn(notify: bool = True):
        """取消当前回合并可选地通知客户端；重复调用是安全的。"""
        nonlocal current_turn_task, current_turn_id
        task = current_turn_task
        turn_id = current_turn_id
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        # The response may already be sent but still queued on the phone.
        if notify and turn_id is not None:
            await send_event(websocket, send_lock, {"type": "response.cancelled", "turn_id": turn_id})
        current_turn_task = None
        current_turn_id = None

    try:
        while True:
            event = json.loads(await websocket.receive_text())
            event_type = event.get("type")
            if event_type == "input_audio_buffer.append":
                pcm_buffer.extend(base64.b64decode(event.get("audio", "")))
            elif event_type == "input_audio_buffer.speech_started":
                # Barge-in: a new speech segment interrupts the current local
                # STT/LLM/TTS turn while the receive loop keeps accepting audio.
                await cancel_current_turn()
                pcm_buffer.clear()
            elif event_type == "input_audio_buffer.clear":
                pcm_buffer.clear()
            elif event_type == "response.cancel":
                await cancel_current_turn()
            elif event_type == "input_audio_buffer.commit":
                pcm = bytes(pcm_buffer); pcm_buffer.clear()
                turn_index += 1
                # Echo the client's ID so it can reject late frames from a
                # cancelled response without waiting for an acknowledgement.
                turn_id = str(event.get("turn_id") or f"turn-{turn_index}")[:128]
                await cancel_current_turn(notify=False)
                current_turn_id = turn_id
                current_turn_task = asyncio.create_task(local_turn(websocket, send_lock, pcm, turn_id, history, user_id))
            elif event_type == "session.update":
                requested_user_id = str(event.get("user_id") or "").strip()
                if requested_user_id:
                    if requested_user_id[:128] != user_id:
                        history.clear()
                    user_id = requested_user_id[:128]
                await send_event(websocket, send_lock, {"type": "session.updated", "provider": "local"})
    except WebSocketDisconnect:
        pass
    finally:
        await cancel_current_turn(notify=False)
