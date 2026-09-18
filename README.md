# Realtime Voice Chat

默认模式使用本机 GPU 和 CPU：Android 端采集 PCM 音频，通过 WebSocket 发送到 FastAPI；本机使用 Faster-Whisper 识别、LM Studio 运行本地 LLM、Windows SAPI 合成本地语音。OpenAI 仅作为显式配置时的可选适配器。

## 运行服务端

```powershell
cd server
.\setup_local.ps1
```

也可以手动使用 Python 3.11：

```powershell
cd server
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

默认服务地址：`ws://电脑局域网IP:8000/ws/realtime`

后续可以直接执行 `.\start_local.ps1`，自动启动 LM Studio 本地 API、加载模型并启动 FastAPI。

先在 LM Studio 中启动 Local Server，并加载 `google/gemma-2-27b`。服务端默认读取 `AI_PROVIDER=lmstudio`，不会调用云端 API。语音识别默认使用 `faster-whisper medium`、CUDA 和 `beam_size=5`，以提升中文识别准确率。

## Android

用 Android Studio 打开 `android` 目录。当前真机服务地址为 `ws://192.168.0.2:8000/ws/realtime`；局域网地址变化后请修改 `MainActivity.kt` 中的 `host`。模拟器可使用 `10.0.2.2` 访问宿主机。

Android 端使用 16kHz、单声道、PCM 16-bit 音频输入和本地 Silero VAD（ONNX Runtime），并播放本机服务返回的 24kHz PCM 语音。页面会显示 faster-whisper 识别出的用户文字，以及 LM Studio 返回的本地模型回答。VAD、语音识别、LLM 和 TTS 全部在手机或局域网电脑本地运行。
