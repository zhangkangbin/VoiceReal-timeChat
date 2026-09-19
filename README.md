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

## 全双工与打断

录音和 Silero VAD 在识别、模型生成以及播放过程中持续工作。用户开口后，手机立即停止并清空旧回答的播放队列，通知服务端取消旧回合；说完后自动识别新的一句话。没有播放后的监听冷却时间，也不需要按住按钮。

- 使用 Android 通信音频模式和设备支持的 AEC（回声消除）、NS（降噪）、AGC（自动增益），日志会显示各效果是否启用。参考 [Android AEC 文档](https://developer.android.com/reference/android/media/audiofx/AcousticEchoCanceler)。扬声器大音量、远距离或不支持 AEC 的手机仍需真机调试；必要时使用耳机。
- 网络收包和音频播放使用独立线程，播放采用非阻塞写入。每个回合有客户端生成的唯一标识，迟到的旧音频、错误和结束事件不会覆盖新回合。
- `input_audio_buffer.speech_started` 清理旧输入并取消当前回答；`input_audio_buffer.commit` 可携带 `turn_id`；服务端在转写、文本、音频、取消及错误事件中原样返回该标识。短噪声段使用 `input_audio_buffer.clear` 丢弃。
- 取消会立即停止旧回合输出，但已开始的 Whisper 原生计算和 Windows SAPI 合成不保证被立即终止。Whisper 使用单工作线程串行执行，防止取消后旧计算与新计算争抢 GPU；旧合成结果会被丢弃。
- 这是持续收音和可打断的双工交互；本地 ASR 仍在一句话结束后识别，不是逐字流式识别。

服务端回归测试（无需加载模型）：

```powershell
cd server
.\.venv\Scripts\python.exe -m unittest -v test_full_duplex
```

真机验收：开始会话，提问后在回答播放中说“停一下，换个问题”；应立即停播，并在新一句结束后显示新的识别和回答。连续打断三次、停止后重启会话，再测试扬声器不同音量下是否发生自我打断。
