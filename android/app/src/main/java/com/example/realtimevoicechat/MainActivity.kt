package com.example.realtimevoicechat

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import okhttp3.*
import org.json.JSONObject
import java.util.Base64
import kotlin.concurrent.thread

class MainActivity : ComponentActivity() {
    private val requestPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermission.launch(Manifest.permission.RECORD_AUDIO)
        }
        setContent { VoiceScreen() }
    }

    @Composable
    private fun VoiceScreen() {
        var active by remember { mutableStateOf(false) }
        var status by remember { mutableStateOf("未连接") }
        var messages by remember { mutableStateOf(emptyList<ChatMessage>()) }
        val client = remember {
            VoiceClient(
                context = this@MainActivity,
                onStatus = { message -> runOnUiThread { status = message } },
                onUserText = { turnId, text ->
                    runOnUiThread {
                        if (text.isNotBlank()) {
                            messages = (messages + ChatMessage("$turnId-user", ChatRole.USER, text)).takeLast(100)
                        }
                    }
                },
                onAssistantText = { turnId, text ->
                    runOnUiThread {
                        if (text.isNotBlank()) {
                            messages = (messages + ChatMessage("$turnId-assistant", ChatRole.ASSISTANT, text)).takeLast(100)
                        }
                    }
                }
            )
        }
        MaterialTheme {
            Surface(modifier = Modifier.fillMaxSize()) {
                Column(modifier = Modifier.fillMaxSize().padding(24.dp)) {
                    Text("实时语音聊天", style = MaterialTheme.typography.headlineMedium)
                    Text(status, modifier = Modifier.padding(top = 8.dp))
                    LazyColumn(
                        modifier = Modifier
                            .fillMaxWidth()
                            .fillMaxHeight(0.70f)
                            .padding(vertical = 16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        items(messages, key = { it.id }) { message ->
                            MessageBubble(message)
                        }
                    }
                    Button(onClick = {
                        if (active) {
                            client.stopSession()
                            active = false
                        } else {
                            client.startSession()
                            active = true
                        }
                    }) { Text(if (active) "停止会话" else "开始会话") }
                }
            }
        }
    }

    private class VoiceClient(
        private val context: Context,
        private val onStatus: (String) -> Unit,
        private val onUserText: (String, String) -> Unit,
        private val onAssistantText: (String, String) -> Unit
    ) {
        private val tag = "RealtimeVoiceChat"
        private val host = "192.168.0.2:8000"
        private val inputSampleRate = SileroVad.SAMPLE_RATE
        private val outputSampleRate = 24000
        private val frameBytes = SileroVad.FRAME_SAMPLES * 2 // 32ms, mono, PCM16
        private val http = OkHttpClient()
        private var socket: WebSocket? = null
        private var recorder: AudioRecord? = null
        private var player: AudioTrack? = null
        @Volatile private var sessionActive = false
        @Volatile private var captureRunning = false
        @Volatile private var turnInFlight = false
        @Volatile private var resetVad = false

        fun startSession() {
            sessionActive = true
            if (socket != null) {
                startCapture()
                return
            }
            val outputBuffer = AudioTrack.getMinBufferSize(
                outputSampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            player = AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build())
                .setAudioFormat(AudioFormat.Builder()
                    .setSampleRate(outputSampleRate)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build())
                .setBufferSizeInBytes(outputBuffer * 2)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()
            player?.play()

            socket = http.newWebSocket(
                Request.Builder().url("ws://$host/ws/realtime").build(),
                object : WebSocketListener() {
                    override fun onOpen(webSocket: WebSocket, response: Response) {
                        onStatus("已连接本地服务，等待说话")
                        startCapture()
                    }

                    override fun onMessage(webSocket: WebSocket, text: String) {
                        val event = runCatching { JSONObject(text) }.getOrNull() ?: return
                        val eventType = event.optString("type")
                        val turnId = event.optString("turn_id", "turn-unknown")
                        when (eventType) {
                            "conversation.transcript" -> {
                                val transcript = event.optString("text").trim()
                                if (transcript.isNotEmpty()) {
                                    onUserText(turnId, transcript)
                                    onStatus("本地识别完成，正在理解…")
                                } else {
                                    onStatus("没有识别到清晰语音")
                                }
                            }
                            "response.text.done" -> {
                                val reply = event.optString("text").trim()
                                if (reply.isNotEmpty()) onAssistantText(turnId, reply)
                                onStatus("正在播放回答…")
                            }
                            "response.audio.delta" -> playAudioDelta(event.optString("delta"))
                            "response.audio.done" -> {
                                turnInFlight = false
                                resetVad = true
                                if (sessionActive) onStatus("等待说话")
                            }
                            "error" -> {
                                turnInFlight = false
                                resetVad = true
                                onStatus("本地错误：${event.optString("message", "未知错误").take(100)}")
                            }
                        }
                    }

                    override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                        socket = null
                        captureRunning = false
                        onStatus("连接失败: ${t.message}")
                        Log.e(tag, "websocket failure", t)
                    }
                }
            )
        }

        private fun startCapture() {
            if (captureRunning) return
            val minBuffer = AudioRecord.getMinBufferSize(
                inputSampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            recorder = AudioRecord(
                // Voice communication enables the phone's built-in echo/noise
                // processing, which is important because TTS plays locally.
                MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                inputSampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                maxOf(minBuffer * 2, frameBytes * 4)
            )
            recorder?.startRecording()
            captureRunning = true
            turnInFlight = false
            resetVad = true
            onStatus("等待说话")
            thread(name = "voice-capture") {
                var vad = SileroVad(context)
                var cooldownFrames = 0
                val buffer = ByteArray(frameBytes)
                try {
                    while (sessionActive) {
                        val count = recorder?.read(buffer, 0, buffer.size, AudioRecord.READ_BLOCKING) ?: -1
                        if (count == buffer.size) {
                            if (resetVad) {
                                vad.reset()
                                resetVad = false
                                // Let the tail of the just-played local TTS
                                // drain before VAD starts a new turn.
                                cooldownFrames = 12 // about 384 ms
                            }
                            // Half-duplex turn handling: do not feed the phone's
                            // own TTS playback back into VAD while a turn is busy.
                            if (turnInFlight) continue
                            if (cooldownFrames > 0) {
                                cooldownFrames--
                                continue
                            }
                            val frame = buffer.copyOf()
                            when (val event = vad.accept(frame)) {
                                is SileroVad.Decision.SpeechStarted -> {
                                    Log.d(tag, "Silero VAD speech started")
                                    onStatus("正在听…")
                                    event.frames.forEach { sendAudio(it) }
                                }
                                is SileroVad.Decision.SpeechFrame -> sendAudio(event.frame)
                                SileroVad.Decision.SpeechEnded -> {
                                    Log.d(tag, "Silero VAD speech ended; committing audio")
                                    turnInFlight = true
                                    socket?.send("{\"type\":\"input_audio_buffer.commit\"}")
                                    onStatus("识别中…")
                                }
                                SileroVad.Decision.Silence -> Unit
                            }
                        } else if (count < 0) {
                            Log.e(tag, "AudioRecord.read failed: $count")
                            onStatus("麦克风读取失败: $count")
                            break
                        }
                    }
                } finally {
                    vad.close()
                    captureRunning = false
                }
            }
        }

        private fun sendAudio(frame: ByteArray) {
            val audio = Base64.getEncoder().encodeToString(frame)
            socket?.send("{\"type\":\"input_audio_buffer.append\",\"audio\":\"$audio\"}")
        }

        private fun playAudioDelta(delta: String) {
            if (delta.isEmpty()) return
            runCatching {
                val pcm = Base64.getDecoder().decode(delta)
                player?.write(pcm, 0, pcm.size)
            }
        }

        fun stopSession() {
            sessionActive = false
            turnInFlight = false
            resetVad = true
            recorder?.stop()
            recorder?.release()
            recorder = null
            player?.stop()
            player?.release()
            player = null
            socket?.close(1000, "session stopped")
            socket = null
            onStatus("已停止")
        }
    }

}

@Composable
private fun MessageBubble(message: ChatMessage) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (message.role == ChatRole.USER) Arrangement.End else Arrangement.Start
    ) {
        Surface(
            modifier = Modifier
                .fillMaxWidth(0.86f)
                .background(Color.Transparent),
            shape = MaterialTheme.shapes.medium,
            color = if (message.role == ChatRole.USER) {
                Color(0xFFE3F2FD)
            } else {
                Color(0xFFF1F8E9)
            }
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                Text(
                    text = if (message.role == ChatRole.USER) "你" else "本地模型",
                    style = MaterialTheme.typography.labelMedium,
                    color = Color(0xFF52606D)
                )
                Text(
                    text = message.text,
                    modifier = Modifier.padding(top = 4.dp),
                    style = MaterialTheme.typography.bodyLarge
                )
            }
        }
    }
}
