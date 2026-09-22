package com.example.realtimevoicechat

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AudioEffect
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.os.Build
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject
import java.util.Base64
import java.util.UUID
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/** Continuous microphone/VAD + independent playback; all inference stays local. */
internal class VoiceClient(
    private val context: Context,
    private val onStatus: (String) -> Unit,
    private val onUserText: (String, String) -> Unit,
    private val onAssistantText: (String, String) -> Unit
) {
    private val http = OkHttpClient()
    private val userId: String by lazy {
        val preferences = context.getSharedPreferences("voice_assistant", Context.MODE_PRIVATE)
        preferences.getString("user_id", null) ?: UUID.randomUUID().toString().also {
            preferences.edit().putString("user_id", it).apply()
        }
    }
    @Volatile private var session: Session? = null

    @Synchronized fun startSession() {
        session?.stop()
        val next = Session()
        session = next
        try {
            next.start()
        } catch (error: Exception) {
            next.fail("无法启动语音：${error.message}", error)
        }
    }

    @Synchronized fun stopSession() {
        session?.stop()
        session = null
        onStatus("已停止")
    }

    private data class AudioChunk(val turnId: String, val pcm: ByteArray, val done: Boolean = false)

    // Every session owns its threads/resources. A late callback from an old
    // WebSocket can never start or release the next session's microphone.
    private inner class Session {
        private val running = AtomicBoolean(true)
        private val stateLock = Any()
        private val queue = LinkedBlockingQueue<AudioChunk>(1024)
        private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        private val effects = mutableListOf<AudioEffect>()
        private var previousMode = audioManager.mode
        private var previousSpeaker = false
        private var previousDevice: AudioDeviceInfo? = null
        private var routeConfigured = false
        private var socket: WebSocket? = null
        private var recorder: AudioRecord? = null
        private var player: AudioTrack? = null
        private var captureThread: Thread? = null
        private var playbackThread: Thread? = null
        private var turnId: String? = null
        private var userSpeaking = false
        private var playing = false
        private var writtenFrames = 0L

        private fun isActive() = running.get() && session === this

        fun start() {
            configureRoute()
            val minBuffer = AudioTrack.getMinBufferSize(24000, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
            require(minBuffer > 0) { "手机不支持 24kHz 播放" }
            player = AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(AudioFormat.Builder().setSampleRate(24000)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setBufferSizeInBytes(minBuffer)
                .setTransferMode(AudioTrack.MODE_STREAM).build()
            check(player?.state == AudioTrack.STATE_INITIALIZED) { "播放器初始化失败" }
            player?.play()
            playbackThread = thread(name = "voice-playback") { playbackLoop() }
            onStatus("连接本地服务…")
            socket = http.newWebSocket(Request.Builder().url("ws://192.168.0.2:8000/ws/realtime").build(),
                object : WebSocketListener() {
                    override fun onOpen(webSocket: WebSocket, response: Response) {
                        synchronized(stateLock) {
                            if (!isActive()) { webSocket.close(1000, "session stopped"); return }
                            socket = webSocket
                            send(JSONObject().put("type", "session.update").put("user_id", userId))
                            try { startCapture() } catch (error: Exception) { fail("麦克风启动失败：${error.message}", error) }
                        }
                    }

                    override fun onMessage(webSocket: WebSocket, text: String) {
                        val event = runCatching { JSONObject(text) }.getOrNull() ?: return
                        synchronized(stateLock) {
                            if (!isActive()) return
                            val type = event.optString("type")
                            val id = event.optString("turn_id")
                            // Ignore old audio, done, errors and cancel acks even
                            // when they arrive after a NEW utterance was committed.
                            if (id.isNotEmpty() && id != turnId) return
                            when (type) {
                                "conversation.transcript" -> {
                                    val transcript = event.optString("text").trim()
                                    if (transcript.isNotEmpty()) onUserText(id, transcript)
                                    onStatus(if (transcript.isEmpty()) "没有识别到清晰语音" else "本地识别完成，正在理解…")
                                }
                                "response.text.done" -> {
                                    val reply = event.optString("text").trim()
                                    if (reply.isNotEmpty()) onAssistantText(id, reply)
                                    onStatus("正在生成语音…")
                                }
                                "response.audio.delta" -> {
                                    if (id.isEmpty() || userSpeaking) return
                                    val pcm = runCatching { Base64.getDecoder().decode(event.optString("delta")) }.getOrNull() ?: return
                                    if (!queue.offer(AudioChunk(id, pcm))) fail("播放队列过长，请重新开始会话")
                                }
                                "response.audio.done" -> {
                                    if (id.isNotEmpty() && !queue.offer(AudioChunk(id, byteArrayOf(), done = true))) {
                                        fail("播放队列过长，请重新开始会话")
                                    }
                                }
                                "response.cancelled" -> {
                                    invalidatePlayback()
                                    onStatus(if (userSpeaking) "正在听…" else "等待说话（可随时打断）")
                                }
                                "error" -> {
                                    invalidatePlayback()
                                    onStatus("本地错误：${event.optString("message", "未知错误").take(100)}")
                                }
                            }
                        }
                    }

                    override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                        if (isActive()) fail("连接失败：${t.message}", t)
                    }

                    override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                        if (isActive()) fail("本地服务已断开，请重新开始会话")
                    }
                })
        }

        @Suppress("DEPRECATION")
        private fun configureRoute() {
            routeConfigured = true
            previousMode = audioManager.mode
            audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
            if (Build.VERSION.SDK_INT >= 31) {
                previousDevice = audioManager.communicationDevice
                // Keep an external headset if selected; otherwise use speaker.
                if (previousDevice == null || previousDevice?.type == AudioDeviceInfo.TYPE_BUILTIN_EARPIECE ||
                    previousDevice?.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER) {
                    audioManager.availableCommunicationDevices.firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER }
                        ?.let { audioManager.setCommunicationDevice(it) }
                }
            } else {
                previousSpeaker = audioManager.isSpeakerphoneOn
                audioManager.isSpeakerphoneOn = true
            }
        }

        @Suppress("DEPRECATION")
        private fun restoreRoute() {
            if (!routeConfigured) return
            runCatching {
                if (Build.VERSION.SDK_INT >= 31) {
                    val old = previousDevice
                    if (old != null) audioManager.setCommunicationDevice(old) else audioManager.clearCommunicationDevice()
                } else audioManager.isSpeakerphoneOn = previousSpeaker
                audioManager.mode = previousMode
            }
            routeConfigured = false
        }

        private fun enableEffect(name: String, create: () -> AudioEffect?) {
            runCatching {
                val effect = create()
                if (effect != null) {
                    effects.add(effect)
                    effect.enabled = true
                }
                Log.i(TAG, "$name enabled=${effect?.enabled == true}")
            }.onFailure { Log.w(TAG, "$name unavailable", it) }
        }

        private fun startCapture() {
            check(context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                "麦克风权限未授予"
            }
            val minBuffer = AudioRecord.getMinBufferSize(SileroVad.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            require(minBuffer > 0) { "手机不支持 16kHz 录音" }
            val input = AudioRecord(MediaRecorder.AudioSource.VOICE_COMMUNICATION, SileroVad.SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuffer * 2, FRAME_BYTES * 4))
            recorder = input
            check(input.state == AudioRecord.STATE_INITIALIZED) { "录音器初始化失败" }
            enableEffect("AEC") { if (AcousticEchoCanceler.isAvailable()) AcousticEchoCanceler.create(input.audioSessionId) else null }
            enableEffect("NS") { if (NoiseSuppressor.isAvailable()) NoiseSuppressor.create(input.audioSessionId) else null }
            enableEffect("AGC") { if (AutomaticGainControl.isAvailable()) AutomaticGainControl.create(input.audioSessionId) else null }
            input.startRecording()
            onStatus("等待说话（可随时打断）")
            captureThread = thread(name = "voice-capture") {
                try {
                    SileroVad(context).use { vad ->
                        val buffer = ByteArray(FRAME_BYTES)
                        var filled = 0
                        while (isActive()) {
                            val count = input.read(buffer, filled, buffer.size - filled, AudioRecord.READ_BLOCKING)
                            if (count < 0) { if (isActive()) error("麦克风读取失败：$count"); break }
                            filled += count
                            if (filled != buffer.size) continue
                            filled = 0
                            // No turn-in-flight gate, cooldown, or remote reset:
                            // Silero processes EVERY frame, including during TTS.
                            val decision = vad.accept(buffer.copyOf())
                            synchronized(stateLock) {
                                if (!isActive()) return@thread
                                when (decision) {
                                    is SileroVad.Decision.SpeechStarted -> {
                                        val interrupted = turnId != null
                                        invalidatePlayback()
                                        userSpeaking = true
                                        send(JSONObject().put("type", "input_audio_buffer.speech_started"))
                                        decision.frames.forEach { sendAudio(it) }
                                        onStatus("正在听…")
                                        Log.i(TAG, "Silero speech started; bargeIn=$interrupted")
                                    }
                                    is SileroVad.Decision.SpeechFrame -> sendAudio(decision.frame)
                                    SileroVad.Decision.SpeechEnded -> {
                                        userSpeaking = false
                                        turnId = UUID.randomUUID().toString()
                                        send(JSONObject().put("type", "input_audio_buffer.commit").put("turn_id", turnId))
                                        onStatus("识别中…（可继续说话）")
                                        Log.i(TAG, "Silero speech ended; commit=$turnId")
                                    }
                                    SileroVad.Decision.SpeechDiscarded -> {
                                        userSpeaking = false
                                        send(JSONObject().put("type", "input_audio_buffer.clear"))
                                        onStatus("等待说话（可随时打断）")
                                    }
                                    SileroVad.Decision.Silence -> Unit
                                }
                            }
                        }
                    }
                } catch (error: Exception) {
                    if (isActive()) fail("语音检测失败：${error.message}", error)
                } finally {
                    synchronized(stateLock) { releaseCapture() }
                }
            }
        }

        private fun send(event: JSONObject) {
            if (isActive() && socket?.send(event.toString()) != true) fail("发送失败，请重新开始会话")
        }

        private fun sendAudio(frame: ByteArray) = send(JSONObject().put("type", "input_audio_buffer.append")
            .put("audio", Base64.getEncoder().encodeToString(frame)))

        // stateLock protects invalidation and each NON-BLOCKING write together.
        // No old chunk can be written after the immediate pause/flush operation.
        private fun invalidatePlayback() {
            turnId = null
            queue.clear()
            playing = false
            player?.pause()
            player?.flush()
            writtenFrames = 0L
            player?.play()
        }

        private fun playbackLoop() {
            try {
                while (isActive()) {
                    val chunk = queue.poll(100, TimeUnit.MILLISECONDS) ?: continue
                    var offset = 0
                    while (isActive()) {
                        val completed = synchronized(stateLock) {
                            val track = player
                            if (track == null || chunk.turnId != turnId || userSpeaking) true
                            else if (chunk.done) {
                                val played = track.playbackHeadPosition.toLong() and 0xffffffffL
                                if (played >= writtenFrames) {
                                    turnId = null
                                    playing = false
                                    onStatus("等待说话（可随时打断）")
                                    true
                                } else false
                            } else {
                                if (!playing) { playing = true; onStatus("正在回答（可直接说话打断）") }
                                val count = track.write(chunk.pcm, offset, chunk.pcm.size - offset, AudioTrack.WRITE_NON_BLOCKING)
                                check(count >= 0) { "音频播放失败：$count" }
                                offset += count
                                writtenFrames += count / 2
                                offset == chunk.pcm.size
                            }
                        }
                        if (completed) break
                        Thread.sleep(5)
                    }
                }
            } catch (error: Exception) {
                if (isActive()) fail("播放失败：${error.message}", error)
            }
        }

        fun fail(message: String, error: Throwable? = null) {
            if (!isActive()) return
            Log.e(TAG, message, error)
            stop()
            if (session === this) onStatus(message)
        }

        private fun releaseCapture() {
            effects.forEach { runCatching { it.release() } }
            effects.clear()
            runCatching { recorder?.release() }
            recorder = null
        }

        fun stop() {
            synchronized(stateLock) {
                if (!running.getAndSet(false)) return
                turnId = null
                queue.clear()
                socket?.close(1000, "session stopped")
                socket = null
                runCatching { recorder?.stop() }
                // The capture thread releases its own recorder after read exits.
                if (captureThread == null) releaseCapture()
                runCatching { player?.pause(); player?.flush(); player?.release() }
                player = null
                restoreRoute()
            }
        }
    }

    companion object {
        private const val TAG = "RealtimeVoiceChat"
        private const val FRAME_BYTES = SileroVad.FRAME_SAMPLES * 2
    }
}
