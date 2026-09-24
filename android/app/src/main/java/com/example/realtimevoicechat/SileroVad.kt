package com.example.realtimevoicechat

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.ArrayDeque

/**
 * Silero VAD v6 的流式 ONNX 封装。
 * 输入必须是 16 kHz、单声道、little-endian PCM16，每帧 512 个采样（1024 字节）。
 * 类内部保留模型 recurrent state 和 64 个采样的上下文，使相邻帧的判断连续。
 */
class SileroVad(context: Context) : AutoCloseable {
    /** 一帧输入后的语音状态；SpeechStarted 携带预滚动帧以免吞掉起始音节。 */
    sealed interface Decision {
        data class SpeechStarted(val frames: List<ByteArray>) : Decision
        data class SpeechFrame(val frame: ByteArray) : Decision
        data object SpeechEnded : Decision
        data object SpeechDiscarded : Decision
        data object Silence : Decision
    }

    /** ONNX Runtime 环境与模型会话，整个 VAD 实例复用一次。 */
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    /** Silero 模型的 recurrent hidden state，下一帧会接着使用。 */
    private var state = FloatArray(2 * 128)
    /** 前一帧末尾上下文采样，用于消除分帧边界造成的概率抖动。 */
    private var audioContext = FloatArray(CONTEXT_SAMPLES)
    private val preRoll = ArrayDeque<ByteArray>()
    private var pendingSpeech = 0
    private var silenceFrames = 0
    private var speechFrames = 0
    private var speaking = false

    /** 从 assets 读取随 APK 打包的 ONNX 模型，并限制 ONNX 推理线程数。 */
    init {
        val model = context.assets.open("silero_vad.onnx").use { it.readBytes() }
        session = OrtSession.SessionOptions().use { options ->
            options.apply {
                setIntraOpNumThreads(1)
                setInterOpNumThreads(1)
            }
            environment.createSession(model, options)
        }
    }

    /** 消费一帧 PCM；通过连续阈值和最短时长抑制瞬时噪声。 */
    fun accept(frame: ByteArray): Decision {
        require(frame.size == FRAME_SAMPLES * 2) { "Silero VAD expects 512 PCM16 samples" }
        val probability = probability(frame)

        if (!speaking) {
            preRoll.addLast(frame.copyOf())
            while (preRoll.size > PRE_ROLL_FRAMES) preRoll.removeFirst()
            if (probability >= SPEECH_THRESHOLD) pendingSpeech++ else pendingSpeech = 0
            if (pendingSpeech >= SPEECH_START_FRAMES) {
                speaking = true
                pendingSpeech = 0
                silenceFrames = 0
                speechFrames = SPEECH_START_FRAMES
                return Decision.SpeechStarted(preRoll.toList())
            }
            return Decision.Silence
        }

        if (probability < SILENCE_THRESHOLD) {
            silenceFrames++
            if (silenceFrames >= SILENCE_END_FRAMES) {
                speaking = false
                silenceFrames = 0
                val validSpeech = speechFrames >= MIN_SPEECH_FRAMES
                speechFrames = 0
                preRoll.clear()
                return if (validSpeech) Decision.SpeechEnded else Decision.SpeechDiscarded
            }
        } else {
            speechFrames++
            silenceFrames = 0
        }
        return Decision.SpeechFrame(frame)
    }

    /** 清空模型状态及当前语音段，供重新开始一轮录音时调用。 */
    fun reset() {
        state.fill(0f)
        audioContext.fill(0f)
        preRoll.clear()
        pendingSpeech = 0
        silenceFrames = 0
        speechFrames = 0
        speaking = false
    }

    /** 将 PCM16 归一化为 [-1,1] 浮点数并执行一次 ONNX 推理。 */
    private fun probability(frame: ByteArray): Float {
        val input = FloatArray(CONTEXT_SAMPLES + FRAME_SAMPLES)
        audioContext.copyInto(input, 0)
        var sourceIndex = 0
        var targetIndex = CONTEXT_SAMPLES
        while (sourceIndex + 1 < frame.size) {
            val sample = ((frame[sourceIndex + 1].toInt() shl 8) or (frame[sourceIndex].toInt() and 0xff)).toShort()
            input[targetIndex++] = sample / 32768.0f
            sourceIndex += 2
        }

        // 全双工会话会持续调用这里；每帧关闭 native tensor，避免长时间通话泄漏 ONNX 缓冲。
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(input),
            longArrayOf(1, input.size.toLong())).use { inputTensor ->
            OnnxTensor.createTensor(environment, FloatBuffer.wrap(state),
                longArrayOf(2, 1, 128)).use { stateTensor ->
                OnnxTensor.createTensor(environment, LongBuffer.wrap(longArrayOf(SAMPLE_RATE.toLong())),
                    longArrayOf()).use { sampleRateTensor ->
                    session.run(mapOf("input" to inputTensor, "state" to stateTensor, "sr" to sampleRateTensor)).use { result ->
                        val output = (result[0] as OnnxTensor).floatBuffer.get(0)
                        val nextState = (result[1] as OnnxTensor).floatBuffer
                        nextState.rewind()
                        nextState.get(state)
                        audioContext = input.copyOfRange(FRAME_SAMPLES, input.size)
                        return output
                    }
                }
            }
        }
    }

    /** 释放 ONNX 会话持有的 native 资源；调用后此 VAD 不应再接收音频。 */
    override fun close() {
        session.close()
    }

    companion object {
        /** VAD 与 AudioRecord 必须匹配的采样率。 */
        const val SAMPLE_RATE = 16_000
        /** 每个输入帧的采样数；512/16000=32 ms。 */
        const val FRAME_SAMPLES = 512
        private const val CONTEXT_SAMPLES = 64
        /** 检测到开口时回送的历史帧数，减少首音节被阈值延迟吞掉的概率。 */
        private const val PRE_ROLL_FRAMES = 6 // 192 ms
        /** 连续超过语音阈值的帧数，确认用户真的开始说话。 */
        private const val SPEECH_START_FRAMES = 3 // 96 ms
        /** 连续低于静音阈值的帧数，确认语音段已经结束。 */
        private const val SILENCE_END_FRAMES = 12 // 384 ms
        /** 语音段最短有效长度；更短的段会作为噪声丢弃。 */
        private const val MIN_SPEECH_FRAMES = 8 // 256 ms voiced, excluding trailing silence
        /** 开始/结束使用滞回阈值，避免概率在边界抖动导致频繁切换。 */
        private const val SPEECH_THRESHOLD = 0.65f
        private const val SILENCE_THRESHOLD = 0.35f
    }
}
