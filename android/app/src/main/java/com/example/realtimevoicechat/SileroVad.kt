package com.example.realtimevoicechat

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.ArrayDeque

/** Streaming Silero VAD v6 ONNX wrapper for 16 kHz PCM16 audio. */
class SileroVad(context: Context) : AutoCloseable {
    sealed interface Decision {
        data class SpeechStarted(val frames: List<ByteArray>) : Decision
        data class SpeechFrame(val frame: ByteArray) : Decision
        data object SpeechEnded : Decision
        data object SpeechDiscarded : Decision
        data object Silence : Decision
    }

    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private var state = FloatArray(2 * 128)
    private var audioContext = FloatArray(CONTEXT_SAMPLES)
    private val preRoll = ArrayDeque<ByteArray>()
    private var pendingSpeech = 0
    private var silenceFrames = 0
    private var speechFrames = 0
    private var speaking = false

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

    fun reset() {
        state.fill(0f)
        audioContext.fill(0f)
        preRoll.clear()
        pendingSpeech = 0
        silenceFrames = 0
        speechFrames = 0
        speaking = false
    }

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

        // Full duplex keeps this path running continuously; close every native
        // tensor per frame instead of leaking ONNX buffers during long calls.
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

    override fun close() {
        session.close()
    }

    companion object {
        const val SAMPLE_RATE = 16_000
        const val FRAME_SAMPLES = 512
        private const val CONTEXT_SAMPLES = 64
        private const val PRE_ROLL_FRAMES = 6 // 192 ms
        private const val SPEECH_START_FRAMES = 3 // 96 ms
        private const val SILENCE_END_FRAMES = 12 // 384 ms
        private const val MIN_SPEECH_FRAMES = 8 // 256 ms voiced, excluding trailing silence
        private const val SPEECH_THRESHOLD = 0.65f
        private const val SILENCE_THRESHOLD = 0.35f
    }
}
