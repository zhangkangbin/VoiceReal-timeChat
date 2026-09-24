package com.example.realtimevoicechat

/**
 * Runtime configuration for a voice client.
 *
 * Keeping transport configuration outside [VoiceClient] gives the session
 * lifecycle code a stable boundary: production can use the default LAN
 * endpoint while tests, an emulator, or a future settings screen can inject
 * another endpoint without changing audio/session behavior.
 */
internal data class VoiceClientConfig(
    val webSocketUrl: String = DEFAULT_WEBSOCKET_URL
) {
    init {
        require(webSocketUrl.startsWith("ws://") || webSocketUrl.startsWith("wss://")) {
            "Voice WebSocket URL must use ws:// or wss://"
        }
    }

    companion object {
        /** Default endpoint used by the current physical-device setup. */
        const val DEFAULT_WEBSOCKET_URL = "ws://192.168.0.2:8000/ws/realtime"
    }
}
