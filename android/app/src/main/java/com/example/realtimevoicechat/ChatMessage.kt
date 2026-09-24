package com.example.realtimevoicechat

/** 聊天消息的发送方，用于决定气泡的对齐方式和颜色。 */
enum class ChatRole {
    USER,
    ASSISTANT
}

/** Compose 列表中展示的一条已经完成的对话文本。 */
data class ChatMessage(
    /** 由回合 ID 和发送方组成的稳定键，避免列表刷新时错位。 */
    val id: String,
    /** 消息来自用户还是本地助手。 */
    val role: ChatRole,
    /** 已完成的识别文本或助手回复文本。 */
    val text: String
)
