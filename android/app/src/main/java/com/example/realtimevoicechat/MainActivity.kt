package com.example.realtimevoicechat

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
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

/** 语音聊天界面入口：申请录音权限并把 VoiceClient 的事件映射为 Compose 状态。 */
class MainActivity : ComponentActivity() {
    /** 系统录音权限请求回调；结果由客户端在真正开始录音时再次校验。 */
    private val requestPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    /** 拍照权限结果；拿到权限后立即继续当前一次模型发起的拍照请求。 */
    private val requestCameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        val request = pendingCameraRequest
        if (granted && request != null) launchCamera(request)
        else if (!granted) {
            pendingCameraRequest = null
            voiceClient?.sendCameraError(request?.first.orEmpty(), request?.second.orEmpty(), "permission_denied", "没有相机权限")
        }
    }

    /** TakePicturePreview 不把照片落盘，适合把本次照片直接压缩后上传给服务端。 */
    private val takePicturePreview = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
        val request = pendingCameraRequest
        pendingCameraRequest = null
        if (request == null) return@registerForActivityResult
        if (bitmap == null) {
            voiceClient?.sendCameraError(request.first, request.second, "cancelled", "拍照已取消")
        } else {
            voiceClient?.sendCameraImage(request.first, request.second, bitmap)
        }
    }

    private var pendingCameraRequest: Triple<String, String, String>? = null
    private var voiceClient: VoiceClient? = null

    /** 创建界面；先请求录音权限，再安装 Compose 内容，避免无权限直接启动采集线程。 */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermission.launch(Manifest.permission.RECORD_AUDIO)
        }
        setContent { VoiceScreen() }
    }

    /** 在主线程启动系统相机；request 中依次为 turn_id、request_id、拍照原因。 */
    private fun launchCamera(request: Triple<String, String, String>) {
        pendingCameraRequest = request
        runCatching { takePicturePreview.launch(null) }
            .onFailure {
                pendingCameraRequest = null
                voiceClient?.sendCameraError(request.first, request.second, "camera_unavailable", "无法打开相机")
            }
    }

    /** 响应服务端的相机回调请求，并把系统权限/取消状态回传给同一回合。 */
    private fun onCameraCaptureRequested(turnId: String, requestId: String, reason: String) {
        runOnUiThread {
            val request = Triple(turnId, requestId, reason)
            pendingCameraRequest = request
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                launchCamera(request)
            } else {
                requestCameraPermission.launch(Manifest.permission.CAMERA)
            }
        }
    }

    @Composable
    /** 显示连接状态、最近消息以及开始/停止会话按钮。 */
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
                onAssistantText = { turnId, text, expression ->
                    runOnUiThread {
                        if (text.isNotBlank()) {
                            messages = (messages + ChatMessage("$turnId-assistant", ChatRole.ASSISTANT, text, expression)).takeLast(100)
                        }
                    }
                },
                onCameraCaptureRequested = { turnId, requestId, reason ->
                    onCameraCaptureRequested(turnId, requestId, reason)
                }
            )
        }
        LaunchedEffect(client) { voiceClient = client }
        // Activity 销毁或 Compose 离开树时释放 WebSocket、录音器、播放器和 VAD 线程。
        DisposableEffect(client) {
            onDispose { client.stopSession() }
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

}

/** 根据发送方将文本显示为左右对齐、颜色不同的消息气泡。 */
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
                message.expression?.takeIf { message.role == ChatRole.ASSISTANT }?.let { expression ->
                    Surface(
                        modifier = Modifier.padding(top = 8.dp),
                        shape = MaterialTheme.shapes.small,
                        color = MaterialTheme.colorScheme.secondaryContainer
                    ) {
                        Text(
                            text = "表情：${expression.name}",
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            style = MaterialTheme.typography.labelMedium
                        )
                    }
                }
            }
        }
    }
}
