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
