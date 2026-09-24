"""本地实时会话的 WebSocket 状态机。

入口模块只负责路由和依赖组装；本模块负责单条本地会话的输入缓冲、历史记录以及
回合取消。把这些状态集中到一个对象后，协议循环不再和模型、语音合成实现耦合，
也方便以后为不同客户端复用同一套回合生命周期。
"""

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from contextlib import suppress

from fastapi import WebSocket, WebSocketDisconnect


SendEvent = Callable[[dict], Awaitable[None]]
TurnHandler = Callable[[bytes, str, list[dict], str], Awaitable[None]]


class LocalRealtimeSession:
    """维护一条本地 provider WebSocket 会话的状态和生命周期。"""

    def __init__(
        self,
        websocket: WebSocket,
        send_event: SendEvent,
        turn_handler: TurnHandler,
        *,
        default_user_id: str,
        max_history_turns: int,
        max_audio_buffer_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        self.websocket = websocket
        self.send_event = send_event
        self.turn_handler = turn_handler
        self.user_id = default_user_id
        self.max_history_turns = max(1, max_history_turns)
        self.max_audio_buffer_bytes = max(1, max_audio_buffer_bytes)
        self.pcm_buffer = bytearray()
        self.history: list[dict] = []
        self.turn_index = 0
        self.current_turn_task: asyncio.Task | None = None
        self.current_turn_id: str | None = None

    async def run(self) -> None:
        """读取并处理客户端事件，断线时保证当前回合被取消。"""
        try:
            while True:
                event = json.loads(await self.websocket.receive_text())
                await self.handle_event(event)
        except WebSocketDisconnect:
            pass
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
            # 协议错误不能让服务端任务带着未观察到的异常退出；通知后关闭当前
            # 会话，避免继续消费一个状态未知的音频缓冲区。
            with suppress(Exception):
                await self.send_event({"type": "error", "message": f"无效会话事件：{error}"})
        finally:
            await self.cancel_current_turn(notify=False)

    async def handle_event(self, event: dict) -> None:
        """处理一个已经解码的客户端协议事件。"""
        event_type = event.get("type")
        if event_type == "input_audio_buffer.append":
            self._append_audio(event.get("audio", ""))
        elif event_type == "input_audio_buffer.speech_started":
            await self.cancel_current_turn()
            self.pcm_buffer.clear()
        elif event_type == "input_audio_buffer.clear":
            self.pcm_buffer.clear()
        elif event_type == "response.cancel":
            await self.cancel_current_turn()
        elif event_type == "input_audio_buffer.commit":
            await self.commit(event.get("turn_id"))
        elif event_type == "session.update":
            await self.update_session(event.get("user_id"))

    def _append_audio(self, encoded_audio: str) -> None:
        """追加一帧 Base64 PCM；空值按协议视为没有新音频。"""
        if encoded_audio:
            frame = base64.b64decode(encoded_audio, validate=True)
            if len(self.pcm_buffer) + len(frame) > self.max_audio_buffer_bytes:
                raise ValueError(f"音频缓冲区超过 {self.max_audio_buffer_bytes} 字节限制")
            self.pcm_buffer.extend(frame)

    async def commit(self, requested_turn_id: object) -> None:
        """提交当前 PCM 并启动一个新的可取消回合。"""
        pcm = bytes(self.pcm_buffer)
        self.pcm_buffer.clear()
        self.turn_index += 1
        turn_id = str(requested_turn_id or f"turn-{self.turn_index}")[:128]
        await self.cancel_current_turn(notify=False)
        self.current_turn_id = turn_id
        self.current_turn_task = asyncio.create_task(
            self._run_turn(pcm, turn_id),
            name=f"local-turn-{turn_id}",
        )

    async def _run_turn(self, pcm: bytes, turn_id: str) -> None:
        """执行回合处理器。

        即使回合已经完成也保留 ``current_turn_id``，这样客户端在已经收到音频后
        说话仍能得到明确的 ``response.cancelled`` 确认。下一次提交或显式取消时，
        ``cancel_current_turn`` 会统一清理这两个引用。
        """
        await self.turn_handler(pcm, turn_id, self.history, self.user_id)
        # 历史窗口是会话状态机的不变量；即使未来替换 turn_handler，也不会
        # 让单个连接无限增长。当前的 local_turn 仍会提前裁剪一次，这里是边界兜底。
        del self.history[:-(self.max_history_turns * 2)]

    async def update_session(self, requested_user_id: object) -> None:
        """切换用户时清空会话历史，并回送统一确认事件。"""
        requested = str(requested_user_id or "").strip()
        if requested:
            requested = requested[:128]
            if requested != self.user_id:
                # 用户切换必须先终止旧用户的回合，否则旧任务可能在清空后的
                # history 上继续追加内容，或用旧身份写入长期记忆。
                await self.cancel_current_turn(notify=False)
                self.history.clear()
            self.user_id = requested
        await self.send_event({"type": "session.updated", "provider": "local"})

    async def cancel_current_turn(self, *, notify: bool = True) -> None:
        """取消当前回合并可选地通知客户端；重复调用是安全的。"""
        task = self.current_turn_task
        turn_id = self.current_turn_id
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if notify and turn_id is not None:
            await self.send_event({"type": "response.cancelled", "turn_id": turn_id})
        self.current_turn_task = None
        self.current_turn_id = None
