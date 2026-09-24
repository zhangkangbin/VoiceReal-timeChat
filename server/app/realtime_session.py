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
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect


SendEvent = Callable[[dict], Awaitable[None]]
TurnHandler = Callable[[bytes, str, list[dict], str, Callable[[str, str], Awaitable[dict]]], Awaitable[None]]


class CameraCaptureError(RuntimeError):
    """相机请求失败或客户端返回了无效照片。"""

    def __init__(self, message: str, *, user_message: str | None = None):
        super().__init__(message)
        self.user_message = user_message or message


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
        max_image_bytes: int = 4 * 1024 * 1024,
        camera_timeout_seconds: int = 30,
    ) -> None:
        self.websocket = websocket
        self.send_event = send_event
        self.turn_handler = turn_handler
        self.user_id = default_user_id
        self.max_history_turns = max(1, max_history_turns)
        self.max_audio_buffer_bytes = max(1, max_audio_buffer_bytes)
        self.max_image_bytes = max(1, max_image_bytes)
        self.camera_timeout_seconds = max(1, camera_timeout_seconds)
        self.pcm_buffer = bytearray()
        self.history: list[dict] = []
        self.turn_index = 0
        self.current_turn_task: asyncio.Task | None = None
        self.current_turn_id: str | None = None
        self.pending_camera: dict | None = None

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
        elif event_type == "camera.capture.result":
            await self._resolve_camera(event, error=False)
        elif event_type == "camera.capture.error":
            await self._resolve_camera(event, error=True)

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

    async def request_camera(self, turn_id: str, reason: str = "describe_scene") -> dict:
        """请求 Android 拍照并等待同一回合、同一请求 ID 的结果。"""
        if self.pending_camera is not None:
            raise CameraCaptureError("已有相机请求正在等待")
        request_id = f"camera-{uuid4().hex}"
        future = asyncio.get_running_loop().create_future()
        self.pending_camera = {"request_id": request_id, "turn_id": turn_id, "future": future}
        try:
            await self.send_event({
                "type": "camera.capture.request",
                "turn_id": turn_id,
                "request_id": request_id,
                "reason": reason[:80],
                "lens": "back",
                "max_width": 1280,
                "max_height": 1280,
                "quality": 85,
            })
            return await asyncio.wait_for(asyncio.shield(future), timeout=self.camera_timeout_seconds)
        except asyncio.TimeoutError as error:
            raise CameraCaptureError("相机请求超时", user_message="拍照等待超时，请重试。") from error
        finally:
            if not future.done():
                future.cancel()
            if self.pending_camera is not None and self.pending_camera.get("request_id") == request_id:
                self.pending_camera = None

    async def _resolve_camera(self, event: dict, *, error: bool) -> None:
        """校验相机回调的回合/请求绑定，阻止迟到照片污染新回合。"""
        pending = self.pending_camera
        if pending is None:
            return
        if (str(event.get("request_id") or "") != pending["request_id"] or
                str(event.get("turn_id") or "") != pending["turn_id"]):
            return
        future = pending["future"]
        if future.done():
            return
        if error:
            message = str(event.get("message") or "拍照已取消")[:200]
            future.set_exception(CameraCaptureError(message, user_message=message))
            return
        mime_type = str(event.get("mime_type") or "image/jpeg").lower()
        if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
            future.set_exception(CameraCaptureError("不支持的图片格式", user_message="照片格式不受支持，请重试。"))
            return
        encoded = event.get("image_base64")
        if not isinstance(encoded, str) or not encoded:
            future.set_exception(CameraCaptureError("照片内容为空", user_message="没有收到照片，请重试。"))
            return
        # Reject oversized Base64 before decoding so a malicious JSON frame cannot
        # allocate an unbounded temporary byte string.
        if len(encoded) > ((self.max_image_bytes * 4) // 3) + 8:
            future.set_exception(CameraCaptureError("照片超过大小限制", user_message="照片太大，请重新拍一张。"))
            return
        try:
            image = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            future.set_exception(CameraCaptureError("照片不是有效的 Base64", user_message="照片上传失败，请重试。"))
            return
        if not image or len(image) > self.max_image_bytes:
            future.set_exception(CameraCaptureError("照片超过大小限制", user_message="照片太大，请重新拍一张。"))
            return
        future.set_result({"bytes": image, "mime_type": mime_type})

    def _cancel_pending_camera(self) -> None:
        pending = self.pending_camera
        self.pending_camera = None
        if pending is not None:
            future = pending.get("future")
            if future is not None and not future.done():
                future.cancel()

    async def _run_turn(self, pcm: bytes, turn_id: str) -> None:
        """执行回合处理器。

        即使回合已经完成也保留 ``current_turn_id``，这样客户端在已经收到音频后
        说话仍能得到明确的 ``response.cancelled`` 确认。下一次提交或显式取消时，
        ``cancel_current_turn`` 会统一清理这两个引用。
        """
        await self.turn_handler(pcm, turn_id, self.history, self.user_id, self.request_camera)
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
        self._cancel_pending_camera()
        if notify and turn_id is not None:
            await self.send_event({"type": "response.cancelled", "turn_id": turn_id})
        self.current_turn_task = None
        self.current_turn_id = None
