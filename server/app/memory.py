"""Local, user-controlled long-term memory for the voice assistant."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

from dotenv import load_dotenv

load_dotenv()


DEFAULT_USER_ID = os.getenv("MEMORY_USER_ID", "default-user")
MAX_MEMORY_VALUE_LENGTH = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Memory:
    id: int
    user_id: str
    type: str
    key: str
    value: str
    confidence: float
    importance: float
    source_turn_id: str | None
    created_at: str
    updated_at: str
    expires_at: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "importance": self.importance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
        }


class MemoryStore:
    """SQLite persistence with one connection per operation."""

    def __init__(self, path: str | Path | None = None):
        default_path = Path(__file__).resolve().parents[1] / "data" / "memory.db"
        self.path = Path(path or os.getenv("MEMORY_DB_PATH", str(default_path)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source_turn_id TEXT,
                    confidence REAL NOT NULL DEFAULT 0.8,
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    status TEXT NOT NULL DEFAULT 'active'
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_active_key
                    ON memories(user_id, type, key) WHERE status = 'active';
                CREATE INDEX IF NOT EXISTS idx_memories_user_status
                    ON memories(user_id, status, updated_at);
                """
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"], user_id=row["user_id"], type=row["type"],
            key=row["key"], value=row["value"], confidence=row["confidence"],
            importance=row["importance"], source_turn_id=row["source_turn_id"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            expires_at=row["expires_at"],
        )

    def save_sync(
        self,
        user_id: str,
        memory_type: str,
        key: str,
        value: str,
        *,
        confidence: float = 0.85,
        importance: float = 0.6,
        source_turn_id: str | None = None,
        expires_at: str | None = None,
    ) -> Memory:
        user_id = (user_id or DEFAULT_USER_ID).strip()[:128]
        memory_type = (memory_type or "fact").strip()[:64]
        key = (key or "fact").strip()[:128]
        value = (value or "").strip()[:MAX_MEMORY_VALUE_LENGTH]
        if not value:
            raise ValueError("记忆内容不能为空")
        now = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, created_at FROM memories WHERE user_id=? AND type=? AND key=? AND status='active'",
                (user_id, memory_type, key),
            ).fetchone()
            if row is None:
                # A model may choose a slightly different key for the same
                # fact. Deduplicate identical values within one memory type.
                row = connection.execute(
                    "SELECT id, created_at FROM memories WHERE user_id=? AND type=? AND value=? AND status='active'",
                    (user_id, memory_type, value),
                ).fetchone()
            if row:
                connection.execute(
                    """UPDATE memories SET value=?, confidence=?, importance=?, source_turn_id=?,
                       updated_at=?, expires_at=? WHERE id=?""",
                    (value, max(0.0, min(confidence, 1.0)), max(0.0, min(importance, 1.0)),
                     source_turn_id, now, expires_at, row["id"]),
                )
                memory_id = row["id"]
                created_at = row["created_at"]
            else:
                cursor = connection.execute(
                    """INSERT INTO memories
                       (user_id, type, key, value, source_turn_id, confidence, importance,
                        created_at, updated_at, expires_at, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')""",
                    (user_id, memory_type, key, value, source_turn_id,
                     max(0.0, min(confidence, 1.0)), max(0.0, min(importance, 1.0)),
                     now, now, expires_at),
                )
                memory_id = cursor.lastrowid
                created_at = now
            saved = connection.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return self._from_row(saved)

    def list_sync(self, user_id: str, limit: int = 50) -> list[Memory]:
        now = _now()
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM memories WHERE user_id=? AND status='active'
                   AND (expires_at IS NULL OR expires_at > ?)
                   ORDER BY importance DESC, updated_at DESC LIMIT ?""",
                (user_id or DEFAULT_USER_ID, now, max(1, min(limit, 100))),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def relevant_sync(self, user_id: str, query: str, limit: int = 8) -> list[Memory]:
        memories = self.list_sync(user_id, limit=100)
        query = (query or "").lower()
        if not query:
            return memories[:limit]

        def score(memory: Memory) -> tuple[int, float, str]:
            haystack = f"{memory.key} {memory.value}".lower()
            exact = 4 if memory.key.lower() in query or memory.value.lower() in query else 0
            terms = [term for term in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", query) if len(term) >= 2]
            overlap = sum(1 for term in terms if term in haystack)
            return exact + overlap, memory.importance, memory.updated_at

        return sorted(memories, key=score, reverse=True)[:limit]

    def delete_sync(self, user_id: str, *, memory_id: int | None = None, query: str | None = None,
                    clear_all: bool = False) -> int:
        user_id = user_id or DEFAULT_USER_ID
        with self._connect() as connection:
            if clear_all:
                cursor = connection.execute(
                    "UPDATE memories SET status='deleted', updated_at=? WHERE user_id=? AND status='active'",
                    (_now(), user_id),
                )
                return cursor.rowcount
            if memory_id is not None:
                cursor = connection.execute(
                    "UPDATE memories SET status='deleted', updated_at=? WHERE id=? AND user_id=? AND status='active'",
                    (_now(), memory_id, user_id),
                )
                return cursor.rowcount
            query = (query or "").strip().lower()
            if not query:
                return 0
            cursor = connection.execute(
                """UPDATE memories SET status='deleted', updated_at=?
                   WHERE user_id=? AND status='active' AND (lower(key) LIKE ? OR lower(value) LIKE ?)""",
                (_now(), user_id, f"%{query}%", f"%{query}%"),
            )
            return cursor.rowcount


class MemoryService:
    def __init__(self, store: MemoryStore | None = None):
        self.store = store or MemoryStore()

    async def save(self, user_id: str, memory_type: str, key: str, value: str, *,
                   source_turn_id: str | None = None, confidence: float = 0.85,
                   importance: float = 0.6) -> Memory:
        return await asyncio.to_thread(
            self.store.save_sync, user_id, memory_type, key, value,
            source_turn_id=source_turn_id, confidence=confidence, importance=importance,
        )

    async def list(self, user_id: str, limit: int = 50) -> list[Memory]:
        return await asyncio.to_thread(self.store.list_sync, user_id, limit)

    async def relevant(self, user_id: str, query: str, limit: int = 8) -> list[Memory]:
        return await asyncio.to_thread(self.store.relevant_sync, user_id, query, limit)

    async def delete(self, user_id: str, *, memory_id: int | None = None,
                     query: str | None = None, clear_all: bool = False) -> int:
        return await asyncio.to_thread(
            self.store.delete_sync, user_id, memory_id=memory_id, query=query, clear_all=clear_all,
        )

    async def extract_explicit(self, user_id: str, text: str, source_turn_id: str | None = None) -> list[Memory]:
        """Save only clearly expressed preferences and habits.

        This deterministic fallback keeps memory useful even when a local
        model does not support tool calls. Casual statements such as "今天想
        吃火锅" are intentionally not captured.
        """
        statement = _clean_statement(text)
        if not statement or _is_memory_command(statement):
            return []
        patterns = [
            ("preference", "likes", r"^(?:我)?(?:喜欢|爱|热爱|偏好)\s*(.+)$"),
            ("preference", "dislikes", r"^(?:我)?(?:不喜欢|讨厌)\s*(.+)$"),
            ("habit", "habit", r"^(?:我)?(?:习惯|通常|一般)\s*(.+)$"),
            ("preference", "response_style", r"^以后(?:请|都|尽量)?\s*(.+)$"),
        ]
        for memory_type, key_prefix, pattern in patterns:
            match = re.match(pattern, statement, re.IGNORECASE)
            if not match:
                continue
            value = match.group(1).strip(" \t，。！？!?,.;；")[:MAX_MEMORY_VALUE_LENGTH]
            if not value:
                return []
            key = key_prefix if key_prefix == "response_style" else f"{key_prefix}:{value[:100]}"
            return [await self.save(user_id, memory_type, key, value, source_turn_id=source_turn_id)]
        return []

    async def handle_explicit_command(self, user_id: str, text: str, source_turn_id: str | None = None) -> dict[str, Any] | None:
        """Handle deterministic memory commands before asking the model."""
        statement = _clean_statement(text)
        if not statement:
            return None
        if re.search(r"(?:清除|删除|忘记)(?:我的)?所有(?:长期)?记忆$", statement):
            count = await self.delete(user_id, clear_all=True)
            return {"handled": True, "reply": f"已清除 {count} 条长期记忆。"}
        if re.search(r"(?:你记得我什么|我有哪些记忆|查看我的记忆|列出我的记忆)$", statement):
            memories = await self.list(user_id)
            return {"handled": True, "reply": _format_memories(memories)}
        delete_match = re.match(r"^(?:请)?(?:忘记|删除)(?:我)?(?:的)?(.+)$", statement)
        if delete_match:
            query = delete_match.group(1).strip(" \t，。！？!?,.;；")
            query = re.sub(r"^(?:喜欢|不喜欢|爱|热爱|偏好|习惯|通常|一般)\s*", "", query)
            count = await self.delete(user_id, query=query)
            return {"handled": True, "reply": f"已删除 {count} 条匹配的记忆。" if count else "没有找到匹配的记忆。"}
        save_match = re.match(r"^(?:请)?(?:记住|记一下)(?:我)?(?:的)?(.+)$", statement)
        if save_match:
            saved = await self.extract_explicit(user_id, save_match.group(1), source_turn_id)
            if not saved:
                value = save_match.group(1).strip(" \t，。！？!?,.;；")[:MAX_MEMORY_VALUE_LENGTH]
                if value:
                    saved = [await self.save(user_id, "fact", f"fact:{value[:100]}", value, source_turn_id=source_turn_id)]
            return {"handled": True, "reply": "好的，我会记住。" if saved else "这句话没有可保存的内容。"}
        return None

    @staticmethod
    def prompt_context(memories: list[Memory]) -> str:
        if not memories:
            return ""
        lines = ["用户长期记忆（仅在与当前问题相关时使用，不要主动暴露记忆来源）："]
        lines.extend(f"- {memory.value}" for memory in memories)
        return "\n".join(lines)


def _clean_statement(text: str) -> str:
    return str(text or "").strip().strip(" \t\r\n，。！？!?,.;；")


def _is_memory_command(statement: str) -> bool:
    return bool(re.match(r"^(?:请)?(?:记住|记一下|忘记|删除|清除|查看|列出|你记得)", statement))


def _format_memories(memories: list[Memory]) -> str:
    if not memories:
        return "我目前还没有保存长期记忆。"
    return "我目前记得：" + "；".join(memory.value for memory in memories[:20]) + "。"


memory_service = MemoryService()


async def memory_save_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    context = arguments.pop("__context", {})
    user_id = context.get("user_id", DEFAULT_USER_ID)
    memory = await memory_service.save(
        user_id,
        str(arguments.get("type") or "fact"),
        str(arguments.get("key") or "fact"),
        str(arguments.get("value") or ""),
        source_turn_id=context.get("turn_id"),
        confidence=float(arguments["confidence"] if arguments.get("confidence") is not None else 0.85),
        importance=float(arguments["importance"] if arguments.get("importance") is not None else 0.6),
    )
    return {"memory": memory.as_dict(), "message": "记忆已保存"}


async def memory_delete_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    context = arguments.pop("__context", {})
    count = await memory_service.delete(
        context.get("user_id", DEFAULT_USER_ID),
        query=arguments.get("query"),
        clear_all=bool(arguments.get("clear_all", False)),
    )
    return {"deleted": count, "message": f"已删除 {count} 条记忆"}


async def memory_list_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    context = arguments.pop("__context", {})
    memories = await memory_service.list(context.get("user_id", DEFAULT_USER_ID), limit=20)
    return {"memories": [memory.as_dict() for memory in memories], "count": len(memories)}
