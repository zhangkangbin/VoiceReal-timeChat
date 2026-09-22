"""Long-term memory tests using an isolated temporary SQLite database."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import memory
from app.tools import execute_tool


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.service = memory.MemoryService(memory.MemoryStore(Path(self.tempdir.name) / "memory.db"))
        self.service_patch = patch.object(memory, "memory_service", self.service)
        self.service_patch.start()

    async def asyncTearDown(self):
        self.service_patch.stop()
        self.tempdir.cleanup()

    async def test_save_and_retrieve_preference(self):
        saved = await self.service.save("user-1", "preference", "likes:咖啡", "咖啡", source_turn_id="turn-1")
        self.assertEqual(saved.value, "咖啡")
        memories = await self.service.relevant("user-1", "我想喝咖啡")
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].key, "likes:咖啡")

    async def test_explicit_statement_is_saved_but_casual_statement_is_not(self):
        saved = await self.service.extract_explicit("user-1", "我喜欢冰美式", "turn-1")
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].value, "冰美式")
        casual = await self.service.extract_explicit("user-1", "今天我想吃火锅", "turn-2")
        self.assertEqual(casual, [])

    async def test_explicit_commands_list_delete_and_clear(self):
        await self.service.extract_explicit("user-1", "我习惯晚上工作", "turn-1")
        result = await self.service.handle_explicit_command("user-1", "你记得我什么")
        self.assertIn("晚上工作", result["reply"])
        result = await self.service.handle_explicit_command("user-1", "忘记我习惯晚上工作")
        self.assertIn("已删除 1 条", result["reply"])
        result = await self.service.handle_explicit_command("user-1", "清除所有记忆")
        self.assertIn("已清除 0 条", result["reply"])

    async def test_memory_function_tools_use_server_context(self):
        saved = await execute_tool(
            "memory_save",
            {"type": "preference", "key": "response_style", "value": "简洁"},
            context={"user_id": "user-2", "turn_id": "turn-2"},
        )
        self.assertTrue(saved["ok"])
        listed = await execute_tool("memory_list", {}, context={"user_id": "user-2"})
        self.assertEqual(listed["count"], 1)
        deleted = await execute_tool(
            "memory_delete", {"query": "简洁"}, context={"user_id": "user-2"}
        )
        self.assertEqual(deleted["deleted"], 1)


if __name__ == "__main__":
    unittest.main()
