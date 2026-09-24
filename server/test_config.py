"""启动配置解析的单元测试。"""

import os
import unittest
from unittest.mock import patch

from app.config import Settings


class SettingsTests(unittest.TestCase):
    def test_defaults_are_stable(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.whisper_model, "medium")
        self.assertEqual(settings.max_history_turns, 30)
        self.assertEqual(settings.max_audio_buffer_bytes, 5 * 1024 * 1024)
        self.assertTrue(settings.function_calls_enabled)

    def test_invalid_numeric_value_fails_with_variable_name(self):
        with patch.dict(os.environ, {"MAX_HISTORY_TURNS": "unbounded"}, clear=True):
            with self.assertRaisesRegex(ValueError, "MAX_HISTORY_TURNS"):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()

