"""本地 TTS provider 的轻量回归测试。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import main
from app.tts import _resample_pcm, neural_model_ready


class TtsTests(unittest.TestCase):
    def test_resample_keeps_pcm16_and_changes_length(self):
        source = b"\x00\x00" * 2205
        converted = _resample_pcm(source, 22050, 24000)
        self.assertGreater(len(converted), len(source))
        self.assertEqual(len(converted) % 2, 0)

    def test_model_ready_requires_core_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(neural_model_ready(str(root)))
            (root / "model.onnx").write_bytes(b"model")
            (root / "tokens.txt").write_text("tokens", encoding="utf-8")
            (root / "lexicon.txt").write_text("lexicon", encoding="utf-8")
            self.assertTrue(neural_model_ready(str(root)))

    def test_auto_engine_falls_back_when_model_is_missing(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(main, "TTS_ENGINE", "auto"), \
                patch.object(main, "TTS_MODEL_DIR", directory), \
                patch.object(main, "synthesize_sapi", return_value=b"fallback") as sapi:
            self.assertEqual(main.synthesize_audio("测试"), b"fallback")
            sapi.assert_called_once_with("测试")


if __name__ == "__main__":
    unittest.main()
