"""本地语音合成适配器。

默认链路仍然可以使用 Windows SAPI；配置为 ``neural`` 时，使用
``sherpa-onnx`` 加载本地中文 VITS 模型。两种引擎都返回 24 kHz、单声道、
16-bit little-endian PCM，因此 WebSocket 和 Android 播放端不需要知道具体
的 TTS 实现。
"""

from __future__ import annotations

import audioop
from array import array
from pathlib import Path
import threading
from typing import Any


OUTPUT_SAMPLE_RATE = 24000


def _float_samples_to_pcm(samples: Any) -> bytes:
    """把 sherpa-onnx 的浮点采样安全地转换为 PCM16。"""
    output = array("h")
    for sample in samples:
        value = max(-1.0, min(1.0, float(sample)))
        output.append(int(value * 32767.0))
    return output.tobytes()


def _resample_pcm(pcm: bytes, sample_rate: int, target_rate: int = OUTPUT_SAMPLE_RATE) -> bytes:
    """按目标协议重采样 PCM16；采样率相同则避免一次额外拷贝。"""
    if not pcm or sample_rate == target_rate:
        return pcm
    converted, _ = audioop.ratecv(pcm, 2, 1, sample_rate, target_rate, None)
    return converted


class NeuralTts:
    """惰性加载且串行访问的 sherpa-onnx 中文 VITS 合成器。"""

    def __init__(self, model_dir: str, *, speaker_id: int = 0, speed: float = 1.0,
                 num_threads: int = 2, provider: str = "cpu") -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.speaker_id = max(0, int(speaker_id))
        self.speed = max(0.5, min(2.0, float(speed)))
        self.num_threads = max(1, int(num_threads))
        self.provider = provider.strip() or "cpu"
        self._tts = None
        self._load_lock = threading.Lock()
        self._synthesis_lock = threading.Lock()

    def _find(self, *names: str) -> Path:
        for name in names:
            candidate = self.model_dir / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"神经 TTS 模型目录缺少文件：{', '.join(names)}；当前目录为 {self.model_dir}"
        )

    def _load(self):
        if self._tts is not None:
            return self._tts
        with self._load_lock:
            if self._tts is not None:
                return self._tts
            try:
                import sherpa_onnx
            except ImportError as error:
                raise RuntimeError(
                    "未安装 sherpa-onnx，请执行 server\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
                ) from error

            model = self._find("zh_CN-xiao_ya-medium.onnx", "model.onnx", "vits-zh-hf-fanchen-wnj.onnx")
            tokens = self._find("tokens.txt")
            lexicon = self._find("lexicon.txt")
            rule_fsts = [
                str(self.model_dir / name)
                for name in ("phone.fst", "date.fst", "number.fst")
                if (self.model_dir / name).is_file()
            ]
            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=str(model),
                        lexicon=str(lexicon),
                        tokens=str(tokens),
                    ),
                    num_threads=self.num_threads,
                    provider=self.provider,
                ),
                rule_fsts=",".join(rule_fsts),
            )
            if not config.validate():
                raise RuntimeError(f"神经 TTS 配置无效：{self.model_dir}")
            self._tts = sherpa_onnx.OfflineTts(config)
            return self._tts

    def synthesize(self, text: str) -> bytes:
        """同步合成一段文本，并返回协议约定的 PCM16。"""
        if not text or not text.strip():
            return b""
        # OfflineTts 的底层运行时不保证并发 generate 安全；串行化只影响 TTS，
        # 不会阻塞 asyncio 事件循环，因为调用者会把它放到线程中执行。
        with self._synthesis_lock:
            tts = self._load()
            audio = tts.generate(text=text.strip(), sid=self.speaker_id, speed=self.speed)
            return _resample_pcm(_float_samples_to_pcm(audio.samples), int(audio.sample_rate))

    @property
    def sample_rate(self) -> int | None:
        """返回已加载模型的采样率；未加载时返回 ``None``。"""
        return int(self._tts.sample_rate) if self._tts is not None else None


def neural_model_ready(model_dir: str) -> bool:
    """只检查模型文件是否齐全，不触发 ONNX Runtime 加载。"""
    root = Path(model_dir).expanduser()
    model_exists = any((root / name).is_file() for name in (
        "zh_CN-xiao_ya-medium.onnx", "model.onnx", "vits-zh-hf-fanchen-wnj.onnx"
    ))
    return model_exists and (root / "tokens.txt").is_file() and (root / "lexicon.txt").is_file()
