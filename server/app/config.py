"""应用配置的单一入口。

环境变量仍然是部署方式，但解析和校验集中在这里，避免入口模块散落一组互相
独立的 ``os.getenv``/``int`` 调用。Settings 先保持轻量 dataclass，后续可以在
创建 FastAPI lifespan 时把它作为依赖注入到 provider 和会话控制器。
"""

from dataclasses import dataclass
import os
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} 必须是整数，收到 {raw!r}") from error
    if value < 1:
        raise ValueError(f"{name} 必须大于 0，收到 {value}")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


def _non_negative_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} 必须是整数，收到 {raw!r}") from error
    if value < 0:
        raise ValueError(f"{name} 不能小于 0，收到 {value}")
    return value


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} 必须是数字，收到 {raw!r}") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间，收到 {value}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """启动时解析的本地服务配置。"""

    whisper_model: str
    whisper_device: str
    whisper_beam_size: int
    max_history_turns: int
    max_audio_buffer_bytes: int
    max_tool_call_rounds: int
    max_image_bytes: int
    camera_timeout_seconds: int
    network_timeout_seconds: int
    network_cache_seconds: int
    tts_engine: str
    tts_model_dir: str
    tts_speaker_id: int
    tts_speed: float
    tts_num_threads: int
    tts_provider: str
    function_calls_enabled: bool
    memory_enabled: bool
    expressions_enabled: bool

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量构造并校验配置；错误会在服务启动时明确暴露。"""
        default_tts_model_dir = str(
            Path(__file__).resolve().parents[1]
            / "models"
            / "vits-piper-zh_CN-xiao_ya-medium"
        )
        tts_engine = os.getenv("TTS_ENGINE", "auto").strip().lower()
        if tts_engine not in {"auto", "sapi", "neural"}:
            raise ValueError(f"TTS_ENGINE 必须是 auto、sapi 或 neural，收到 {tts_engine!r}")
        return cls(
            whisper_model=os.getenv("WHISPER_MODEL", "medium"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cuda"),
            whisper_beam_size=_positive_int("WHISPER_BEAM_SIZE", 5),
            max_history_turns=_positive_int("MAX_HISTORY_TURNS", 30),
            max_audio_buffer_bytes=_positive_int("MAX_AUDIO_BUFFER_BYTES", 5 * 1024 * 1024),
            max_tool_call_rounds=_positive_int("MAX_TOOL_CALL_ROUNDS", 3),
            max_image_bytes=_positive_int("MAX_IMAGE_BYTES", 4 * 1024 * 1024),
            camera_timeout_seconds=_positive_int("CAMERA_TIMEOUT_SECONDS", 30),
            network_timeout_seconds=_positive_int("NETWORK_TIMEOUT_SECONDS", 8),
            network_cache_seconds=_non_negative_int("NETWORK_CACHE_SECONDS", 300),
            tts_engine=tts_engine,
            tts_model_dir=os.getenv("TTS_MODEL_DIR", default_tts_model_dir),
            tts_speaker_id=_non_negative_int("TTS_SPEAKER_ID", 0),
            tts_speed=_bounded_float("TTS_SPEED", 1.0, 0.5, 2.0),
            tts_num_threads=_positive_int("TTS_NUM_THREADS", 2),
            tts_provider=os.getenv("TTS_PROVIDER", "cpu").strip().lower() or "cpu",
            function_calls_enabled=_bool("FUNCTION_CALLS_ENABLED", True),
            memory_enabled=_bool("MEMORY_ENABLED", True),
            expressions_enabled=_bool("EXPRESSIONS_ENABLED", True),
        )
