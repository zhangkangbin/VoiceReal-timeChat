"""应用配置的单一入口。

环境变量仍然是部署方式，但解析和校验集中在这里，避免入口模块散落一组互相
独立的 ``os.getenv``/``int`` 调用。Settings 先保持轻量 dataclass，后续可以在
创建 FastAPI lifespan 时把它作为依赖注入到 provider 和会话控制器。
"""

from dataclasses import dataclass
import os


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


@dataclass(frozen=True, slots=True)
class Settings:
    """启动时解析的本地服务配置。"""

    whisper_model: str
    whisper_device: str
    whisper_beam_size: int
    max_history_turns: int
    max_audio_buffer_bytes: int
    max_tool_call_rounds: int
    function_calls_enabled: bool
    memory_enabled: bool

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量构造并校验配置；错误会在服务启动时明确暴露。"""
        return cls(
            whisper_model=os.getenv("WHISPER_MODEL", "medium"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cuda"),
            whisper_beam_size=_positive_int("WHISPER_BEAM_SIZE", 5),
            max_history_turns=_positive_int("MAX_HISTORY_TURNS", 30),
            max_audio_buffer_bytes=_positive_int("MAX_AUDIO_BUFFER_BYTES", 5 * 1024 * 1024),
            max_tool_call_rounds=_positive_int("MAX_TOOL_CALL_ROUNDS", 3),
            function_calls_enabled=_bool("FUNCTION_CALLS_ENABLED", True),
            memory_enabled=_bool("MEMORY_ENABLED", True),
        )

