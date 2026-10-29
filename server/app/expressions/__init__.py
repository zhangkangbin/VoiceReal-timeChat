"""Offline expression vocabulary matching and final assistant-reply parsing.

The JSON asset is shared with Android at build time.  Matching is deliberately
deterministic: unknown words stay unmatched instead of pretending that string
similarity provides semantic understanding.  No model or network is loaded.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any


CATALOG_PATH = Path(__file__).with_name("assets") / "expressions.v1.json"
MAX_WORD_LENGTH = 64
MAX_REPLY_LENGTH = 64 * 1024
_ID = re.compile(r"[a-z]+(?:_[a-z]+)*\Z")
_COMPOUND_SEPARATOR = re.compile(r"\s*(?:又|和|及|与|或|、|,|，|\+|/|／)\s*")
_JSON_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t]*\Z", re.IGNORECASE)
_OTHER_CODE_FENCE = re.compile(r"\A```(?P<language>[^\r\n`]+)\r?\n[\s\S]*\r?\n```[ \t]*\Z")
_ENVELOPE_START = re.compile(r'\{\s*"(?:text|expression)"\s*:')


class InvalidExpressionCatalog(ValueError):
    """A bundled vocabulary is inconsistent or ambiguous."""


class InvalidAssistantReply(ValueError):
    """A structured final reply is unsafe to treat as spoken answer text."""


@dataclass(frozen=True, slots=True)
class ParsedReply:
    text: str
    expression_word: str | None = None


class ExpressionCatalog:
    """Validated vocabulary, with exact lookup tables built once at startup."""

    def __init__(self, payload: dict[str, Any]):
        if not isinstance(payload, dict):
            raise InvalidExpressionCatalog("表情词库必须是 JSON 对象")
        version = payload.get("version")
        rows = payload.get("expressions")
        count = payload.get("expression_count")
        if not isinstance(version, str) or not version.strip() or len(version) > 64:
            raise InvalidExpressionCatalog("表情词库缺少有效版本")
        if type(count) is not int or count != 60 or not isinstance(rows, list) or len(rows) != count:
            raise InvalidExpressionCatalog("表情词库必须有且只有 60 项")
        self.version = version
        self._payload = deepcopy(payload)
        self._ids: dict[str, dict] = {}
        self._names: dict[str, dict] = {}
        self._aliases: dict[str, dict] = {}
        all_terms: set[str] = set()
        reserved = {"match", "no_match", "invalid_input"}

        def register(term: object, table: dict, entry: dict) -> None:
            if not isinstance(term, str) or not term or term != term.strip() or len(term) > MAX_WORD_LENGTH:
                raise InvalidExpressionCatalog("名称、ID 和别名必须是非空短文本")
            key = term.casefold()
            if key in all_terms or key in reserved:
                raise InvalidExpressionCatalog(f"表情词重复或使用保留状态：{term}")
            all_terms.add(key)
            table[term] = entry

        for index, entry in enumerate(self._payload["expressions"], start=1):
            if not isinstance(entry, dict) or type(entry.get("index")) is not int or entry["index"] != index:
                raise InvalidExpressionCatalog("表情序号必须从 1 到 60 连续排列")
            expression_id = entry.get("id")
            if not isinstance(expression_id, str) or _ID.fullmatch(expression_id) is None:
                raise InvalidExpressionCatalog("表情 ID 必须是小写 snake_case")
            if entry.get("type") not in {"emotion", "facial_action", "gesture", "state"}:
                raise InvalidExpressionCatalog("表情类型无效")
            for field in ("group", "description", "boundary"):
                if not isinstance(entry.get(field), str) or not entry[field].strip():
                    raise InvalidExpressionCatalog(f"表情缺少 {field}")
            aliases = entry.get("aliases")
            if not isinstance(aliases, list):
                raise InvalidExpressionCatalog("表情 aliases 必须是数组")
            register(expression_id, self._ids, entry)
            register(entry.get("name"), self._names, entry)
            for alias in aliases:
                register(alias, self._aliases, entry)

        policy = self._payload.get("matching_policy")
        if not isinstance(policy, dict):
            raise InvalidExpressionCatalog("表情词库缺少匹配规则")
        laughter = policy.get("repeat_laughter_rule")
        if not isinstance(laughter, dict) or laughter.get("pattern") != "^哈{2,}$" or laughter.get("expression_id") != "happy":
            raise InvalidExpressionCatalog("连续笑声规则必须保留为整词匹配开心")
        if "happy" not in self._ids or "neutral" not in self._ids:
            raise InvalidExpressionCatalog("表情词库缺少兼容性标签")
        ambiguity = policy.get("ambiguous_terms")
        if not isinstance(ambiguity, list):
            raise InvalidExpressionCatalog("歧义词规则必须是数组")
        self._ambiguous: set[str] = set()
        for item in ambiguity:
            if not isinstance(item, dict):
                raise InvalidExpressionCatalog("歧义词规则无效")
            term = item.get("term")
            candidates = item.get("candidates")
            if (not isinstance(term, str) or not term or term != term.strip()
                    or len(term) > MAX_WORD_LENGTH or term.casefold() in all_terms
                    or term in self._ambiguous or item.get("result") != "NO_MATCH"
                    or not isinstance(candidates, list) or len(candidates) < 2
                    or any(not isinstance(value, str) or value not in self._ids for value in candidates)):
                raise InvalidExpressionCatalog("歧义词与确定匹配冲突或候选无效")
            self._ambiguous.add(term)

    @classmethod
    def from_path(cls, path: Path | str) -> "ExpressionCatalog":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise InvalidExpressionCatalog(f"无法读取表情词库：{path}") from error
        return cls(payload)

    def as_dict(self) -> dict:
        """Return a detached payload so API consumers cannot mutate live indexes."""
        payload = deepcopy(self._payload)
        payload["matcher"] = "rules-v1"
        # The original design asset mentions the future semantic stage.  Runtime
        # capabilities are explicit so clients never confuse a plan with a model.
        payload["semantic_matching_enabled"] = False
        return payload

    def _exact(self, word: str) -> tuple[dict | None, str | None]:
        entry = self._ids.get(word.lower())
        if entry is not None:
            return entry, "id"
        entry = self._names.get(word)
        if entry is not None:
            return entry, "name"
        entry = self._aliases.get(word)
        return (entry, "alias") if entry is not None else (None, None)

    def match(self, word: object) -> dict:
        def result(status: str, normalized: str | None, entry: dict | None = None, source: str | None = None) -> dict:
            return {
                "status": status,
                "word": normalized,
                "expression_id": entry["id"] if entry else None,
                "name": entry["name"] if entry else None,
                "source": source,
                "score": None,
                "catalog_version": self.version,
            }

        if not isinstance(word, str):
            return result("INVALID_INPUT", None)
        normalized = word.strip()
        if not normalized or len(normalized) > MAX_WORD_LENGTH:
            return result("INVALID_INPUT", None)
        entry, source = self._exact(normalized)
        if entry is not None:
            return result("MATCH", normalized, entry, source)
        if re.fullmatch(r"哈{2,}", normalized):
            return result("MATCH", normalized, self._ids["happy"], "alias")
        if normalized in self._ambiguous:
            return result("NO_MATCH", normalized)
        parts = _COMPOUND_SEPARATOR.split(normalized)
        if len(parts) > 1 and all(part and self._exact(part)[0] is not None for part in parts):
            return result("INVALID_INPUT", normalized)
        return result("NO_MATCH", normalized)


@lru_cache(maxsize=1)
def _default_catalog() -> ExpressionCatalog:
    return ExpressionCatalog.from_path(CATALOG_PATH)


def get_expression_catalog() -> dict:
    """Public JSON catalog for the HTTP API; Android uses the same asset file."""
    return _default_catalog().as_dict()


def match_expression(word: object) -> dict:
    return _default_catalog().match(word)


def _contains_explicit_envelope(content: str) -> bool:
    """Recognize a wrapped/damaged protocol object, not prose naming a field.

    The protocol normally starts with text or expression.  For a valid wrapped
    object whose first key is different, JSONDecoder inspects its top-level keys.
    Merely mentioning a field, brackets, or an unrelated JSON object is allowed.
    """
    if _ENVELOPE_START.search(content):
        return True
    start = content.find("{")
    if start < 0:
        return False
    try:
        candidate, _ = json.JSONDecoder().raw_decode(content, start)
    except (ValueError, RecursionError):
        return False
    return isinstance(candidate, dict) and bool({"text", "expression"} & candidate.keys())


def parse_assistant_reply(raw: object) -> ParsedReply:
    """Extract a final JSON envelope while retaining older plain-text replies.

    Invalid expression metadata is ignored, but invalid answer text is rejected.
    Recognizable malformed envelopes raise a controlled exception rather than
    speaking JSON or treating it as a memory/chat message.
    """
    if not isinstance(raw, str) or not raw.strip() or len(raw) > MAX_REPLY_LENGTH:
        raise InvalidAssistantReply("模型回复为空、类型错误或超过长度限制")
    original = raw.strip()
    # Programming examples with an explicit non-JSON language are legacy answer
    # text, even when a code sample happens to contain these dictionary keys.
    code_fence = _OTHER_CODE_FENCE.fullmatch(original)
    if code_fence and code_fence.group("language").strip().lower() != "json":
        return ParsedReply(original)
    content = original
    fenced = _JSON_FENCE.fullmatch(original)
    if fenced:
        content = fenced.group("body").strip()
    try:
        payload = json.loads(content)
    except (ValueError, RecursionError) as error:
        if _contains_explicit_envelope(content):
            raise InvalidAssistantReply("模型回复不是完整的表情 JSON 对象") from error
        return ParsedReply(original)
    if isinstance(payload, dict):
        if not ({"text", "expression"} & payload.keys()):
            return ParsedReply(original)
    elif _contains_explicit_envelope(content):
        # A protocol object wrapped in an array is not the single-reply contract.
        raise InvalidAssistantReply("表情回复必须是单个 JSON 对象")
    else:
        return ParsedReply(original)
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_REPLY_LENGTH:
        raise InvalidAssistantReply("模型回复缺少有效 text")
    expression = payload.get("expression")
    if not isinstance(expression, str) or not expression.strip() or len(expression.strip()) > MAX_WORD_LENGTH:
        expression = None
    else:
        expression = expression.strip()
        if expression.upper() in {"NO_MATCH", "NONE", "NULL"}:
            expression = None
    return ParsedReply(text.strip(), expression)


def expression_prompt() -> str:
    """Instructions for final content only, leaving tool-call messages intact."""
    names = "、".join(entry["name"] for entry in _default_catalog()._payload["expressions"])
    return (
        "\n最终回答请只输出一个 JSON 对象，不要 Markdown 代码围栏："
        '{"text":"向用户自然回复的正文","expression":"一个表情词或 null"}。'
        "text 是要朗读的自然语言回复，不得包含表情元数据。"
        "expression 表示助手回应用户时应展示的一个主表情，不是对用户心理状态的诊断；"
        "没有合适表情时使用 JSON null。"
        f"优先从这些标准名称选择：{names}。"
        "也可使用明确的近义表情词，例如狂喜、哈哈都对应开心。"
        "不要输出多个表情、数组或复合情绪。"
        "此格式只约束最终回答；需要工具时仍按原工具调用协议执行，工具参数不受此格式约束。"
    )


__all__ = [
    "CATALOG_PATH", "ExpressionCatalog", "InvalidAssistantReply", "InvalidExpressionCatalog",
    "MAX_WORD_LENGTH", "ParsedReply", "expression_prompt", "get_expression_catalog",
    "match_expression", "parse_assistant_reply",
]
