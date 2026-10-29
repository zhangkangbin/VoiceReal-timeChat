"""No-network vocabulary and structured-expression reply regression tests."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from app.expressions import (
    CATALOG_PATH,
    ExpressionCatalog,
    InvalidAssistantReply,
    InvalidExpressionCatalog,
    expression_prompt,
    get_expression_catalog,
    match_expression,
    parse_assistant_reply,
)


DOCS = Path(__file__).resolve().parents[1] / "docs" / "expression-mapping"


class ExpressionCatalogTests(unittest.TestCase):
    def test_catalog_matches_reviewed_design_and_has_unique_exact_entries(self):
        payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload, json.loads((DOCS / "expressions.v1.json").read_text(encoding="utf-8")))
        rows = payload["expressions"]
        self.assertEqual(len(rows), 60)
        self.assertEqual(sum(len(row["aliases"]) for row in rows), 305)
        for row in rows:
            for value, source in [(row["id"], "id"), (row["name"], "name"), *[(alias, "alias") for alias in row["aliases"]]]:
                with self.subTest(value=value):
                    result = match_expression(value)
                    self.assertEqual(result["status"], "MATCH")
                    self.assertEqual(result["expression_id"], row["id"])
                    self.assertEqual(result["source"], source)
                    self.assertIsNone(result["score"])

    def test_reviewed_mapping_examples(self):
        cases = json.loads((DOCS / "mapping-examples.v1.json").read_text(encoding="utf-8"))["cases"]
        self.assertEqual(len(cases), 39)
        for case in cases:
            with self.subTest(value=case["input"]):
                result = match_expression(case["input"])
                self.assertEqual(result["status"], case["status"])
                self.assertEqual(result["expression_id"], case["expression_id"])

    def test_formatting_and_size_boundaries(self):
        self.assertEqual(match_expression(" HAPPY ")["expression_id"], "happy")
        self.assertEqual(match_expression("哈" * 64)["expression_id"], "happy")
        for value in ["哈" * 65, "  ", None, 1, True, {"word": "开心"}]:
            with self.subTest(value=value):
                result = match_expression(value)
                self.assertEqual(result["status"], "INVALID_INPUT")
                self.assertIsNone(result["expression_id"])
                self.assertIsNone(result["word"])
        # Python len counts code points rather than UTF-16 code units.
        self.assertEqual(match_expression("😀" * 64)["status"], "NO_MATCH")
        self.assertEqual(match_expression("😀" * 65)["status"], "INVALID_INPUT")

    def test_never_uses_substrings_or_strips_negation(self):
        self.assertEqual(match_expression("不同意")["expression_id"], "reject")
        for value in ["不开心", "不是开心", "不伤心", "开心吗", "非常开心的用户"]:
            with self.subTest(value=value):
                self.assertEqual(match_expression(value)["status"], "NO_MATCH")
        for value in ["开心又难过", "开心、难过", "happy / sad", "微笑和眨眼"]:
            with self.subTest(value=value):
                self.assertEqual(match_expression(value)["status"], "INVALID_INPUT")

    def test_no_match_is_distinct_from_real_neutral_expression(self):
        result = match_expression("天气")
        self.assertEqual(result["status"], "NO_MATCH")
        self.assertIsNone(result["expression_id"])
        self.assertEqual(match_expression("平静")["expression_id"], "neutral")

    def test_api_payload_cannot_modify_cached_matcher(self):
        payload = get_expression_catalog()
        payload["expressions"][0]["name"] = "改名"
        payload["expressions"][0]["aliases"].append("新词")
        self.assertEqual(match_expression("开心")["name"], "开心")
        self.assertEqual(match_expression("新词")["status"], "NO_MATCH")
        self.assertEqual(get_expression_catalog()["matcher"], "rules-v1")
        self.assertFalse(get_expression_catalog()["semantic_matching_enabled"])

    def test_invalid_catalogs_are_rejected_before_matching(self):
        base = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        mutations = [
            lambda p: p.update(expression_count=59),
            lambda p: p.update(expression_count=True),
            lambda p: p.update(version=""),
            lambda p: p["expressions"].pop(),
            lambda p: p["expressions"][1].update(id="happy"),
            lambda p: p["expressions"][1].update(name="开心"),
            lambda p: p["expressions"][1]["aliases"].append("狂喜"),
            lambda p: p["expressions"][1]["aliases"].append("HAPPY"),
            lambda p: p["expressions"][1]["aliases"].append("NO_MATCH"),
            lambda p: p["expressions"][1].update(index=1),
            lambda p: p["expressions"][1].update(type="unknown"),
            lambda p: p["expressions"][1].update(aliases="浅笑"),
            lambda p: p["expressions"][1].update(description=""),
            lambda p: p["matching_policy"]["repeat_laughter_rule"].update(expression_id="laugh"),
            lambda p: p["matching_policy"]["ambiguous_terms"][0].update(term="开心"),
            lambda p: p["matching_policy"]["ambiguous_terms"][0].update(candidates=["unknown", "happy"]),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutations.index(mutation)):
                payload = deepcopy(base)
                mutation(payload)
                with self.assertRaises(InvalidExpressionCatalog):
                    ExpressionCatalog(payload)
        with self.assertRaises(InvalidExpressionCatalog):
            ExpressionCatalog([])
        with self.assertRaises(InvalidExpressionCatalog):
            ExpressionCatalog.from_path(DOCS / "missing-catalog.json")


class AssistantReplyParsingTests(unittest.TestCase):
    def test_final_json_extracts_text_and_expression_without_metadata(self):
        result = parse_assistant_reply('{"text":"真为你高兴！","expression":"狂喜"}')
        self.assertEqual(result.text, "真为你高兴！")
        self.assertEqual(result.expression_word, "狂喜")
        self.assertEqual(match_expression(result.expression_word)["expression_id"], "happy")

    def test_fenced_json_and_legacy_plain_text(self):
        for fence in ["json", "JSON", ""]:
            reply = parse_assistant_reply('```' + fence + '\n{"text":"你好！","expression":"打招呼"}\n```')
            self.assertEqual(reply.text, "你好！")
            self.assertEqual(reply.expression_word, "打招呼")
        for plain in ["  你好，今天过得怎么样？  ", "答案是 42。", "42"]:
            result = parse_assistant_reply(plain)
            self.assertEqual(result.text, plain.strip())
            self.assertIsNone(result.expression_word)

    def test_legacy_prose_json_and_code_examples_are_not_expression_envelopes(self):
        legacy = [
            '[提示] 请先检查网络。',
            '字段 "text": 表示回复正文。',
            '{"name":"小明"}',
            '{"schema":{"text":"示例字段"}}',
            '{}', '[]', '[1, 2, 3]', '[{"name":"小明"}]',
            'true', 'false', 'null',
            '```json\n{"name":"小明"}\n```',
            '```json\n[1, 2, 3]\n```',
            '```\n普通代码示例\n```',
            '```python\nprint({"text": "示例字段"})\n```',
            '```json\nnot json\n```',
        ]
        for value in legacy:
            with self.subTest(value=value):
                result = parse_assistant_reply(value)
                self.assertEqual(result.text, value)
                self.assertIsNone(result.expression_word)

    def test_invalid_expression_metadata_does_not_destroy_valid_text(self):
        for expression in [None, [], ["开心"], {"name": "开心"}, 1, False, "", "  ", "哈" * 65, "NO_MATCH", "null"]:
            with self.subTest(expression=expression):
                reply = parse_assistant_reply(json.dumps({"text": "正常回复", "expression": expression}, ensure_ascii=False))
                self.assertEqual(reply.text, "正常回复")
                self.assertIsNone(reply.expression_word)
        self.assertIsNone(parse_assistant_reply('{"text":"只有正文"}').expression_word)

    def test_bad_envelopes_cannot_reach_tts_as_literal_metadata(self):
        invalid = [None, 3, "", " ", "x" * (64 * 1024 + 1), '[{"text":"你好"}]',
                   '{"text":null}', '{"text":[]}', '{"text":42}', '{"text":""}', '{"text":"  "}',
                   '{"text":"你好","expression":"开心"', '```json\n{"text":"你好"}',
                   '先看结果：{"text":"你好","expression":"开心"}',
                   '<think>分析完成</think>{"text":"你好","expression":"开心"}',
                   '结果：{"extra":true,"text":"你好","expression":"开心"}',
                   '{"expression":"开心"}', '{"text":"你好"}\n额外尾部']
        for value in invalid:
            with self.subTest(value=str(value)[:80]):
                with self.assertRaises(InvalidAssistantReply):
                    parse_assistant_reply(value)

    def test_prompt_covers_all_choices_and_preserves_tool_protocol(self):
        prompt = expression_prompt()
        for expression in get_expression_catalog()["expressions"]:
            self.assertIn(expression["name"], prompt)
        self.assertIn("助手回应用户时", prompt)
        self.assertIn("只约束最终回答", prompt)
        self.assertIn("工具参数不受此格式约束", prompt)


if __name__ == "__main__":
    unittest.main()
