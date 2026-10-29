package com.example.realtimevoicechat.expression

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class ExpressionMatcherTest {
    private fun catalogText() = File(requireNotNull(System.getProperty("expressionCatalogPath")))
        .readText(Charsets.UTF_8)

    private fun matcher() = ExpressionMatcher.fromJson(catalogText())

    @Test fun allCatalogNamesIdsAndAliasesMatchLocally() {
        val matcher = matcher()
        assertEquals(60, matcher.expressions.size)
        assertEquals(305, matcher.expressions.sumOf { it.aliases.size })
        matcher.expressions.forEach { expression ->
            (listOf(expression.id, expression.name) + expression.aliases).forEach { term ->
                val result = matcher.match(term)
                assertEquals(term, ExpressionMatchStatus.MATCH, result.status)
                assertEquals(term, expression.id, result.expression?.id)
            }
        }
    }

    @Test fun sharedMappingExamplesPass() {
        val matcher = matcher()
        val examples = JSONObject(File(requireNotNull(System.getProperty("expressionExamplesPath")))
            .readText(Charsets.UTF_8)).getJSONArray("cases")
        assertEquals(39, examples.length())
        for (index in 0 until examples.length()) {
            val example = examples.getJSONObject(index)
            val result = matcher.match(example.opt("input"))
            assertEquals(example.toString(), example.getString("status"), result.status.name)
            val expectedId = if (example.isNull("expression_id")) null else example.getString("expression_id")
            assertEquals(example.toString(), expectedId, result.expression?.id)
        }
    }

    @Test fun knownWordsDoNotUseSubstringMatching() {
        val matcher = matcher()
        assertEquals("happy", matcher.match("狂喜").expression?.id)
        assertEquals("happy", matcher.match("哈哈哈哈").expression?.id)
        assertEquals("laugh", matcher.match("哈哈大笑").expression?.id)
        assertEquals("reject", matcher.match("不同意").expression?.id)
        assertNull(matcher.match("我不开心").expression)
        assertNull(matcher.match("哈哈，不开心").expression)
        assertEquals("sad", matcher.match("难过").expression?.id)
        assertEquals("heartbroken", matcher.match("伤心").expression?.id)
    }

    @Test fun unknownAndInvalidInputsStayDistinctFromNeutral() {
        val matcher = matcher()
        assertEquals("neutral", matcher.match("平静").expression?.id)
        listOf("天气", "头痛", "呵呵", "不好意思", "挥手", "笑哭", "难受").forEach {
            assertEquals(it, ExpressionMatchStatus.NO_MATCH, matcher.match(it).status)
        }
        listOf(null, JSONObject.NULL, 3, emptyList<String>(), "", "  ", "哈".repeat(65),
            "开心和难过", "happy/sad").forEach {
            assertEquals(it.toString(), ExpressionMatchStatus.INVALID_INPUT, matcher.match(it).status)
        }
        assertEquals("happy", matcher.match("  HAPPY  ").expression?.id)
    }

    @Test fun staleOrUnboundEventsCannotAttachAnExpression() {
        val resolver = ExpressionEventResolver(matcher())
        assertEquals("happy", resolver.resolve("狂喜", "turn-2", "turn-2", false)?.id)
        assertNull(resolver.resolve("狂喜", "turn-1", "turn-2", false))
        assertNull(resolver.resolve("狂喜", "", "turn-2", false))
        assertNull(resolver.resolve("狂喜", null, "turn-2", false))
        assertNull(resolver.resolve("狂喜", "turn-2", null, false))
        assertNull(resolver.resolve("狂喜", "turn-2", "turn-2", true))
        assertNull(resolver.resolve("呵呵", "turn-2", "turn-2", false))
        assertNull(resolver.resolve(JSONObject.NULL, "turn-2", "turn-2", false))
    }

    @Test fun conflictingCatalogTermsFailInsteadOfOverwriting() {
        val root = JSONObject(catalogText())
        root.getJSONArray("expressions").getJSONObject(1).getJSONArray("aliases").put("狂喜")
        assertThrows(IllegalArgumentException::class.java) { ExpressionMatcher.fromJson(root.toString()) }
    }

    @Test fun catalogCountAndAmbiguityReferencesAreValidated() {
        val root = JSONObject(catalogText())
        root.put("expression_count", 61)
        assertThrows(IllegalArgumentException::class.java) { ExpressionMatcher.fromJson(root.toString()) }
        root.put("expression_count", 60)
        root.getJSONObject("matching_policy").getJSONArray("ambiguous_terms")
            .getJSONObject(0).getJSONArray("candidates").put("nonexistent")
        assertThrows(IllegalArgumentException::class.java) { ExpressionMatcher.fromJson(root.toString()) }
    }
}
