package com.example.realtimevoicechat.expression

import org.json.JSONObject
import java.util.Locale

/** A stable resource identity; images/animations can later be bound to [id]. */
data class Expression(
    val id: String,
    val name: String,
    val description: String,
    val aliases: List<String>
)

enum class ExpressionMatchStatus { MATCH, NO_MATCH, INVALID_INPUT }

data class ExpressionMatch(
    val status: ExpressionMatchStatus,
    val expression: Expression? = null,
    val source: String? = null,
    val catalogVersion: String
)

/** Offline, deterministic matching of the upstream model's expression word only. */
class ExpressionMatcher private constructor(
    val catalogVersion: String,
    val expressions: List<Expression>,
    private val ids: Map<String, Expression>,
    private val names: Map<String, Expression>,
    private val aliases: Map<String, Expression>,
    private val ambiguousTerms: Set<String>
) {
    fun match(input: Any?): ExpressionMatch {
        if (input !is String) return result(ExpressionMatchStatus.INVALID_INPUT)
        val word = input.trim()
        if (word.isEmpty() || word.codePointCount(0, word.length) > MAX_WORD_LENGTH) {
            return result(ExpressionMatchStatus.INVALID_INPUT)
        }
        ids[word.lowercase(Locale.ROOT)]?.let { return matched(it, "id") }
        names[word]?.let { return matched(it, "name") }
        aliases[word]?.let { return matched(it, "alias") }
        if (word.matches(REPEATED_LAUGHTER)) return matched(ids.getValue("happy"), "alias")
        if (word in ambiguousTerms) return result(ExpressionMatchStatus.NO_MATCH)

        // Whole known terms only: never strip meaningful characters such as 不.
        val parts = word.split(COMPOUND_SEPARATOR).map(String::trim)
        if (parts.size > 1 && parts.all { known(it) }) {
            return result(ExpressionMatchStatus.INVALID_INPUT)
        }
        return result(ExpressionMatchStatus.NO_MATCH)
    }

    private fun known(word: String): Boolean = word.isNotEmpty() &&
        (word.lowercase(Locale.ROOT) in ids || word in names || word in aliases)

    private fun matched(expression: Expression, source: String) = ExpressionMatch(
        ExpressionMatchStatus.MATCH, expression, source, catalogVersion
    )

    private fun result(status: ExpressionMatchStatus) = ExpressionMatch(
        status = status, catalogVersion = catalogVersion
    )

    companion object {
        const val ASSET_NAME = "expressions.v1.json"
        const val MAX_WORD_LENGTH = 64
        private val REPEATED_LAUGHTER = Regex("哈{2,}")
        private val COMPOUND_SEPARATOR = Regex("[又和、,，及与或+/／]")
        private val VALID_ID = Regex("[a-z]+(?:_[a-z]+)*")

        /** Fail clearly on a conflicting catalog instead of silently overwriting an alias. */
        fun fromJson(json: String): ExpressionMatcher {
            val root = JSONObject(json)
            val version = root.getString("version").also { require(it.isNotBlank()) }
            val entries = root.getJSONArray("expressions")
            require(entries.length() == root.getInt("expression_count") && entries.length() > 0)
            val ids = linkedMapOf<String, Expression>()
            val names = linkedMapOf<String, Expression>()
            val aliases = linkedMapOf<String, Expression>()
            val allTerms = mutableSetOf<String>()
            val expressions = mutableListOf<Expression>()
            fun claim(term: String) {
                require(term.isNotBlank() && term == term.trim()) { "Invalid expression term" }
                require(allTerms.add(term.lowercase(Locale.ROOT))) { "Duplicate expression term: $term" }
            }
            for (index in 0 until entries.length()) {
                val item = entries.getJSONObject(index)
                val aliasJson = item.getJSONArray("aliases")
                val expression = Expression(
                    id = item.getString("id"),
                    name = item.getString("name"),
                    description = item.getString("description"),
                    aliases = List(aliasJson.length()) { aliasJson.getString(it) }
                )
                require(expression.id.matches(VALID_ID))
                require(expression.description.isNotBlank())
                listOf(expression.id, expression.name).plus(expression.aliases).forEach(::claim)
                ids[expression.id] = expression
                names[expression.name] = expression
                expression.aliases.forEach { aliases[it] = expression }
                expressions += expression
            }
            require("happy" in ids) { "Repeated laughter requires the happy expression" }
            val policy = root.getJSONObject("matching_policy")
            val ambiguityJson = policy.getJSONArray("ambiguous_terms")
            val ambiguous = mutableSetOf<String>()
            for (index in 0 until ambiguityJson.length()) {
                val item = ambiguityJson.getJSONObject(index)
                val term = item.getString("term")
                claim(term)
                require(item.getString("result") == "NO_MATCH")
                val candidates = item.getJSONArray("candidates")
                for (candidateIndex in 0 until candidates.length()) {
                    require(candidates.getString(candidateIndex) in ids)
                }
                ambiguous += term
            }
            return ExpressionMatcher(version, expressions.toList(), ids, names, aliases, ambiguous)
        }
    }
}

/** An expression without the current turn identity must never change the visible response. */
class ExpressionEventResolver(private val matcher: ExpressionMatcher) {
    fun resolve(
        word: Any?,
        eventTurnId: String?,
        activeTurnId: String?,
        userSpeaking: Boolean
    ): Expression? {
        if (userSpeaking || eventTurnId.isNullOrBlank() || activeTurnId.isNullOrBlank() ||
            eventTurnId != activeTurnId) return null
        return matcher.match(word).expression
    }
}
