package com.example.realtimevoicechat

import android.content.Context
import java.util.UUID

/**
 * Stable identity used to scope the conversation history on the local server.
 *
 * The storage boundary keeps persistence details out of the session and makes
 * it possible to provide a different identity source (for example, a test
 * identity or an account-backed identity) without changing audio behavior.
 */
internal interface UserIdStore {
    fun getOrCreate(): String
}

/** SharedPreferences-backed implementation used by the Android client. */
internal class SharedPreferencesUserIdStore(context: Context) : UserIdStore {
    private val preferences = context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)

    override fun getOrCreate(): String = synchronized(this) {
        preferences.getString(USER_ID_KEY, null) ?: UUID.randomUUID().toString().also { id ->
            preferences.edit().putString(USER_ID_KEY, id).apply()
        }
    }

    private companion object {
        const val PREFERENCES_NAME = "voice_assistant"
        const val USER_ID_KEY = "user_id"
    }
}
