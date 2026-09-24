// 插件解析仓库；Gradle 会从这里取得 Android、Kotlin 和 Compose 插件。
pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
// 依赖统一从官方仓库解析，禁止子项目偷偷声明额外仓库以保证构建可复现。
dependencyResolutionManagement { repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS); repositories { google(); mavenCentral() } }
// 工程名及唯一的 Android 应用模块。
rootProject.name = "RealtimeVoiceChat"
include(":app")
