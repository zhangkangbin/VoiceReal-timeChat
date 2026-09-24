// Android 应用模块的插件：Android 构建、Kotlin 编译以及 Jetpack Compose 编译器。
plugins { id("com.android.application"); id("org.jetbrains.kotlin.android"); id("org.jetbrains.kotlin.plugin.compose") }

// 这些参数决定 APK 的包名、最低系统版本和 Kotlin/Java 字节码版本。
android { namespace = "com.example.realtimevoicechat"; compileSdk = 35
    defaultConfig { applicationId = "com.example.realtimevoicechat"; minSdk = 26; targetSdk = 35; versionCode = 1; versionName = "0.1.0" }
    // Compose 需要在 Android 构建阶段启用对应的编译扩展。
    buildFeatures { compose = true }
    // 项目使用 Java 17 的 API/字节码目标，需与本地 Android Studio JDK 保持一致。
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}

// UI、协程、网络和本地 ONNX Runtime 依赖；AAR 直接放在 app/libs 目录中，
// 因而不需要再从远程仓库下载同一个推理运行时。
dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.10.0")
    implementation("androidx.compose.ui:ui:1.7.6")
    implementation("androidx.compose.material3:material3:1.3.1")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation(files("libs/onnxruntime-android-1.19.2.aar"))
}
