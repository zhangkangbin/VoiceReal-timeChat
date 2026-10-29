# 表情模块使用说明

## 已接入的行为

当前默认本地聊天链路会在同一次 LM Studio 请求中要求最终回复采用以下格式：

```json
{"text":"太好了，真为你高兴！","expression":"狂喜"}
```

`text` 是要展示和朗读的正文，`expression` 是助手回应时展示的单个表情词。
模型在工具调用中仍按原有工具协议工作，只有最终答复使用此格式。照片分析的最终
答复同样支持表情。显式记忆命令仍直接回复，不额外调用模型生成表情。

服务端只将正文写入会话历史并交给 TTS。普通文本回复保持兼容，缺少表情时返回
`NO_MATCH`；识别到破损的结构化回复时使用受控错误回复，避免读出 JSON 元数据。

## 模块边界

- `server/app/expressions/`：独立 Python 模块，词库校验、离线匹配、模型输出解析和提示词。
- `server/app/expressions/assets/expressions.v1.json`：运行时唯一词库，60 类、305 个别名。
- Android `com.example.realtimevoicechat.expression`：独立 Kotlin 本地匹配模块。
- Android Gradle 将服务端模块的 assets 目录作为额外资产来源；APK 内自带词库，运行时无需访问服务端匹配 API。
- Android 收到表情词后自己匹配，不信任服务端指定的表情 ID；只接受当前回合的表情数据。
- 助手消息气泡显示匹配后的中文名称。目前没有图片、动画或额外语义模型；稳定 ID 可供后续绑定资源。

文档目录下的 `expressions.v1.json` 是最初方案的快照。正式维护时修改运行时词库，
如需同步文档快照，应同时更新版本和示例。Android 必须重新构建才能更新内置词库。

## 匹配规则

标准英文 ID（忽略大小写）→ 中文名称 → 完整别名 → 连续“哈”规则 → 歧义词拒识 → 未知词拒识。

只去除首尾空白，不删除否定词，不按子串命中，不使用字符串相似度冒充语义理解。

- `狂喜`、`哈哈`、`哈哈哈哈` → `happy / 开心`。
- `哈哈大笑`、`爆笑` → `laugh / 大笑`。
- `难过` → `sad / 难过`，不能被相近的伤心覆盖。
- `不同意` → `reject / 拒绝`，不能因为包含同意而触发点头。
- `呵呵`、`不好意思`、`挥手`、`不开心` → `NO_MATCH`。
- 空值、非字符串、超过 64 个 Unicode 码点、明确组合多个已知表情 → `INVALID_INPUT`。
- `NO_MATCH` 不代表平静。平静是 60 个真实表情之一。

本版没有开放词汇语义兜底。遇到新词先收集并审核，可补充唯一别名；达到语义模型
训练条件后再按 `plan.md` 的路线加入编码器和量化版本。

## 接口

获取词库：`GET /api/expressions`。

独立匹配：`POST /api/expressions/match`，JSON 请求体为 `{"word":"狂喜"}`。
此接口独立于聊天开关，不启动或调用任何模型；方便开发和诊断。

正常命中的关键返回字段：

```json
{
  "status": "MATCH",
  "expression_id": "happy",
  "name": "开心",
  "source": "alias",
  "score": null,
  "catalog_version": "1.0.0"
}
```

确定性匹配不伪造概率，`score` 为 null。未知或无效输入的 `expression_id`、`name` 为 null。

WebSocket 在原有 `response.text.done` 上增加字段，不增加额外的时序事件：

```json
{
  "type": "response.text.done",
  "turn_id": "当前回合 ID",
  "text": "太好了，真为你高兴！",
  "final": true,
  "expression_word": "狂喜",
  "expression": {
    "status": "MATCH",
    "expression_id": "happy",
    "name": "开心",
    "source": "alias",
    "score": null,
    "catalog_version": "1.0.0"
  }
}
```

`POST /api/local-chat` 同样返回 `reply`、`expression_word`、`expression`。
Android 只使用原始 `expression_word` 做本地匹配。打断、取消、旧回合和缺少有效回合 ID
的事件不会更新表情。浏览器测试页展示服务端匹配结果，并提供独立的表情词测试表单。

## 配置和适用范围

`EXPRESSIONS_ENABLED=true` 默认开启，控制 LM Studio 提示词、回复解析和附加元数据。
设为 `false` 后恢复原有普通文本调用和回复字段；独立词库查询/匹配 API 仍可使用。
`/health` 的 `expression_matching` 报告开关、词库版本、条目数与匹配方式。

该聊天集成目前用于 LM Studio 本地模式及其图片分析；OpenAI Realtime 转发没有加入
表情生成协议。它不会因本功能新增云端调用或下载权重。

## 验证

服务端离线回归（从 server 目录执行）：

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_expressions test_expression_integration
```

Android 单元测试和 APK 编译（从 android 目录执行，使用已配置的 JDK/SDK）：

```powershell
.\gradlew.bat testDebugUnitTest assembleDebug
```

人工验证：

1. 服务首页“表情词匹配”输入狂喜，应显示开心；输入呵呵，应显示未匹配。
2. Android 开始聊天，观察助手回复气泡中的表情名称；朗读不能包含 expression 字段。
3. 在模型回答时打断，确认旧回合表情不会附到新回复。
4. 将 `EXPRESSIONS_ENABLED=false` 后重启服务，普通聊天仍工作且新消息不显示表情。

现有 39 个映射示例属于确定性契约验收，不是未知表达上的模型准确率成绩。
