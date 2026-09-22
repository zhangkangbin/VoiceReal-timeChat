# Function Call 开发指南

本文说明如何在 VoiceReal-timeChat 项目中新增业务函数，并让本地 LM Studio 和 OpenAI Realtime 共用同一套工具实现。

## 1. 当前架构

Function Call 的核心代码位于：

```text
server/app/tools.py
```

一次完整调用的流程如下：

```text
用户语音
  → Whisper 转写
  → 模型选择工具并生成参数
  → 服务端校验工具名称和参数
  → 服务端执行工具
  → 将结构化结果返回模型
  → 模型生成最终文本
  → TTS 合成并发送到 Android
```

工具始终由 FastAPI 服务端执行。Android 端不保存业务密钥，也不直接执行普通业务函数。

## 2. 新增业务函数的三个步骤

新增函数时，需要在 `server/app/tools.py` 中完成以下三项工作：

1. 编写异步处理函数。
2. 在 `TOOL_DEFINITIONS` 中添加工具定义和参数 JSON Schema。
3. 在 `TOOL_HANDLERS` 中注册工具名称和处理函数。

完成注册后，本地 LM Studio 和 OpenAI Realtime 会自动获取同一份工具列表，不需要分别维护两套定义。

## 3. 编写工具处理函数

所有处理函数都应使用以下形式：

```python
async def tool_name(arguments: dict[str, Any]) -> dict[str, Any]:
    ...
```

示例：新增一个订单查询函数。

```python
async def get_order_status(arguments: dict[str, Any]) -> dict[str, Any]:
    order_number = str(arguments.get("order_number") or "").strip()
    if not order_number:
        raise ToolExecutionError("订单号不能为空")

    # 在这里调用真实数据库或业务接口。
    return {
        "order_number": order_number,
        "status": "shipped",
        "estimated_delivery": "2026-09-25",
    }
```

处理函数要求：

- 必须是 `async def`。
- 输入必须是字典。
- 返回值建议是字典，便于模型理解。
- 可预期的业务错误使用 `ToolExecutionError`。
- 不要直接返回密码、Token、数据库连接信息或内部异常堆栈。
- 不要相信模型传入的参数，必须再次校验。

## 4. 添加工具定义

在 `TOOL_DEFINITIONS` 数组中添加工具的名称、描述和参数结构：

```python
{
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "根据用户提供的订单号查询订单状态和预计送达日期。",
        "parameters": {
            "type": "object",
            "properties": {
                "order_number": {
                    "type": "string",
                    "description": "用户可见的订单号，例如 ORD-20260921-001。",
                }
            },
            "required": ["order_number"],
            "additionalProperties": False,
        },
    },
}
```

工具描述应明确说明：

- 工具能做什么。
- 什么情况下应该调用。
- 每个参数的含义和格式。
- 哪些参数是必填项。

工具名称只能表达一个明确动作。建议使用 `snake_case`，例如：

```text
get_order_status
search_products
create_reminder
get_weather
```

## 5. 注册处理函数

将工具名称和处理函数加入 `TOOL_HANDLERS`：

```python
TOOL_HANDLERS: dict[str, ToolHandler] = {
    "get_current_time": get_current_time,
    "get_order_status": get_order_status,
}
```

工具名称必须与 `TOOL_DEFINITIONS` 中的 `function.name` 完全一致。未注册的名称会被服务端拒绝，并返回“未知工具”。

## 6. 完整示例

下面展示新增 `get_order_status` 所需的核心代码：

```python
async def get_order_status(arguments: dict[str, Any]) -> dict[str, Any]:
    order_number = str(arguments.get("order_number") or "").strip()
    if not order_number:
        raise ToolExecutionError("订单号不能为空")

    return {
        "order_number": order_number,
        "status": "shipped",
        "estimated_delivery": "2026-09-25",
    }


TOOL_DEFINITIONS = [
    # 已有工具……
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "根据订单号查询订单状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {
                        "type": "string",
                        "description": "用户可见的订单号。",
                    }
                },
                "required": ["order_number"],
                "additionalProperties": False,
            },
        },
    },
]


TOOL_HANDLERS = {
    "get_current_time": get_current_time,
    "get_order_status": get_order_status,
}
```

注册完成后，用户可以询问：

```text
帮我查一下订单 ORD-20260921-001 到哪里了。
```

模型会生成工具调用，服务端执行 `get_order_status`，然后模型根据返回结果组织最终语音回答。

## 7. 调用外部 API

如果工具需要访问 HTTP API，建议使用异步客户端：

```python
import httpx


async def get_weather(arguments: dict[str, Any]) -> dict[str, Any]:
    city = str(arguments.get("city") or "").strip()
    if not city:
        raise ToolExecutionError("城市不能为空")

    async with httpx.AsyncClient(timeout=8, trust_env=False) as client:
        response = await client.get(
            "https://example.com/weather",
            params={"city": city},
        )
        response.raise_for_status()
        data = response.json()

    return {
        "city": city,
        "condition": data.get("condition"),
        "temperature": data.get("temperature"),
    }
```

API Key 应通过环境变量读取，不能写入源码、工具描述或发送到 Android。

## 8. 超时和错误处理

所有工具统一通过 `execute_tool()` 执行，默认超时时间是 10 秒。

标准返回格式：

```json
{
  "ok": true,
  "result": "..."
}
```

失败时：

```json
{
  "ok": false,
  "error": "错误说明"
}
```

处理函数应使用以下原则：

- 参数错误：抛出 `ToolExecutionError`。
- 远程接口超时：返回可理解的业务错误。
- 内部异常：不要把异常堆栈发送给模型或用户。
- 有副作用的操作：增加确认、幂等请求 ID 和权限检查。

## 9. 有副作用的工具

查询类工具通常可以直接执行，例如：

```text
get_current_time
get_order_status
search_products
```

以下工具会修改外部状态，应增加用户确认：

```text
create_order
send_message
delete_file
cancel_booking
update_password
```

建议流程：

```text
模型识别操作意图
  → 服务端生成待确认操作
  → 用户明确确认
  → 服务端执行
  → 返回执行结果
```

不要仅凭模型的一次调用直接执行删除、支付、发送消息等高风险操作。

## 10. 添加测试

Function Call 测试位于：

```text
server/test_function_call.py
```

每个新工具至少测试：

- 正常参数可以成功执行。
- 缺少必填参数时返回错误。
- 非法参数不会执行真实业务操作。
- 未注册的工具会被拒绝。
- 外部接口超时可以正确恢复。
- 工具结果可以继续交给模型生成最终回答。

测试示例：

```python
async def test_get_order_status(self):
    result = await execute_tool(
        "get_order_status",
        {"order_number": "ORD-20260921-001"},
    )
    self.assertTrue(result["ok"])
    self.assertEqual(result["order_number"], "ORD-20260921-001")
```

运行测试：

```powershell
cd server
.\.venv\Scripts\python.exe -m unittest -v test_full_duplex.py test_function_call.py
```

## 11. 配置项

Function Call 默认开启，可通过环境变量控制：

```text
FUNCTION_CALLS_ENABLED=true
MAX_TOOL_CALL_ROUNDS=3
```

`MAX_TOOL_CALL_ROUNDS` 用于防止模型在同一轮对话中无限调用工具。

如果 LM Studio 当前加载的模型不支持原生 tool calling，服务端会回退到普通对话。要使用 Function Call，需要加载支持工具调用的 Instruct 模型。

## 12. 新工具检查清单

提交新工具前确认：

- [ ] 处理函数使用 `async def`。
- [ ] 工具已加入 `TOOL_DEFINITIONS`。
- [ ] 工具已加入 `TOOL_HANDLERS`。
- [ ] 工具名称在定义和注册表中完全一致。
- [ ] 所有模型参数都经过服务端校验。
- [ ] 没有在代码或返回值中暴露密钥。
- [ ] 外部请求设置了超时。
- [ ] 高风险操作需要明确确认。
- [ ] 已添加正常、错误和超时测试。
- [ ] 已完成语音端到端测试。

