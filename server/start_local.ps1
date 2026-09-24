# 启动脚本：LM Studio、模型和 FastAPI 任一环节失败都应让脚本返回错误。
$ErrorActionPreference = "Stop"
$lms = "D:\Program Files\LM Studio\resources\app\.webpack\lms.exe"
# 这里使用固定的 LM Studio 安装路径；找不到时给出可读错误，而不是静默跳过。
if (-not (Test-Path -LiteralPath $lms)) { throw "找不到 LM Studio lms.exe：$lms" }

# 启动兼容 OpenAI API 的本地服务，然后让指定模型常驻 GPU。
& $lms server start
# Keep one fast local model resident on the RTX 5060 Ti.  A single
# generation lane is the right setting for one voice conversation: it avoids
# duplicate model instances and gives the active request the full GPU path.
& $lms load "google/gemma-3-12b" --gpu max --context-length 4096 --parallel 1 --identifier "google/gemma-3-12b" --yes
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
