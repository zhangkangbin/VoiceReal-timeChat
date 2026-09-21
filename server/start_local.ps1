$ErrorActionPreference = "Stop"
$lms = "D:\Program Files\LM Studio\resources\app\.webpack\lms.exe"
if (-not (Test-Path -LiteralPath $lms)) { throw "找不到 LM Studio lms.exe：$lms" }

& $lms server start
# Keep one fast local model resident on the RTX 5060 Ti.  A single
# generation lane is the right setting for one voice conversation: it avoids
# duplicate model instances and gives the active request the full GPU path.
& $lms load "google/gemma-3-12b" --gpu max --context-length 4096 --parallel 1 --identifier "google/gemma-3-12b" --yes
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
