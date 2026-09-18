$ErrorActionPreference = "Stop"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "需要安装 Python 3.11，并确保 py 启动器可用。"
}

py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "本地 Python 依赖安装完成。"
Write-Host "下一步：在 LM Studio 中启动 Local Server，并加载 google/gemma-3-4b，然后执行："
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host "uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
