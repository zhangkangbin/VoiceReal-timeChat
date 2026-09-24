# 安装脚本：任何一步失败都立即停止，避免用户误以为依赖已经准备好。
$ErrorActionPreference = "Stop"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "需要安装 Python 3.11，并确保 py 启动器可用。"
}

# 使用 Python 3.11 创建项目独立虚拟环境，避免污染系统 Python。
py -3.11 -m venv .venv
# 先升级 pip，再安装锁定/约束在 requirements.txt 中的运行依赖。
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "本地 Python 依赖安装完成。"
Write-Host "下一步：在 LM Studio 中启动 Local Server，并加载 google/gemma-3-12b（Instruct Q4_K_M），然后执行："
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host "uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
