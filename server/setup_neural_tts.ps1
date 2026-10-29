# 下载项目使用的中文 sherpa-onnx VITS 音色模型。
$ErrorActionPreference = "Stop"

$serverRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelRoot = Join-Path $serverRoot "models"
$modelName = "vits-piper-zh_CN-xiao_ya-medium"
$modelDir = Join-Path $modelRoot $modelName
$archive = Join-Path $modelRoot "$modelName.tar.bz2"
$url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/$modelName.tar.bz2"

New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null
if (-not (Test-Path (Join-Path $modelDir "zh_CN-xiao_ya-medium.onnx"))) {
    Write-Host "下载中文神经 TTS 音色模型：$modelName"
    Invoke-WebRequest -Uri $url -OutFile $archive
    tar -xjf $archive -C $modelRoot
    Remove-Item -LiteralPath $archive -Force
} else {
    Write-Host "中文神经 TTS 音色模型已经存在：$modelDir"
}

Write-Host "神经 TTS 已准备完成。TTS_ENGINE=auto 会优先使用该音色，失败时回退 Windows SAPI。"
