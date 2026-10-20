param([int]$Port = 8000, [string]$BindAddress = '127.0.0.1')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$env:COLMAP_BIN = Join-Path $repoRoot '.tools/colmap-4.2.1-cuda/bin/colmap.exe'
$python = Join-Path $repoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $env:COLMAP_BIN) -or -not (Test-Path -LiteralPath $python)) {
    throw 'Run scripts/setup_gpu.ps1 first'
}
$env:QT_PLUGIN_PATH = Join-Path $repoRoot '.tools/colmap-4.2.1-cuda/plugins'
$env:COLMAP_NUM_THREADS = '4'
$env:DATA_DIR = Join-Path $repoRoot 'data'
Set-Location -LiteralPath $repoRoot
Write-Output "Tare GPU mesh server: http://${BindAddress}:$Port"
& $python -m uvicorn app.main:app --app-dir server --host $BindAddress --port $Port
exit $LASTEXITCODE
