$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$env:COLMAP_BIN = Join-Path $repoRoot '.tools/colmap-4.2.1-cuda/bin/colmap.exe'
$env:QT_PLUGIN_PATH = Join-Path $repoRoot '.tools/colmap-4.2.1-cuda/plugins'
$env:DATA_DIR = Join-Path $repoRoot 'data'
$env:COLMAP_NUM_THREADS = '4'
& (Join-Path $repoRoot '.venv/Scripts/python.exe') (Join-Path $PSScriptRoot 'smoke_native_gpu.py')
exit $LASTEXITCODE
