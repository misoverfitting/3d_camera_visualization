param([switch]$SkipPython)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$toolsRoot = Join-Path $repoRoot '.tools'
$archivePath = Join-Path $toolsRoot 'colmap-4.2.1-cuda.zip'
$installRoot = Join-Path $toolsRoot 'colmap-4.2.1-cuda'
$expectedHash = 'e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881'
New-Item -ItemType Directory -Force $toolsRoot | Out-Null
if (-not (Test-Path -LiteralPath $archivePath)) {
    Invoke-WebRequest 'https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip' -OutFile $archivePath
}
if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
    throw 'COLMAP archive checksum mismatch. Installation stopped.'
}
if (-not (Test-Path -LiteralPath (Join-Path $installRoot 'bin/colmap.exe'))) {
    Expand-Archive -LiteralPath $archivePath -DestinationPath $installRoot
}
& (Join-Path $installRoot 'bin/colmap.exe') -h
if ($LASTEXITCODE -ne 0) { throw 'COLMAP could not start' }
& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if ($LASTEXITCODE -ne 0) { throw 'NVIDIA GPU/driver unavailable' }
if (-not $SkipPython) {
    $python = Join-Path $repoRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        & py -3.11 -m venv (Join-Path $repoRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11 and rerun setup' }
    }
    & $python -m pip install -r (Join-Path $repoRoot 'server/requirements-dev.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed' }
}
Write-Output 'GPU mesh runtime installed. Run scripts/start_gpu.ps1 to start the server.'
