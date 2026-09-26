# One-shot battlefield v3 build (Windows): pools -> selftest -> listening sheet -> test / val / train -> report.
#   pwsh scripts/build_battlefield.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = ".venv\Scripts\python.exe"
if (-not (Test-Path "data\pools.parquet")) { & $py -m ancdata.cli pools --workers 4 }
if (-not (Test-Path "data\snippets\lombard6s\meta.parquet")) { & $py -m ancdata.cli snippets --seconds 6 }
& $py -m ancdata.cli battlefield-selftest --n 100
& $py -m ancdata.cli battlefield-listen --split test --n 10
& $py -m ancdata.cli battlefield --split test --no-report
& $py -m ancdata.cli battlefield --split val --no-report
& $py -m ancdata.cli battlefield --split train --no-report
& $py -m ancdata.cli battlefield-report
