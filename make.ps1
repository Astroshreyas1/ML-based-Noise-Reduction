<# Windows twin of the Makefile.  Usage:  pwsh make.ps1 <target>   (targets: same as `make help`) #>
param([Parameter(Position = 0)][string]$Target = "help", [string]$Cfg = "configs/train.yaml", [int]$NPerPreset = 60, [int]$TrainN = 10000, [int]$ValN = 500)
$ErrorActionPreference = "Stop"
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
$root = $env:ANC_DATA_ROOT ?? "data"
switch ($Target) {
    "help"         { Get-Content $PSCommandPath | Select-String '^\s+"([a-z-]+)"\s+\{' | ForEach-Object { $_.Matches[0].Groups[1].Value } }
    "setup"        { & $py -m pip install -r requirements.txt; & $py -m pip install -e . }
    "download"     { & pwsh scripts/download_sources.ps1 }
    "download-all" { & pwsh scripts/download_sources.ps1 -All }
    "rir"          { & $py -m ancdata.cli rir --n-per-preset $NPerPreset }
    "registry"     { & $py -m ancdata.cli registry }
    "smoke"        { & $py -m ancdata.cli selftest --smoke }
    "test"         { & $py -m pytest -q }
    "selftest"     { & $py -m ancdata.cli selftest --config $Cfg }
    "evals"        {
        & $py -m ancdata.cli materialize --config configs/eval_standard.yaml --split test --n 2000
        & $py -m ancdata.cli materialize --config configs/eval_generalization.yaml --split test --n 1000
        try { & $py -m ancdata.cli materialize --config configs/eval_lombard.yaml --split test --n 500 } catch { Write-Host "eval_lombard skipped: $_" }
    }
    "dump"         {
        & $py -m ancdata.cli materialize --config $Cfg --split train --n $TrainN --fmt flac --out (Join-Path $root "train_dump")
        & $py -m ancdata.cli materialize --config $Cfg --split val   --n $ValN   --fmt flac --out (Join-Path $root "val_dump")
    }
    "datasets"     { foreach ($t in "registry", "selftest", "evals", "dump", "baseline", "plots") { & $PSCommandPath $t -Cfg $Cfg -TrainN $TrainN -ValN $ValN } }
    "baseline"     { Get-ChildItem (Join-Path $root "eval") -Directory | Where-Object { Test-Path (Join-Path $_.FullName "meta.jsonl") } | ForEach-Object { & $py -m ancdata.cli evaluate $_.FullName } }
    "plots"        { & $py -m ancdata.cli plots --config $Cfg }
    "docker"       { docker build -t ancdata . }
    default        { Write-Error "unknown target $Target" }
}
