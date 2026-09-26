# Demo day: start the ANC-Net demo server and open the page.
#   pwsh demo/run_demo.ps1            (uses model/runs/v3a/ckpt_best.pt, CPU inference, test-split noise)
#   pwsh demo/run_demo.ps1 -Cuda      (GPU inference; only when no training run is using it)
param([switch]$Cuda, [string]$Ckpt = "model/runs/v3a/ckpt_best.pt", [int]$Port = 8765)
Set-Location (Join-Path $PSScriptRoot "..")
$args = @("demo/server.py", "--ckpt", $Ckpt, "--port", $Port)
if ($Cuda) { $args += "--cuda" }
Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList $args -WindowStyle Minimized
Start-Sleep 8
Start-Process "http://localhost:$Port"
