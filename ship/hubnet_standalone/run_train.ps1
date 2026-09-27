# usage: .\run_train.ps1 D:\data\battlefield_v31
param([Parameter(Mandatory=$true)][string]$Data)
Set-Location $PSScriptRoot
python -m hubnet.bench
python -m hubnet.train --data $Data --out runs/hub1
python -m hubnet.evaluate --data $Data --ckpt runs/hub1/ckpt_best.pt
