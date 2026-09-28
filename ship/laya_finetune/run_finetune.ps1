# Windows workstation: .\run_finetune.ps1        (single GPU; DDP/NCCL is Linux-only)
Set-Location $PSScriptRoot
python train_laya.py --data data --out laya_radio --epochs 3 --micro 16 --accum 2
python eval_laya.py --data data --model laya_radio --split val --n 600
python eval_laya.py --data data --model laya_radio --split test
