#!/usr/bin/env bash
# Linux workstation: ./run_finetune.sh    multi-GPU: NGPU=2 ./run_finetune.sh
set -euo pipefail
cd "$(dirname "$0")"
if [ "${NGPU:-1}" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NGPU" train_laya.py --data data --out laya_radio --epochs 3 --micro 16 --accum 1
else
  python train_laya.py --data data --out laya_radio --epochs 3 --micro 16 --accum 2
fi
python eval_laya.py --data data --model laya_radio --split val --n 600
python eval_laya.py --data data --model laya_radio --split test
