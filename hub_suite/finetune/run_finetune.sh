#!/usr/bin/env bash
# Workstation: fine-tune Laya on radio-hub typed decisions, then evaluate base vs fine-tuned.
#   ./finetune/run_finetune.sh            (1 GPU)      NGPU=2 ./finetune/run_finetune.sh   (DDP)
set -euo pipefail
cd "$(dirname "$0")/.."
python finetune/make_dataset.py --n 12000                       # regenerate (deterministic) if data/ is missing
if [ "${NGPU:-1}" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NGPU" finetune/train_laya.py --data finetune/data --out artefacts/laya_radio --epochs 3 --micro 16 --accum 2
else
  python finetune/train_laya.py --data finetune/data --out artefacts/laya_radio --epochs 3 --micro 32 --accum 1
fi
python finetune/eval_laya.py --data finetune/data --model artefacts/laya_radio --split val --n 600
python finetune/eval_laya.py --data finetune/data --model artefacts/laya_radio --split test
