#!/usr/bin/env bash
# usage: ./run_train.sh /path/to/battlefield_v31 [extra args]   (multi-GPU: NGPU=4 ./run_train.sh ...)
set -euo pipefail
DATA=${1:?path to battlefield_v31}; shift || true
cd "$(dirname "$0")"
python -m hubnet.bench
if [ "${NGPU:-1}" -gt 1 ]; then
  torchrun --nproc_per_node="$NGPU" -m hubnet.train --data "$DATA" --out runs/hub1 "$@"
else
  python -m hubnet.train --data "$DATA" --out runs/hub1 "$@"
fi
python -m hubnet.evaluate --data "$DATA" --ckpt runs/hub1/ckpt_best.pt
