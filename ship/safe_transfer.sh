#!/usr/bin/env bash
# Transfer data/battlefield_v31_8k to the workstation split-by-split, checking free disk
# space (via df on the remote) between each split, and aborting before it gets critical.
# This machine's disk is SHARED with another user's long-running job, so we never blindly
# stream 17GB in one shot again.
set -euo pipefail
SSH_KEY=~/.ssh/anc_workstation
HOST=ccps@172.1.40.74
MIN_FREE_GB=4   # abort if free space would drop below this after the next split

remote_free_gb() {
  ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=10 "$HOST" \
    "df -B1 / | tail -1 | awk '{print int(\$4/1073741824)}'"
}

cd "$(dirname "$0")/../data/battlefield_v31_8k"
ssh -i "$SSH_KEY" -o BatchMode=yes "$HOST" "mkdir -p ~/anc_work/data/battlefield_v31_8k"

for split in val test train words; do
  free=$(remote_free_gb)
  echo "before $split: ${free} GB free on remote"
  if [ "$free" -lt "$MIN_FREE_GB" ]; then
    echo "ABORT: only ${free} GB free, below ${MIN_FREE_GB} GB safety margin. Stopping."
    exit 1
  fi
  echo "=== transferring $split ==="
  tar cf - "$split" | ssh -i "$SSH_KEY" -o BatchMode=yes -o Compression=no "$HOST" \
    "tar xf - -C ~/anc_work/data/battlefield_v31_8k"
  echo "=== $split done, remote now $(remote_free_gb) GB free ==="
done

for f in DATASET.md battlefield_v31.yaml; do
  [ -f "$f" ] && scp -i "$SSH_KEY" -o BatchMode=yes "$f" "$HOST:~/anc_work/data/battlefield_v31_8k/" || true
done
echo "ALL DONE"
