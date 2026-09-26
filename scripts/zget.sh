#!/usr/bin/env bash
# Ranged-parallel downloader for slow single-stream hosts (Zenodo, OpenSLR).
#   bash scripts/zget.sh <url> <out_file> [parts=8]
# Idempotent: skips if out_file already has the full Content-Length; each part resumes.
set -euo pipefail
URL="$1"; OUT="$2"; N="${3:-8}"
mkdir -p "$(dirname "$OUT")"
# HEAD until we get a real file (Zenodo rate-limits with a tiny HTML page whose Content-Length is ~92 B)
SIZE=""
for h in $(seq 1 12); do
  HDR=$(curl -sIL "$URL")
  SIZE=$(echo "$HDR" | grep -i "^content-length" | tail -1 | tr -dc '0-9')
  CT=$(echo "$HDR" | grep -i "^content-type" | tail -1)
  if [[ -n "$SIZE" && "$SIZE" -gt 100000 ]] && ! echo "$CT" | grep -qi "text/html"; then break; fi
  echo "zget: HEAD attempt $h got size=${SIZE:-none} ($CT); waiting"; SIZE=""; sleep 20
done
if [[ -z "$SIZE" ]]; then echo "zget: could not get a valid content-length for $URL"; exit 1; fi
if [[ -f "$OUT" && "$(stat -c %s "$OUT")" == "$SIZE" ]]; then echo "have $OUT"; exit 0; fi
TMP="$OUT.parts"; mkdir -p "$TMP"
CH=$(( SIZE / N + 1 ))
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$(( (i+1)*CH - 1 )); [[ $e -ge $SIZE ]] && e=$((SIZE-1))
  [[ $s -gt $e ]] && continue
  echo "$i $s $e"
done > "$TMP/ranges.txt"
# each part resumes from its own partial length; the whole phase repeats until every part is complete
for attempt in 1 2 3 4 5 6 7 8; do
  cat "$TMP/ranges.txt" | xargs -P "$N" -L 1 bash -c '
    i=$0; s=$1; e=$2; p="'"$TMP"'/part_$i"
    have=0; [[ -f "$p" ]] && have=$(stat -c %s "$p")
    want=$((e - s + 1))
    if [[ $have -ge $want ]]; then exit 0; fi
    curl -sL --retry 20 --retry-all-errors --retry-delay 3 -r "$((s+have))-$e" -o "$p.tmp" "'"$URL"'"; [[ -f "$p.tmp" ]] && cat "$p.tmp" >> "$p"; rm -f "$p.tmp"   # partial bytes are contiguous: keep them
  '
  complete=1
  while read -r i s e; do
    p="$TMP/part_$i"; have=0; [[ -f "$p" ]] && have=$(stat -c %s "$p")
    [[ $have -ge $((e - s + 1)) ]] || complete=0
  done < "$TMP/ranges.txt"
  [[ $complete == 1 ]] && break
  echo "zget: attempt $attempt incomplete for $OUT, retrying parts"; sleep 10
done
cat $(ls "$TMP"/part_* | sort -t_ -k2 -n) > "$OUT"
GOT=$(stat -c %s "$OUT")
if [[ "$GOT" != "$SIZE" ]]; then echo "zget: size mismatch $GOT != $SIZE for $OUT (parts kept; re-run to resume)"; rm -f "$OUT"; exit 1; fi
rm -rf "$TMP"
echo "done $OUT ($((SIZE/1048576)) MB)"
