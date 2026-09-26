#!/usr/bin/env bash
# Download the public source piles into $ANC_DATA_ROOT/sources (default ./data/sources).
# Idempotent: skips anything already extracted. Run in the background at hour 0.
#
#   bash scripts/download_sources.sh            # LibriSpeech dev-clean + train-clean-100, ESC-50
#   bash scripts/download_sources.sh --all      # + Zenodo gunshot range set, MAD (Kaggle CLI), RIRS_NOISES
#
# Manual (forms / logins):
#   UrbanSound8K  https://urbansounddataset.weebly.com/urbansound8k.html  -> unzip, then copy the
#                 class 6 (gun_shot) wavs into sources/impulsive/esc50_gunshot/
#   Lombard GRID  https://spandh.dcs.shef.ac.uk/avlombard/  -> audio/*.wav into sources/speech/lombardgrid/
set -euo pipefail
ROOT="${ANC_DATA_ROOT:-data}/sources"
mkdir -p "$ROOT"
ALL=0; [[ "${1:-}" == "--all" ]] && ALL=1

fetch() {  # url, dest file
  if [[ -f "$2" ]]; then echo "have $2"; else echo "get  $1"; curl -L --retry 5 -o "$2" "$1"; fi
}

# --- LibriSpeech (CC BY 4.0) ------------------------------------------------
mkdir -p "$ROOT/speech/librispeech" /tmp/ancdl
for part in dev-clean train-clean-100; do
  if [[ -d "$ROOT/speech/librispeech/LibriSpeech/$part" ]]; then echo "have LibriSpeech/$part"; continue; fi
  fetch "https://www.openslr.org/resources/12/$part.tar.gz" "/tmp/ancdl/$part.tar.gz"
  tar -xzf "/tmp/ancdl/$part.tar.gz" -C "$ROOT/speech/librispeech"
done

# --- ESC-50 (CC BY-NC 3.0) --------------------------------------------------
if [[ ! -d "$ROOT/noise/esc50/ESC-50-master" ]]; then
  mkdir -p "$ROOT/noise/esc50"
  fetch "https://github.com/karolpiczak/ESC-50/archive/master.zip" /tmp/ancdl/esc50.zip
  unzip -q -o /tmp/ancdl/esc50.zip -d "$ROOT/noise/esc50"
else echo "have ESC-50"; fi

if [[ $ALL == 1 ]]; then
  # --- Multi-firearm multi-orientation gunshot set (Zenodo 7004819) --------
  mkdir -p "$ROOT/impulsive/field"
  if [[ -z "$(ls -A "$ROOT/impulsive/field")" ]]; then
    echo "Zenodo record 7004819: fetching file list"
    curl -s "https://zenodo.org/api/records/7004819" \
      | python3 -c 'import json,sys; [print(f["links"]["self"], f["key"]) for f in json.load(sys.stdin)["files"]]' \
      | while read -r url key; do fetch "$url" "/tmp/ancdl/$key"; done
    for z in /tmp/ancdl/*.zip; do case "$z" in *esc50*) ;; *) unzip -q -o "$z" -d "$ROOT/impulsive/field" ;; esac; done
  else echo "have Zenodo gunshots"; fi

  # --- Military Audio Dataset (CC BY 4.0) via Kaggle CLI -------------------
  if command -v kaggle >/dev/null 2>&1; then
    mkdir -p /tmp/ancdl/mad
    kaggle datasets download -d junewookim/mad-dataset-military-audio-dataset -p /tmp/ancdl/mad --unzip || true
    python3 -m ancdata.cli import-mad /tmp/ancdl/mad || echo "import-mad: check the extracted layout (<root>/<class>/*.wav)"
  else echo "kaggle CLI not found: download MAD manually, then: ancdata import-mad <dir>"; fi

  # --- RIRS_NOISES (Apache 2.0), optional real RIRs -------------------------
  fetch "https://www.openslr.org/resources/28/rirs_noises.zip" /tmp/ancdl/rirs_noises.zip
  mkdir -p "$ROOT/rir/real" && unzip -q -o /tmp/ancdl/rirs_noises.zip -d "$ROOT/rir/real"
fi
echo "done. next: ancdata rir && ancdata registry && ancdata selftest --config configs/train.yaml"
