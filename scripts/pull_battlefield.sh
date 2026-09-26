#!/usr/bin/env bash
# Pull every openly licensed battlefield-relevant audio corpus into $ANC_DATA_ROOT/raw/<name>.
# Resumable; re-run to continue. Sorting into piles is a separate step (ancdata import-*).
#
#   bash scripts/pull_battlefield.sh            # everything below, sequentially, logs to raw/pull.log
#
# Needs Kaggle credentials (~/.kaggle/kaggle.json) for MAD only; everything else is direct.
set -uo pipefail
ROOT="${ANC_DATA_ROOT:-data}"; RAW="$ROOT/raw"; mkdir -p "$RAW"
ZGET="$(dirname "$0")/zget.sh"
Z7="/c/Program Files/7-Zip/7z.exe"; [[ -x "$Z7" ]] || Z7=7z
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$RAW/pull.log"; }
mark(){ log "MILESTONE $*"; }

# ---------------------------------------------------------------- FSD50K (CC BY / CC0 per clip)
# 200 classes incl. Gunshot, Machine gun, Explosion, Boom, Fireworks, Helicopter, Aircraft, Jet engine,
# Propeller, Engine, Heavy engine, Truck, Shout, Yell, Screaming, Crowd, Siren, Wind, Rain, ...
F="$RAW/fsd50k"; mkdir -p "$F"
for k in FSD50K.ground_truth.zip FSD50K.metadata.zip FSD50K.doc.zip FSD50K.eval_audio.zip FSD50K.eval_audio.z01 \
         FSD50K.dev_audio.zip FSD50K.dev_audio.z01 FSD50K.dev_audio.z02 FSD50K.dev_audio.z03 FSD50K.dev_audio.z04 FSD50K.dev_audio.z05; do
  bash "$ZGET" "https://zenodo.org/records/4060432/files/$k?download=1" "$F/$k" 8 2>&1 | tee -a "$RAW/pull.log"
done
if [[ ! -d "$F/FSD50K.eval_audio" ]]; then "$Z7" x -y -o"$F" "$F/FSD50K.eval_audio.zip" >/dev/null && mark "fsd50k eval extracted"; fi
if [[ ! -d "$F/FSD50K.dev_audio" ]]; then "$Z7" x -y -o"$F" "$F/FSD50K.dev_audio.zip" >/dev/null && mark "fsd50k dev extracted"; fi
for k in FSD50K.ground_truth.zip FSD50K.metadata.zip; do "$Z7" x -y -o"$F" "$F/$k" >/dev/null; done
mark "fsd50k done: $(find "$F" -name '*.wav' | wc -l) wavs"

# ---------------------------------------------------------------- UrbanSound8K (CC BY-NC 4.0): gun_shot, siren, engine_idling, jackhammer, drilling
U="$RAW/urbansound8k"; mkdir -p "$U"
bash "$ZGET" "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1" "$U/UrbanSound8K.tar.gz" 8 2>&1 | tee -a "$RAW/pull.log"
[[ -d "$U/UrbanSound8K" ]] || /c/Windows/System32/tar.exe -xzf "$U/UrbanSound8K.tar.gz" -C "$U"
mark "urbansound8k done: $(find "$U" -name '*.wav' | wc -l) wavs"

# ---------------------------------------------------------------- DEMAND 16 kHz (CC BY-SA 4.0): transport / street / field ambiences, 16-channel
D="$RAW/demand"; mkdir -p "$D"
for e in TBUS TCAR TMETRO STRAFFIC SPSQUARE NFIELD NPARK NRIVER; do
  bash "$ZGET" "https://zenodo.org/records/1227121/files/${e}_16k.zip?download=1" "$D/${e}_16k.zip" 4 2>&1 | tee -a "$RAW/pull.log"
  [[ -d "$D/$e" ]] || "$Z7" x -y -o"$D" "$D/${e}_16k.zip" >/dev/null
done
mark "demand done: $(find "$D" -name '*.wav' | wc -l) wavs"

# ---------------------------------------------------------------- Drone propellers (GitHub, Bebop/Mambo, indoor + augmented)
G="$RAW/drone_audio"; [[ -d "$G/.git" ]] || git clone -q --depth 1 https://github.com/saraalemadi/DroneAudioDataset "$G"
mark "drone_audio done: $(find "$G" -name '*.wav' | wc -l) wavs"

# ---------------------------------------------------------------- BGG / PUBG in-game gun sounds (eval-only realism reference)
B="$RAW/bgg"; [[ -d "$B/.git" ]] || git clone -q --depth 1 https://github.com/junwoopark92/PUBG-Gun-Sound-Dataset "$B"
mark "bgg repo cloned: $(find "$B" -name '*.wav' | wc -l) wavs (data may be linked externally; see README)"

# ---------------------------------------------------------------- RIRS_NOISES (Apache 2.0): real + simulated RIRs, point-source noises
R="$RAW/rirs_noises"; mkdir -p "$R"
bash "$ZGET" "https://www.openslr.org/resources/28/rirs_noises.zip" "$R/rirs_noises.zip" 8 2>&1 | tee -a "$RAW/pull.log"
[[ -d "$R/RIRS_NOISES" ]] || "$Z7" x -y -o"$R" "$R/rirs_noises.zip" >/dev/null
mark "rirs_noises done"

# ---------------------------------------------------------------- IDMT-Traffic (CC BY 4.0): 17,506 vehicle passings incl. trucks/buses, 2 s stereo
I="$RAW/idmt_traffic"; mkdir -p "$I"
bash "$ZGET" "https://zenodo.org/records/7551553/files/IDMT_Traffic.zip?download=1" "$I/IDMT_Traffic.zip" 8 2>&1 | tee -a "$RAW/pull.log"
[[ -d "$I/IDMT_Traffic" || -d "$I/audio" ]] || "$Z7" x -y -o"$I" "$I/IDMT_Traffic.zip" >/dev/null
mark "idmt_traffic done: $(find "$I" -name '*.wav' | wc -l) wavs"

# ---------------------------------------------------------------- MUSAN (CC BY 4.0): noise (930 files), music, speech (babble source)
M="$RAW/musan"; mkdir -p "$M"
bash "$ZGET" "https://www.openslr.org/resources/17/musan.tar.gz" "$M/musan.tar.gz" 8 2>&1 | tee -a "$RAW/pull.log"
[[ -d "$M/musan" ]] || /c/Windows/System32/tar.exe -xzf "$M/musan.tar.gz" -C "$M"
mark "musan done: $(find "$M" -name '*.wav' | wc -l) wavs"

# ---------------------------------------------------------------- MAD (CC BY 4.0) via Kaggle — credentials required
K="$RAW/mad"; mkdir -p "$K"
if command -v kaggle >/dev/null 2>&1 && [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
  kaggle datasets download -d junewookim/mad-dataset-military-audio-dataset -p "$K" --unzip 2>&1 | tee -a "$RAW/pull.log"
  mark "mad done: $(find "$K" -name '*.wav' | wc -l) wavs"
else
  mark "mad SKIPPED: no Kaggle credentials. Put kaggle.json in ~/.kaggle (pip install kaggle) and re-run, or download https://www.kaggle.com/datasets/junewookim/mad-dataset-military-audio-dataset manually into $K"
fi

mark "ALL-DONE"
