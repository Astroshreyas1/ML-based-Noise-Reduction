"""Source registry: index every source file once, assign splits permanently.

Why splits live here and not in the sampler: if the sampler decided splits at
draw time, one bug could leak test data into training. The registry writes a
``split`` column; the sampler only ever queries ``split == "train"`` and
``eval_only == False``. Leakage becomes structurally impossible.

Splits are assigned by hash, not randomly:

* speech        -> keyed on ``speaker_id`` (speaker-disjoint)
* ESC-50        -> keyed on the official ``fold`` (folds 1-3 train, 4 val, 5 test)
* everything else -> keyed on the relative path

Voice screening is mandatory for noise piles: a noise clip containing speech
teaches the model that voices should be removed. WebRTC VAD has false
positives on harmonic noise (rotor, siren), so flagged files are not deleted:
they are EXCLUDED from the manifest and listed in ``<root>/voice_flagged.txt``
for a human to listen to. Restore a genuine noise clip by adding its relative
path to ``<root>/voice_allowlist.txt`` and re-running the registry. Files that
really contain a voice should be deleted or moved with ``ancdata screen``.
"""
from __future__ import annotations

import csv
import hashlib
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import soundfile as sf

from .audio import crest_factor_db, load_mono
from .config import SR
from .paths import (EVAL_ONLY_PILES, PILES, data_root, manifest_path, pile_dir, pile_sclass,
                    quarantine_dir, to_relative)

AUDIO_EXT = {".wav", ".flac", ".ogg"}
MIN_DURATION_S = 0.2            # speech / ambience
MIN_TRANSIENT_S = 0.002         # impulsive / hard_negative clips are legitimately short

# ESC-50 category -> role. Anything not listed is excluded (animals, human
# non-speech such as coughing/laughing which would contaminate the noise pile).
ESC50_ROLE: dict[str, str] = {
    # continuous / ambience
    "helicopter": "ambience", "engine": "ambience", "siren": "ambience", "wind": "ambience",
    "rain": "ambience", "chainsaw": "ambience", "train": "ambience", "airplane": "ambience",
    "washing_machine": "ambience", "vacuum_cleaner": "ambience", "sea_waves": "ambience",
    "crackling_fire": "ambience", "car_horn": "ambience",
    # hard negatives: loud, broadband, NOT to be confused with gunfire
    "clapping": "hard_negative", "door_wood_knock": "hard_negative", "glass_breaking": "hard_negative",
    "footsteps": "hard_negative", "can_opening": "hard_negative", "door_wood_creaks": "hard_negative",
    "mouse_click": "hard_negative", "keyboard_typing": "hard_negative", "hand_saw": "hard_negative",
    # held out of training for the generalisation eval (unseen impulsive / broadband)
    "fireworks": "holdout", "thunderstorm": "holdout",
}
HOLDOUT_GROUP = "gen"  # name used in configs/eval_generalization.yaml

# Curated corpora whose noise classes are human-verified speech-free (ESC-50 is
# Freesound clips checked per class). The VAD gate is for YouTube-sourced piles
# (MAD, FSD50K scrapes) where shouting is documented; on ESC-50 it produced
# 292/880 false positives (sirens, horns, chainsaws) and zero true ones.
SCREEN_EXEMPT_PILES = frozenset({"esc50", "fixture_noise", "fixture_hardneg", "fixture_gunshot"})

# Military Audio Dataset class -> pile name under paths.PILES. "communication" is
# speech and must never be indexed.
MAD_CLASS_PILE: dict[str, str] = {
    "vehicle": "mad_vehicle", "helicopter": "mad_helicopter", "fighter": "mad_fighter",
    "shelling": "mad_shelling", "gunshot": "mad_gunshot", "footsteps": "mad_footsteps",
}

LICENCE: dict[str, str] = {
    "librispeech": "CC BY 4.0", "lombardgrid": "research (check terms)", "local_speakers": "consent form",
    "esc50": "CC BY-NC 3.0", "mad_vehicle": "CC BY 4.0", "mad_helicopter": "CC BY 4.0",
    "mad_fighter": "CC BY 4.0", "mad_shelling": "CC BY 4.0", "mad_gunshot": "CC BY 4.0",
    "mad_footsteps": "CC BY 4.0", "musan_noise": "CC BY 4.0", "esc50_gunshot": "CC BY-NC 4.0 (UrbanSound8K)",
    "field_gunshots": "check", "local_transients": "own recording", "realcheck": "consent form",
    "fixture_speech": "synthetic", "fixture_noise": "synthetic", "fixture_hardneg": "synthetic",
    "fixture_gunshot": "synthetic",
}


@dataclass
class Row:
    path: str            # relative to ANC_DATA_ROOT, POSIX
    pile: str
    sclass: str          # speech / ambience / impulsive / hard_negative / realcheck
    category: str        # helicopter, gunshot, ..., or "speech"
    duration_s: float
    sample_rate_native: int
    speaker_id: str | None
    fold: int | None
    crest_factor_db: float
    peak: float
    has_voice: bool
    licence: str
    split: str
    eval_only: bool
    holdout_group: str | None


def assign_split(key: str, ratios: tuple[float, float, float] = (0.90, 0.05, 0.05)) -> str:
    """Deterministic split from a stable key. Same key -> same split, forever."""
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < ratios[0]:
        return "train"
    if h < ratios[0] + ratios[1]:
        return "val"
    return "test"


def esc50_fold_split(fold: int) -> str:
    if fold in (1, 2, 3):
        return "train"
    if fold == 4:
        return "val"
    if fold == 5:
        return "test"
    raise ValueError(f"bad ESC-50 fold {fold}")


# --------------------------------------------------------------------------
# Voice screening
# --------------------------------------------------------------------------

def _periodic_fraction(audio: np.ndarray, sr: int, frame_ms: float = 30.0,
                       f_lo: float = 70.0, f_hi: float = 400.0, thresh: float = 0.5) -> np.ndarray:
    """Per-frame flag: normalised autocorrelation has a peak > `thresh` at a
    lag corresponding to a speech F0 (70-400 Hz). Cheap voicing detector that
    wind, engines and most broadband noise do not trigger."""
    n = int(sr * frame_ms / 1000.0)
    lag_lo, lag_hi = int(sr / f_hi), int(sr / f_lo)
    frames = audio[: len(audio) - len(audio) % n].reshape(-1, n)
    frames = frames - frames.mean(axis=1, keepdims=True)
    out = np.zeros(len(frames), dtype=bool)
    for i, fr in enumerate(frames):
        e0 = float(np.dot(fr, fr))
        if e0 < 1e-8:
            continue
        spec = np.fft.rfft(fr, n=2 * n)
        ac = np.fft.irfft(spec * np.conj(spec))[:n] / e0
        out[i] = ac[lag_lo:lag_hi].max() > thresh
    return out


def has_voice(audio: np.ndarray, sr: int = SR, aggressiveness: int = 3,
              min_voiced_frames: int = 8, min_voiced_frac: float = 0.10) -> bool:
    """Voice screen = WebRTC VAD AND F0-band periodicity on the same 30 ms frame.

    WebRTC alone flags almost any harmonic or low-passed noise; periodicity
    alone flags sirens. Requiring both keeps false positives to tonal cases a
    human can clear quickly. Raises ImportError if webrtcvad is missing —
    screening is never silently skipped.
    """
    try:
        import webrtcvad
    except ImportError as e:  # pragma: no cover
        raise ImportError("webrtcvad missing: pip install webrtcvad-wheels") from e
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    vad = webrtcvad.Vad(aggressiveness)
    n = int(sr * 0.03)
    frame_bytes = n * 2
    n_frames = (len(pcm) - frame_bytes) // frame_bytes
    if n_frames <= 0:
        return False
    periodic = _periodic_fraction(audio, sr)
    voiced = 0
    for i in range(n_frames):
        if i < len(periodic) and periodic[i] and vad.is_speech(pcm[i * frame_bytes:(i + 1) * frame_bytes], sr):
            voiced += 1
    return voiced >= min_voiced_frames and voiced / n_frames >= min_voiced_frac


# --------------------------------------------------------------------------
# Per-pile indexing
# --------------------------------------------------------------------------

def _audio_files(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXT)


def _esc50_meta(d: Path) -> dict[str, tuple[int, str]]:
    """filename -> (fold, category) from meta/esc50.csv, or from the filename
    pattern ``{fold}-{clip}-{take}-{target}.wav`` if the CSV is absent."""
    csv_paths = list(d.rglob("esc50.csv"))
    if csv_paths:
        out = {}
        with csv_paths[0].open("r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out[r["filename"]] = (int(r["fold"]), r["category"])
        return out
    return {}


def index_pile(name: str, screen: bool = True, verbose: bool = True) -> list[Row]:
    from tqdm import tqdm

    d = pile_dir(name)
    if not d.exists():
        return []
    sclass_default = pile_sclass(name)
    eval_only = name in EVAL_ONLY_PILES
    esc_meta = _esc50_meta(d) if name == "esc50" else {}
    rows: list[Row] = []
    excluded_clipped: list[str] = []
    for p in tqdm(_audio_files(d), desc=f"index {name}", disable=not verbose):
        info = sf.info(str(p))
        duration = float(info.frames) / float(info.samplerate)
        min_dur = MIN_TRANSIENT_S if sclass_default in ("impulsive", "hard_negative") else MIN_DURATION_S
        if duration < min_dur:
            raise ValueError(f"{p}: too short ({duration:.3f}s < {min_dur}s)")
        audio = load_mono(p)
        pk = float(np.abs(audio).max())
        cf = crest_factor_db(audio)

        sclass, category, fold, holdout, speaker = sclass_default, "unknown", None, None, None
        if name == "esc50":
            fn = p.name
            if fn in esc_meta:
                fold, category = esc_meta[fn]
            else:
                parts = p.stem.split("-")
                fold, category = int(parts[0]), f"target_{parts[-1]}"
            role = ESC50_ROLE.get(category)
            if role is None:
                continue  # excluded category
            if role == "holdout":
                sclass, holdout = "ambience", HOLDOUT_GROUP
            else:
                sclass = role
        elif sclass_default == "speech":
            speaker = p.stem.split("-")[0].split("_")[0]
            category = "speech"
            if name == "lombardgrid":
                # Lombard GRID: s<spk>_<l|p>_<sentence>.wav  (l = Lombard, p = plain)
                parts = p.stem.split("_")
                if len(parts) < 3 or parts[1] not in ("l", "p"):
                    raise ValueError(f"{p}: unexpected Lombard GRID filename; expected s<spk>_<l|p>_<sentence>")
                category = "speech_lombard" if parts[1] == "l" else "speech_plain"
            # LibriSpeech has a few peak-normalised files; only real clipping (many
            # full-scale samples) makes a target unusable
            if float(np.mean(np.abs(audio) > 0.999)) > 1e-3:
                excluded_clipped.append(to_relative(p))
                continue
        elif name.startswith("mad_"):
            category = name.split("_", 1)[1]
            if category == "shelling":
                holdout = HOLDOUT_GROUP
        elif name == "esc50_gunshot" or name == "field_gunshots" or name == "fixture_gunshot":
            category = "gunshot"
        elif sclass_default == "hard_negative":
            category = p.stem.split("_")[0]
        elif sclass_default == "ambience":
            category = p.parent.name if p.parent != d else p.stem.split("_")[0]

        voiced = False
        if screen and name not in SCREEN_EXEMPT_PILES and sclass in ("ambience", "impulsive", "hard_negative"):
            voiced = has_voice(audio)

        if sclass == "speech":
            split = assign_split(f"{name}:{speaker}")
        elif fold is not None:
            split = esc50_fold_split(fold)
        else:
            split = assign_split(to_relative(p))
        if eval_only:
            split = "test"

        rows.append(Row(
            path=to_relative(p), pile=name, sclass=sclass, category=category,
            duration_s=duration, sample_rate_native=int(info.samplerate), speaker_id=speaker,
            fold=fold, crest_factor_db=cf, peak=pk, has_voice=bool(voiced),
            licence=LICENCE.get(name, "unknown"), split=split, eval_only=eval_only,
            holdout_group=holdout,
        ))
    if excluded_clipped:
        rep = data_root() / "speech_clipped_excluded.txt"
        with rep.open("a", encoding="utf-8") as fh:
            fh.writelines(r + "\n" for r in excluded_clipped)
        if verbose:
            print(f"{name}: {len(excluded_clipped)} clipped speech files excluded -> {rep}")
    return rows


def _allowlist() -> set[str]:
    p = data_root() / "voice_allowlist.txt"
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")}


def validate(row: Row) -> None:
    min_dur = MIN_TRANSIENT_S if row.sclass in ("impulsive", "hard_negative") else MIN_DURATION_S
    if row.duration_s < min_dur:
        raise ValueError(f"{row.path}: too short")
    if row.sclass == "speech" and not row.speaker_id:
        raise ValueError(f"{row.path}: speech without speaker_id")
    if row.sclass in ("ambience", "impulsive", "hard_negative") and row.has_voice:
        raise ValueError(f"{row.path}: voice detected in a noise source")
    if row.split not in ("train", "val", "test"):
        raise ValueError(f"{row.path}: bad split {row.split}")


def build_manifest(piles: Iterable[str] | None = None, screen: bool = True,
                   out: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    piles = list(piles) if piles else [p for p in PILES if pile_dir(p).exists()]
    rows: list[Row] = []
    for name in piles:
        rows.extend(index_pile(name, screen=screen, verbose=verbose))
    if not rows:
        raise RuntimeError("no source files found under " + str(pile_dir(piles[0]).parent.parent))
    allow = _allowlist()
    kept, flagged = [], []
    for r in rows:
        if r.has_voice and r.path in allow:
            r.has_voice = False
        if r.has_voice:
            flagged.append(r)
            continue
        validate(r)
        kept.append(r)
    report = data_root() / "voice_flagged.txt"
    header = ("# VAD flagged these noise files; they are EXCLUDED from the manifest.\n"
              "# Listen to each. Genuine noise -> add its path to voice_allowlist.txt and re-run\n"
              "# `ancdata registry`. Real voice -> delete it or `ancdata screen --pile <name>`.\n")
    report.write_text(header + "".join(r.path + "\n" for r in flagged), encoding="utf-8")
    if verbose and flagged:
        print(f"voice screen: {len(flagged)} flagged -> EXCLUDED, listed in {report} (hand-check required)")
    if not kept:
        raise RuntimeError("every source file was excluded; check voice_flagged.txt")
    df = pd.DataFrame([asdict(r) for r in kept])
    out = Path(out) if out else manifest_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    if verbose:
        summary(df)
    return df


def load_manifest(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else manifest_path()
    if not path.exists():
        raise FileNotFoundError(f"manifest missing: {path}. Run `ancdata registry` first.")
    df = pd.read_parquet(path)
    required = {"path", "pile", "sclass", "category", "split", "eval_only", "holdout_group", "speaker_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: manifest missing columns {required - set(df.columns)}")
    return df


def summary(df: pd.DataFrame) -> None:
    print(f"indexed {len(df)} files -> {manifest_path()}")
    print(df.groupby(["sclass", "split"]).size().unstack(fill_value=0).to_string())
    sp = df[df.sclass == "speech"]
    if len(sp):
        per = sp.groupby("split").speaker_id.nunique()
        tr = set(sp[sp.split == "train"].speaker_id)
        te = set(sp[sp.split == "test"].speaker_id)
        print(f"speakers: {per.to_dict()}  disjoint: {'OK' if not (tr & te) else 'FAIL'}")
    ho = df[df.holdout_group.notna()]
    if len(ho):
        print(f"holdout group '{HOLDOUT_GROUP}': {len(ho)} files, categories {sorted(ho.category.unique())}")


def screen_pile(name: str, move: bool = True, verbose: bool = True) -> list[Path]:
    """Run VAD over a noise pile; move flagged files to quarantine/<pile>/.
    Returns the flagged list so a human can listen. Restore genuine noise by
    moving it back; delete anything with a voice."""
    from tqdm import tqdm

    d = pile_dir(name)
    if pile_sclass(name) == "speech":
        raise ValueError(f"{name} is a speech pile; screening is for noise piles")
    flagged: list[Path] = []
    for p in tqdm(_audio_files(d), desc=f"screen {name}", disable=not verbose):
        if has_voice(load_mono(p)):
            flagged.append(p)
    if move:
        q = quarantine_dir() / name
        q.mkdir(parents=True, exist_ok=True)
        for p in flagged:
            shutil.move(str(p), str(q / p.name))
    if verbose:
        print(f"{name}: {len(flagged)} flagged" + (f" -> {quarantine_dir() / name}" if move else ""))
    return flagged


def import_mad(mad_root: Path, verbose: bool = True) -> None:
    """Copy a Military Audio Dataset checkout into the pile layout by class.
    Expects <mad_root>/<class>/*.wav. 'communication' is skipped on purpose."""
    mad_root = Path(mad_root)
    for cls, pile in MAD_CLASS_PILE.items():
        src = mad_root / cls
        if not src.exists():
            continue
        dst = pile_dir(pile)
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in _audio_files(src):
            shutil.copy2(p, dst / p.name)
            n += 1
        if verbose:
            print(f"MAD {cls}: {n} files -> {dst}")
    if (mad_root / "communication").exists() and verbose:
        print("MAD communication: SKIPPED (human speech; never a noise source)")
