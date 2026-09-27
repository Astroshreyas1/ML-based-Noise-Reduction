"""Radio-procedure speech: extra 6 s snippets with military / radio vocabulary.

    ancdata radio-snippets            -> data/snippets/radio6s/{audio/, meta.parquet}
    ancdata snippet-words             -> words.parquet next to each snippet set (faster-whisper word timings)

Sources (docs/research/RADIO_VOCAB_2026-09-27.md; user decision 2026-09-27: real + capped TTS):
    atcosim         real air-traffic-control controller speech (10 speakers; NATO alphabet callsigns, figures,
                    climb / descend / roger / correction). Same-speaker utterances joined with natural pauses.
    speechcommands  Google Speech Commands v2: digits, "tree", yes / no, up / down / left / right, forward /
                    backward, stop / go ... one speaker's words strung into figure / direction strings.
    tts             Piper LibriTTS-R (904 voices) speaking generated radio transmissions (radio_vocab.radio_phrase):
                    the only free source of wilco / niner / medevac / say again / I spell. Capped (config share),
                    never in the test split.

Every snippet is made Lombard-like with the repo's WORLD transform (DECISIONS.md: the target speech is
Lombard; these corpora are calm), level-normalised, 6.0 s. Splits by speaker (hash), like GRID / AVID.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from .audio import active_rms_db, db_to_lin, write_wav
from .config import SR
from .lombard import lombard
from .paths import data_root
from .registry import assign_split
from .snippets import best_cut, speech_fraction, trim_edges

SEG = int(6.0 * SR)
SC_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "tree", "yes", "no", "up",
            "down", "left", "right", "forward", "backward", "stop", "go", "follow", "on", "off"]


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x.astype(np.float32)
    g = np.gcd(int(sr), SR)
    return resample_poly(x, SR // g, int(sr) // g).astype(np.float32)


def _join(parts: list[np.ndarray], rng: np.random.Generator, gap: tuple[float, float]) -> np.ndarray:
    out: list[np.ndarray] = []
    for i, p in enumerate(parts):
        if i:
            out.append(np.zeros(int(rng.uniform(*gap) * SR), np.float32))
        out.append(p)
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def _finish(x: np.ndarray, rng: np.random.Generator, lead: float = 0.15) -> np.ndarray:
    """Lombard-ise, place in a 6 s frame (cut at a pause when longer), normalise the level."""
    y, _ = lombard(x, rng, strength=float(rng.uniform(0.5, 1.0)))
    y = np.asarray(y, np.float32)
    pad = int(rng.uniform(0.05, lead) * SR)
    y = np.concatenate([np.zeros(pad, np.float32), y])
    if len(y) > SEG:
        cut = best_cut(y, int(0.8 * SEG), SEG)
        y = y[:cut]
    y = np.pad(y, (0, SEG - len(y)))
    y = y * db_to_lin(-26.0 - active_rms_db(y))
    pk = float(np.abs(y).max())
    return (y * (0.9 / pk) if pk > 0.9 else y).astype(np.float32)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
def _atcosim_utterances() -> pd.DataFrame:
    root = data_root() / "raw" / "atcosim"
    frames = []
    for f in sorted(root.glob("*.parquet")):
        d = pd.read_parquet(f)
        d["_file"] = f.name
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _atcosim_audio(row: Any) -> np.ndarray:
    a = row["audio"]
    x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
    return _resample(x.mean(1), sr)


def build_atcosim(rng: np.random.Generator, per_speaker_s: float | None = None) -> list[dict[str, Any]]:
    df = _atcosim_utterances()
    if df.empty:
        return []
    spk_col = next((c for c in ("speaker_id", "speaker", "spk") if c in df.columns), None)
    txt_col = next((c for c in ("text", "transcription", "sentence") if c in df.columns), None)
    if spk_col is None:                           # ATCOSIM ids look like sm1_01_001: speaker = first token
        id_col = next(c for c in ("id", "segment", "file", "path") if c in df.columns)
        df["speaker_id"] = df[id_col].astype(str).str.extract(r"atcosim_([a-z]{2}\d)", expand=False).fillna("unk")
        spk_col = "speaker_id"
    out = []
    spks = sorted(df[spk_col].unique())             # only 10 talkers: fixed 8 / 1 / 1 split by speaker
    fixed = {s: "train" for s in spks}
    if len(spks) >= 3:
        fixed[spks[-2]], fixed[spks[-1]] = "val", "test"
    for spk, g in df.groupby(spk_col):
        g = g.sample(frac=1.0, random_state=int(rng.integers(2 ** 31))).reset_index(drop=True)
        split = fixed[spk]
        buf, words = [], []
        dur = 0.0
        for _, r in g.iterrows():
            x = trim_edges(_atcosim_audio(r))
            if len(x) < int(0.5 * SR):
                continue
            buf.append(x)
            words.append(str(r[txt_col]) if txt_col else "")
            dur += len(x) / SR + 0.4
            if dur >= 5.6:
                y = _finish(_join(buf, rng, (0.25, 0.7)), rng)
                out.append({"corpus": "atcosim", "speaker_id": f"atcosim_{spk}", "split": split,
                            "transcript": " / ".join(words), "audio": y, "licence": "ATCOSIM (R&D incl. commercial, no redistribution)"})
                buf, words, dur = [], [], 0.0
    return out


def build_speechcommands(rng: np.random.Generator, n_per_split: dict[str, int]) -> list[dict[str, Any]]:
    root = data_root() / "raw" / "speech_commands"
    if not root.exists():
        return []
    clips: dict[str, list[tuple[str, Path]]] = {}
    for w in SC_WORDS:
        for p in (root / w).glob("*.wav"):
            spk = p.stem.split("_nohash_")[0]
            clips.setdefault(spk, []).append((w, p))
    spk_by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for spk, cl in clips.items():
        if len(cl) >= 8:
            spk_by_split[assign_split(f"sc:{spk}")].append(spk)
    out = []
    for split, n in n_per_split.items():
        pool = sorted(spk_by_split[split])
        if not pool:
            continue
        for _ in range(n):
            spk = pool[int(rng.integers(len(pool)))]
            cl = clips[spk]
            order = rng.permutation(len(cl))
            parts, words, dur = [], [], 0.0
            for j in order:
                w, p = cl[j]
                x, sr = sf.read(str(p), dtype="float32")
                x = trim_edges(_resample(x, sr))
                if len(x) < int(0.15 * SR) or np.abs(x).max() < 0.01:
                    continue
                parts.append(x)
                words.append(w)
                dur += len(x) / SR + 0.2
                if dur >= 5.4:
                    break
            if len(parts) < 4:
                continue
            y = _finish(_join(parts, rng, (0.08, 0.35)), rng)
            out.append({"corpus": "speechcommands", "speaker_id": f"sc_{spk}", "split": split,
                        "transcript": " ".join(words), "audio": y, "licence": "CC BY 4.0"})
    return out


def build_tts(rng: np.random.Generator, n_per_split: dict[str, int]) -> list[dict[str, Any]]:
    from piper import PiperVoice, SynthesisConfig

    from .radio_vocab import radio_phrase
    model = data_root() / "raw" / "tts" / "en_US-libritts_r-medium.onnx"
    if not model.exists():
        return []
    voice = PiperVoice.load(str(model))
    ranges = {"train": (0, 800), "val": (800, 904)}            # speakers disjoint; TTS never in test
    out = []
    for split, n in n_per_split.items():
        if split not in ranges:
            continue
        for _ in range(n):
            spk = int(rng.integers(*ranges[split]))
            cfg = SynthesisConfig(speaker_id=spk, length_scale=float(rng.uniform(0.85, 1.1)),
                                  noise_scale=float(rng.uniform(0.5, 0.8)), noise_w_scale=float(rng.uniform(0.6, 1.0)))
            parts, texts, dur = [], [], 0.0
            while dur < 5.0 and len(parts) < 3:
                txt = radio_phrase(rng)
                chunks = list(voice.synthesize(txt, cfg))
                x = np.concatenate([c.audio_float_array for c in chunks])
                x = trim_edges(_resample(x, chunks[0].sample_rate))
                parts.append(x)
                texts.append(txt)
                dur += len(x) / SR + 0.5
            y = _finish(_join(parts, rng, (0.3, 0.8)), rng)
            out.append({"corpus": "tts", "speaker_id": f"piper_libritts_r_{spk}", "split": split,
                        "transcript": " ".join(texts), "audio": y, "licence": "Piper LibriTTS-R voice (CC BY 4.0 data)"})
    return out


# --------------------------------------------------------------------------
# Build the set
# --------------------------------------------------------------------------
def build_radio_snippets(out: Path | None = None, seed: int = 20260927,
                         sc_n: dict[str, int] | None = None, tts_n: dict[str, int] | None = None) -> pd.DataFrame:
    out = out or data_root() / "snippets" / "radio6s"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    items = build_atcosim(np.random.default_rng([seed, 1]))
    items += build_speechcommands(np.random.default_rng([seed, 2]), sc_n or {"train": 2000, "val": 150, "test": 150})
    items += build_tts(np.random.default_rng([seed, 3]), tts_n or {"train": 1200, "val": 100})
    rows = []
    counters: dict[str, int] = {}
    for it in items:
        c = it["corpus"]
        counters[c] = counters.get(c, 0) + 1
        sid = f"{c}_{counters[c]:05d}"
        rel = f"audio/{sid}.wav"
        write_wav(out / rel, it["audio"], fmt="pcm16")
        rows.append({"id": sid, "path": rel, "corpus": c, "kind": "radio", "speaker_id": it["speaker_id"], "gender": "U",
                     "effort": "lombard_world", "lombard_class": True, "split": it["split"], "seconds": 6.0,
                     "n_utterances": 1, "segments": "[]", "sentence_ids": "[]", "transcript": it["transcript"],
                     "speech_fraction": float(speech_fraction(it["audio"])), "active_rms_dbfs": float(active_rms_db(it["audio"])),
                     "peak": float(np.abs(it["audio"]).max()), "spl_leq_a_db": float("nan"), "native_sr": SR,
                     "licence": it["licence"], "seed": seed, "extra": "{}"})
    df = pd.DataFrame(rows)
    df.to_parquet(out / "meta.parquet", index=False)
    print(df.groupby(["corpus", "split"]).size().unstack(fill_value=0).to_string())
    del rng
    return df


# --------------------------------------------------------------------------
# Word timings (faster-whisper) for keyword spans
# --------------------------------------------------------------------------
_DIG = "zero one two three four five six seven eight nine".split()


def _expand(word: str, a: float, b: float) -> list[tuple[str, float, float]]:
    """Whisper writes figures as numerals ("479", "10-4"): spell digits out one per token, the word's
    time span split evenly, so radio figures count as keywords. Letters are lower-cased and stripped."""
    w = word.strip().lower()
    toks: list[str] = []
    for part in re.findall(r"[a-z][a-z\-']*|\d", w):
        toks.append(_DIG[int(part)] if part.isdigit() else re.sub(r"[^a-z\-]", "", part))
    toks = [x for x in toks if x]
    if not toks:
        return []
    d = (b - a) / len(toks)
    return [(x, a + i * d, a + (i + 1) * d) for i, x in enumerate(toks)]


def cuda12_dlls() -> None:
    """faster-whisper's CTranslate2 wheel links CUDA 12 cuBLAS / cuDNN; torch here ships CUDA 13.
    The nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels provide the DLLs; put them on the search path."""
    import os
    import site
    for sp in site.getsitepackages():
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
            d = Path(sp) / "Lib" / "site-packages" / sub if not (Path(sp) / sub).exists() else Path(sp) / sub
            if d.exists():
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def snippet_words(root: Path, model_size: str = "small.en", device: str = "cuda") -> pd.DataFrame:
    """Word timings for every snippet of a set -> root/words.parquet: id, words (JSON list of
    {w, start, end, kw}), n_keywords. Timings are whisper's (tens of ms accurate), enough for
    loss weighting and keyword-span metrics."""
    cuda12_dlls()
    from faster_whisper import WhisperModel

    from .radio_vocab import keyword_hits
    meta = pd.read_parquet(root / "meta.parquet")
    wm = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
    rows = []
    from tqdm import tqdm
    for r in tqdm(meta.itertuples(index=False), total=len(meta), desc=f"words {root.name}"):
        x, sr = sf.read(str(root / r.path), dtype="float32")
        segs, _ = wm.transcribe(x, language="en", word_timestamps=True, beam_size=1, vad_filter=False,
                                condition_on_previous_text=False)
        words = []
        for s in segs:
            for w in s.words or []:
                for tok, a0, b0 in _expand(w.word, float(w.start), float(w.end)):
                    words.append({"w": tok, "start": a0, "end": b0})
        kws = keyword_hits([w["w"] for w in words])
        for w, k in zip(words, kws):
            w["kw"] = bool(k)
        rows.append({"id": r.id, "words": json.dumps(words), "n_keywords": int(sum(k for k in kws))})
    df = pd.DataFrame(rows)
    df.to_parquet(root / "words.parquet", index=False)
    return df
