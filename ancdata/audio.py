"""Small audio helpers shared across the chain. All float32, mono, 16 kHz."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import SR

EPS = 1e-9


def load_mono(path: str | Path, sr: int = SR) -> np.ndarray:
    """Read any wav/flac, mix to mono, resample to `sr`, return float32."""
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if file_sr != sr:
        g = np.gcd(int(file_sr), int(sr))
        audio = resample_poly(audio, sr // g, file_sr // g).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + EPS))


def rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(rms(x) + EPS)


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)) + EPS)


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def normalise_rms(x: np.ndarray, target_db: float) -> np.ndarray:
    return (x * (db_to_lin(target_db) / rms(x))).astype(np.float32)


def active_rms_db(x: np.ndarray, sr: int = SR, frame_ms: float = 20.0, top_frac: float = 0.5) -> float:
    """RMS in dB over the loudest `top_frac` of frames. Ignores pauses, so a
    segment that is mostly silence but has one word still registers."""
    n = int(sr * frame_ms / 1000.0)
    if len(x) < n:
        return rms_db(x)
    frames = x[: len(x) - len(x) % n].reshape(-1, n)
    e = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + EPS)
    e.sort()
    top = e[int(len(e) * (1 - top_frac)):]
    return float(20.0 * np.log10(np.mean(top) + EPS))


def tile_or_crop(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random crop to n samples, tiling first if the clip is shorter."""
    if len(x) == 0:
        raise ValueError("empty audio")
    if len(x) < n:
        reps = int(np.ceil(n / len(x))) + 1
        x = np.tile(x, reps)
    start = int(rng.integers(0, len(x) - n + 1))
    return x[start:start + n]


def trim_silence(x: np.ndarray, thresh: float = 1e-3) -> np.ndarray:
    """Strip leading/trailing digital silence (ESC-50 pads short clips with
    zeros; scaling such a clip to a target RMS makes the audible part far too loud)."""
    idx = np.where(np.abs(x) > thresh)[0]
    if len(idx) == 0:
        return x
    return x[idx[0]: idx[-1] + 1]


def place_or_crop(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Speech-safe alternative to tile_or_crop: a short utterance is placed at
    a random offset inside `n` samples of silence (pauses are realistic;
    looping an utterance is not)."""
    if len(x) >= n:
        start = int(rng.integers(0, len(x) - n + 1))
        return x[start:start + n]
    out = np.zeros(n, dtype=np.float32)
    start = int(rng.integers(0, n - len(x) + 1))
    out[start:start + len(x)] = x
    return out


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def crest_factor_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10((np.max(np.abs(x)) + 1e-12) / (np.sqrt(np.mean(x ** 2)) + 1e-12)))


def event_crest_db(x: np.ndarray, sr: int = SR, window_ms: float = 200.0, pre_ms: float = 5.0) -> float:
    """Crest factor over a fixed window starting just before the global peak,
    zero-padded if the clip is shorter. Comparable between a 30 ms synthetic
    event and a 5 s field recording; the plain crest factor is not."""
    n = int(sr * window_ms / 1000.0)
    pk = int(np.argmax(np.abs(x)))
    a = max(0, pk - int(sr * pre_ms / 1000.0))
    w = x[a: a + n]
    if len(w) < n:
        w = np.pad(w, (0, n - len(w)))
    return float(20.0 * np.log10((np.abs(w).max() + 1e-12) / (np.sqrt(np.mean(w ** 2)) + 1e-12)))


def attack_time_ms(x: np.ndarray, sr: int = SR, window_ms: float = 1.5) -> float:
    """10 %-90 % rise time of the FIRST transient, in ms.

    Onset = first sample reaching 10 % of the global peak; the local peak is
    the maximum within `window_ms` after onset (so a 3-round burst or a
    ground-bounce echo does not stretch the measurement to the loudest round).
    """
    env = np.abs(x)
    pk = float(env.max())
    if pk <= 0:
        return float("nan")
    onset_idx = np.where(env >= 0.1 * pk)[0]
    if len(onset_idx) == 0:
        return float("nan")
    onset = int(onset_idx[0])
    w = int(sr * window_ms / 1000.0)
    local = onset + int(env[onset: onset + w].argmax())
    lo = np.where(env[onset: local + 1] >= 0.1 * env[local])[0]
    hi = np.where(env[onset: local + 1] >= 0.9 * env[local])[0]
    if len(lo) == 0 or len(hi) == 0:
        return float("nan")
    return float((hi[0] - lo[0]) / sr * 1000.0)


def write_wav(path: str | Path, x: np.ndarray, sr: int = SR, fmt: str = "float") -> None:
    """fmt: 'float' (32-bit wav, exact), 'pcm16' (16-bit wav, half size), 'flac' (16-bit, ~quarter size)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(x, dtype=np.float32).T if x.ndim == 2 else np.asarray(x, dtype=np.float32)
    if fmt == "float":
        sf.write(str(path), data, sr, subtype="FLOAT")
    elif fmt == "pcm16":
        sf.write(str(path), np.clip(data, -1, 1), sr, subtype="PCM_16")
    elif fmt == "flac":
        sf.write(str(path.with_suffix(".flac")), np.clip(data, -1, 1), sr, format="FLAC", subtype="PCM_16")
    else:
        raise ValueError(f"unknown fmt {fmt!r}")
