"""Lombard (vocal effort) transform and fitting.

People talking over 80+ dB SPL do not simply get louder. Measured changes
(Lu & Cooke 2008/2009, Garnier et al. 2010, Summers et al. 1988; see
docs/LIT_REVIEW.md section 3):

* F0 rises 30-80 Hz (up to 100+ at extreme effort) and F0 range widens
* spectral tilt flattens: 5-10 dB more energy in the 1-3 kHz band
* F1 rises (mouth opens wider); vowel space expands
* duration increases 5-20 % (vowels lengthen more than consonants)
* the voice becomes more periodic / tense (less breathy)
* level rises ~0.3-0.6 dB per dB of noise above ~50 dB SPL (Lombard slope)

The transform is applied to the *speech itself* before the training target is
captured (docs/REVIEW.md #2), so it is speech diversity, not a distortion the
network undoes. Its strength ``s`` in [0, 1] is coupled to the noise level of
the scene by the chain (dose-response), so quiet scenes carry calm speech and
loud scenes carry strained speech.

Implementation: WORLD vocoder decomposition (F0 / spectral envelope /
aperiodicity via ``pyworld``). Each parameter is modified in its own domain,
then resynthesised. A whole-spectrum pitch shift (librosa) moves the formants
with F0 and sounds like a smaller talker, not an effortful one; that path is
kept only as a fallback when pyworld is unavailable.

``fit_lombard_stats`` estimates the per-speaker spread from paired calm/loud
takes recorded in-house (scripts/record_session.md) and writes
``configs/lombard_stats.yaml``; until then the literature ranges above apply.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.signal import butter, sosfilt

from .audio import EPS, active_rms_db, load_mono
from .config import SR

# Full-strength (s = 1) ranges. Each example draws inside the range, then the
# chain scales the draw by s. Literature placeholders until fitted.
DEFAULT_STATS: dict[str, dict[str, float]] = {
    "f0_ratio":        {"low": 1.15, "high": 1.45},   # +15..45 % F0  (~ +30..80 Hz)
    "band_boost_db":   {"low": 5.0,  "high": 11.0},   # extra energy 1-3 kHz (tilt flattening)
    "low_cut_db":      {"low": 0.0,  "high": 3.0},    # slight loss below 300 Hz
    "f1_warp":         {"low": 1.04, "high": 1.12},   # F1 (and low formants) up 4..12 %
    "stretch":         {"low": 1.05, "high": 1.22},   # duration ratio
    "periodicity":     {"low": 0.3,  "high": 0.6},    # aperiodicity multiplied by (1 - this)
    "gain_db":         {"low": 6.0,  "high": 14.0},   # measured only; the chain sets level
}
FRAME_MS = 5.0


def _have_world() -> bool:
    try:
        import pyworld  # noqa: F401
        return True
    except ImportError:
        return False


def _draw(stats: dict[str, dict[str, float]], key: str, rng: np.random.Generator) -> float:
    st = stats[key]
    if "mean" in st:  # fitted: sample from the measured spread
        return float(rng.normal(st["mean"], st.get("std", 0.0)))
    return float(rng.uniform(st["low"], st["high"]))


def _band_gain_curve(freqs: np.ndarray, boost_db: float, low_cut_db: float) -> np.ndarray:
    """Smooth gain (linear amplitude) vs frequency: -low_cut below 300 Hz,
    ramp up through 500 Hz-1 kHz, +boost over 1-3 kHz, back to +boost/2 by 5 kHz."""
    lf = np.log2(np.maximum(freqs, 20.0))

    def ramp(a, b):  # 0 -> 1 between log2(a) and log2(b)
        return np.clip((lf - np.log2(a)) / (np.log2(b) - np.log2(a)), 0.0, 1.0)

    g_db = -low_cut_db * (1.0 - ramp(150.0, 400.0)) + boost_db * ramp(500.0, 1000.0) \
        - 0.5 * boost_db * ramp(3000.0, 5500.0)
    return 10.0 ** (g_db / 20.0)


def _warp_envelope(sp: np.ndarray, freqs: np.ndarray, f1_warp: float) -> np.ndarray:
    """Push low formants up by `f1_warp` (x), tapering to no warp above 2.5 kHz:
    env_new(f) = env(f / w(f))."""
    w = 1.0 + (f1_warp - 1.0) * (1.0 - np.clip((freqs - 1200.0) / 1300.0, 0.0, 1.0))
    src = freqs / w
    out = np.empty_like(sp)
    for i in range(sp.shape[0]):
        out[i] = np.interp(src, freqs, sp[i])
    return out


def _stretch_frames(a: np.ndarray, ratio: float) -> np.ndarray:
    n = a.shape[0]
    m = max(2, int(round(n * ratio)))
    src = np.linspace(0, n - 1, m)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    t = (src - i0)[:, None] if a.ndim == 2 else (src - i0)
    return (1 - t) * a[i0] + t * a[i1]


def lombard_world(x: np.ndarray, rng: np.random.Generator, strength: float = 1.0,
                  stats: dict[str, dict[str, float]] | None = None, sr: int = SR
                  ) -> tuple[np.ndarray, dict[str, Any]]:
    """WORLD-domain Lombard transform. Returns (audio, params). Output length
    changes with the duration stretch; the caller crops/pads."""
    import pyworld as pw

    st = stats or DEFAULT_STATS
    s = float(np.clip(strength, 0.0, 1.0))
    f0_ratio = 1.0 + s * (_draw(st, "f0_ratio", rng) - 1.0)
    boost = s * _draw(st, "band_boost_db", rng)
    low_cut = s * _draw(st, "low_cut_db", rng)
    f1_warp = 1.0 + s * (_draw(st, "f1_warp", rng) - 1.0)
    stretch = 1.0 + s * (_draw(st, "stretch", rng) - 1.0)
    period = s * _draw(st, "periodicity", rng)

    xd = np.ascontiguousarray(x, dtype=np.float64)
    f0, t = pw.dio(xd, sr, frame_period=FRAME_MS)
    f0 = pw.stonemask(xd, f0, t, sr)
    sp = pw.cheaptrick(xd, f0, t, sr)
    ap = pw.d4c(xd, f0, t, sr)
    freqs = np.linspace(0, sr / 2, sp.shape[1])

    voiced = f0 > 0
    f0_new = f0.copy()
    f0_new[voiced] = f0[voiced] * f0_ratio
    sp_new = _warp_envelope(sp, freqs, f1_warp) * _band_gain_curve(freqs, boost, low_cut)[None, :] ** 2
    ap_new = np.clip(ap * (1.0 - period), 0.0, 1.0)

    if abs(stretch - 1.0) > 1e-3:
        f0_new = _stretch_frames(f0_new, stretch)
        sp_new = _stretch_frames(sp_new, stretch)
        ap_new = _stretch_frames(ap_new, stretch)

    y = pw.synthesize(np.ascontiguousarray(f0_new), np.ascontiguousarray(sp_new),
                      np.ascontiguousarray(ap_new), sr, frame_period=FRAME_MS)
    y = np.nan_to_num(y).astype(np.float32)
    # keep the input's level; the chain owns gain
    y *= (np.sqrt(np.mean(x ** 2)) + EPS) / (np.sqrt(np.mean(y ** 2)) + EPS)
    params = {"strength": s, "f0_ratio": f0_ratio, "band_boost_db": boost, "low_cut_db": low_cut,
              "f1_warp": f1_warp, "stretch": stretch, "periodicity": period, "backend": "world"}
    return y, params


# --------------------------------------------------------------------------
# Fallback (no pyworld): tilt + global pitch shift + stretch. Crude; formants
# move with F0. Kept so the pipeline never silently drops the stage.
# --------------------------------------------------------------------------

def apply_spectral_tilt(x: np.ndarray, tilt_db_per_oct: float, sr: int = SR,
                        pivot_hz: float = 1000.0) -> np.ndarray:
    if abs(tilt_db_per_oct) < 1e-3:
        return x
    g = 10.0 ** (tilt_db_per_oct * 3.0 / 20.0)
    sos = butter(1, pivot_hz, btype="high", fs=sr, output="sos")
    return (x + (g - 1.0) * sosfilt(sos, x)).astype(np.float32)


def lombard_fallback(x: np.ndarray, rng: np.random.Generator, strength: float = 1.0,
                     stats: dict[str, dict[str, float]] | None = None, sr: int = SR
                     ) -> tuple[np.ndarray, dict[str, Any]]:
    import librosa

    st = stats or DEFAULT_STATS
    s = float(np.clip(strength, 0.0, 1.0))
    f0_ratio = 1.0 + s * (_draw(st, "f0_ratio", rng) - 1.0)
    stretch = 1.0 + s * (_draw(st, "stretch", rng) - 1.0)
    tilt = s * _draw(st, "band_boost_db", rng) / 3.0
    y = librosa.effects.time_stretch(x.astype(np.float32), rate=1.0 / stretch)
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=12.0 * np.log2(f0_ratio))
    y = apply_spectral_tilt(y, tilt, sr)
    y *= (np.sqrt(np.mean(x ** 2)) + EPS) / (np.sqrt(np.mean(y ** 2)) + EPS)
    return y.astype(np.float32), {"strength": s, "f0_ratio": f0_ratio, "stretch": stretch,
                                  "tilt_db_per_oct": tilt, "backend": "librosa_fallback"}


def lombard(x: np.ndarray, rng: np.random.Generator, stats: dict[str, Any] | None = None,
            sr: int = SR, strength: float = 1.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (perturbed, params). Length may change (duration stretch)."""
    if _have_world():
        return lombard_world(x, rng, strength, stats, sr)
    return lombard_fallback(x, rng, strength, stats, sr)


# --------------------------------------------------------------------------
# Fitting from in-house paired takes
# --------------------------------------------------------------------------

def spectral_band_ratio_db(x: np.ndarray, sr: int = SR) -> float:
    """Energy in 1-3 kHz relative to 100-500 Hz, in dB (tilt proxy)."""
    n = 2048
    spec = np.abs(np.fft.rfft(x, n=max(n, len(x)))) ** 2
    f = np.fft.rfftfreq(max(n, len(x)), 1.0 / sr)
    hi = spec[(f >= 1000) & (f <= 3000)].sum()
    lo = spec[(f >= 100) & (f <= 500)].sum()
    return float(10.0 * np.log10((hi + 1e-12) / (lo + 1e-12)))


def speech_duration_s(x: np.ndarray, sr: int = SR, frame_ms: float = 20.0, thresh_db: float = -40.0) -> float:
    n = int(sr * frame_ms / 1000.0)
    frames = x[: len(x) - len(x) % n].reshape(-1, n)
    e = 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + EPS)
    return float(np.sum(e > thresh_db) * frame_ms / 1000.0)


def median_f0_hz(x: np.ndarray, sr: int = SR) -> float:
    if _have_world():
        import pyworld as pw
        xd = np.ascontiguousarray(x, dtype=np.float64)
        f0, t = pw.dio(xd, sr, frame_period=FRAME_MS)
        f0 = pw.stonemask(xd, f0, t, sr)
        v = f0[f0 > 0]
        return float(np.median(v)) if len(v) else float("nan")
    import librosa
    f0, voiced, _ = librosa.pyin(x, fmin=60, fmax=400, sr=sr)
    v = f0[np.isfinite(f0) & voiced]
    return float(np.median(v)) if len(v) else float("nan")


def fit_lombard_stats(pairs: list[tuple[str | Path, str | Path]]) -> dict[str, dict[str, float]]:
    """pairs: [(calm_path, loud_path), ...], one per speaker, same sentences.
    Returns full-strength stats as {mean, std} per parameter (the chain scales
    by strength). Keeps std: the spread is the point."""
    if len(pairs) < 2:
        raise ValueError("need at least 2 speaker pairs to estimate a spread")
    f0r, boost, stretch, gain = [], [], [], []
    for calm_p, loud_p in pairs:
        calm, loud = load_mono(calm_p), load_mono(loud_p)
        f0r.append(median_f0_hz(loud) / max(median_f0_hz(calm), 1.0))
        boost.append(spectral_band_ratio_db(loud) - spectral_band_ratio_db(calm))
        stretch.append(speech_duration_s(loud) / max(speech_duration_s(calm), 1e-3))
        gain.append(active_rms_db(loud) - active_rms_db(calm))

    def ms(v: list[float]) -> dict[str, float]:
        a = np.asarray(v, dtype=np.float64)
        a = a[np.isfinite(a)]
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if len(a) > 1 else 0.0)}

    out = dict(DEFAULT_STATS)
    out.update({"f0_ratio": ms(f0r), "band_boost_db": ms(boost), "stretch": ms(stretch), "gain_db": ms(gain)})
    return out


def load_stats(path: str | Path | None) -> dict[str, dict[str, float]]:
    if path is None or not Path(path).exists():
        return DEFAULT_STATS
    with Path(path).open("r", encoding="utf-8") as fh:
        st = yaml.safe_load(fh)
    for k in DEFAULT_STATS:
        if k not in st:
            raise ValueError(f"{path}: missing lombard stat {k}")
    return st


def save_stats(stats: dict[str, dict[str, float]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(stats, fh, sort_keys=True)
