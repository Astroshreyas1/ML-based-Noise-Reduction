"""The capsule channel: what turns "speech plus noise" into "a recording".

Everything the boom mic hears goes through the same electronics, so speech
and noise share one colouration and one set of non-linearities:

    high-pass 90 Hz -> presence peak (+3 dB @ 3 kHz) -> low-pass 6.8 kHz
    -> pink self-noise floor -> tanh saturation -> AGC / limiter
    (instantaneous attack, ~250 ms release, threshold -8 dBFS, make-up)
    -> hard clip -> 16-bit quantisation

The AGC is the important part for realism: after every gunshot the whole
channel ducks for a quarter of a second, speech included, exactly as a real
headset does. Listening trial 2 (docs/NOISE_LAYERING.md section 6b) picked
this variant over the same scene without a shared channel.

Target rule (DECISIONS.md): the AGC gain is a *linear, time-varying gain*
applied to the mixture, so it is applied to the dry target too (the target
tracks the input speech level, as with the ADC gain in the old chain). EQ,
self-noise, saturation, clipping and quantisation are input-only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from .adc import radio_codec
from .audio import EPS, db_to_lin, rms
from .config import SR
from .physics import _pink

DEFAULT = dict(hp_hz=90.0, presence_hz=3000.0, presence_db=3.0, presence_q=1.0, lp_hz=6800.0,
               floor_dbfs=-55.0, drive=1.4, agc_threshold_dbfs=-8.0, agc_release_ms=250.0,
               makeup_db=6.0, bits=16, radio_p=0.0)
REF_DEFAULT = dict(hp_hz=60.0, presence_hz=2000.0, presence_db=0.0, presence_q=1.0, lp_hz=7200.0,
                   floor_dbfs=-52.0, drive=1.2, agc_threshold_dbfs=-6.0, agc_release_ms=150.0,
                   makeup_db=3.0, bits=16, radio_p=0.0)


def _biquad_peaking(fc: float, gain_db: float, q: float, sr: int):
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b = np.array([1 + alpha * a, -2 * cw, 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * cw, 1 - alpha / a])
    return b / den[0], den / den[0]


def _lowpass1(x: np.ndarray, fc: float, sr: int = SR) -> np.ndarray:
    """One-pole smoother started at steady state (no ramp-from-zero transient)."""
    k = np.exp(-2 * np.pi * fc / sr)
    y, _ = lfilter([1 - k], [1, -k], x, zi=[k * float(x[0])])
    return y.astype(np.float32)


def peak_hold_release(a: np.ndarray, release_coef: float, block: int = 4096) -> np.ndarray:
    """env[n] = max(a[n], env[n-1] * r), vectorised blockwise:
    within a block env = r^k * cummax(a * r^-k), carrying the previous block's
    last value. Exact, and ~100x faster than a Python loop."""
    a = np.asarray(a, dtype=np.float64)
    out = np.empty_like(a)
    carry = 0.0
    k = np.arange(block)
    up = release_coef ** (-k)          # grows to r^-4096 ~ e^1 for a 250 ms release: safe
    down = release_coef ** k
    for s in range(0, len(a), block):
        seg = a[s: s + block]
        m = len(seg)
        scaled = seg * up[:m]
        scaled[0] = max(scaled[0], carry * up[0] * release_coef)   # carry decays one step into the block
        env = np.maximum.accumulate(scaled) * down[:m]
        out[s: s + m] = env
        carry = env[-1]
    return out


def mic_eq(x: np.ndarray, p: dict[str, Any], sr: int = SR) -> np.ndarray:
    y = sosfilt(butter(2, p["hp_hz"], "hp", fs=sr, output="sos"), x).astype(np.float32)
    if p["presence_db"]:
        b, a = _biquad_peaking(p["presence_hz"], p["presence_db"], p["presence_q"], sr)
        y = lfilter(b, a, y).astype(np.float32)
    return sosfilt(butter(2, p["lp_hz"], "lp", fs=sr, output="sos"), y).astype(np.float32)


def capsule_channel(x: np.ndarray, rng: np.random.Generator, params: dict[str, Any] | None = None,
                    sr: int = SR) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Returns (output, agc_gain_trajectory, stats). The gain trajectory is
    the linear time-varying gain (AGC x make-up) to apply to the target."""
    p = {**DEFAULT, **(params or {})}
    n = len(x)
    y = mic_eq(x, p, sr)
    floor = _pink(n, rng).astype(np.float32)
    y = y + floor / (rms(floor) + EPS) * db_to_lin(p["floor_dbfs"])
    drive = float(p["drive"])
    y = (np.tanh(y * drive) / np.tanh(drive)).astype(np.float32)
    thr = db_to_lin(p["agc_threshold_dbfs"])
    rel = float(np.exp(-1.0 / (p["agc_release_ms"] / 1000.0 * sr)))
    env = peak_hold_release(np.abs(y), rel)
    gain = np.minimum(1.0, thr / (env + EPS)).astype(np.float32)
    gain = _lowpass1(gain, 800.0, sr) * db_to_lin(p["makeup_db"])
    y = y * gain
    if p.get("radio_p", 0) and rng.random() < p["radio_p"]:
        y = radio_codec(y, rng, sr, band_hz=(250.0, 4000.0), dropout_p=0.01)
    y = np.clip(y, -1, 1)
    q = float(2 ** (int(p["bits"]) - 1) - 1)
    y = (np.round(y * q) / q).astype(np.float32)
    stats = {"agc_min_gain_db": float(20 * np.log10(gain.min() / db_to_lin(p["makeup_db"]) + EPS)),
             "clip_pct": float(np.mean(np.abs(y) >= 0.999) * 100)}
    return y, gain.astype(np.float32), stats


def ref_channel(x: np.ndarray, rng: np.random.Generator, params: dict[str, Any] | None = None,
                sr: int = SR) -> tuple[np.ndarray, dict[str, float]]:
    y, _, st = capsule_channel(x, rng, {**REF_DEFAULT, **(params or {})}, sr)
    return y, st
