"""Front-end simulation: ADC overload and radio codec.

Mixing in floating point means nothing ever overloads, so a model trained
that way has never seen a clipped gunshot and meets one in the field with no
idea what it is. Both functions here are applied to the *input* only; the
target stays dry.

There is deliberately no post-mix loudness normalisation anywhere in the
chain. Levels are set before mixing; clipping happens after. Normalising the
mixture afterwards would undo the very distortion this stage exists to add.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from .config import SR


def simulate_adc(x: np.ndarray, headroom_db: float, bits: int = 16) -> np.ndarray:
    """Random input level, hard clip at full scale, quantise.

    `headroom_db` < 0 means deliberate overload. Works on (T,) or (C, T);
    the same gain is applied to every channel (one preamp, one clip point).
    """
    peak = float(np.max(np.abs(x))) + 1e-9
    target = 10.0 ** (-headroom_db / 20.0)
    y = x * (target / peak)
    y = np.clip(y, -1.0, 1.0)
    q = float(2 ** (bits - 1) - 1)
    return (np.round(y * q) / q).astype(np.float32)


def _mu_law(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round(y * 127.0) / 127.0
    return np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu))) / mu


def radio_codec(x: np.ndarray, rng: np.random.Generator, sr: int = SR,
                band_hz: tuple[float, float] = (300.0, 3400.0),
                dropout_p: float = 0.02, frame_ms: float = 20.0) -> np.ndarray:
    """Cheap proxy for a tactical voice codec (CVSD / MELPe class).

    band-pass -> 8 kHz -> 8-bit mu-law companding -> random packet dropout
    -> back to `sr`. Not a real codec; the point is bandwidth loss,
    companding noise and frame erasures, which is what the model will see on
    a radio uplink. Swap in a real MELPe encoder later without touching the
    chain.
    """
    sos = butter(4, band_hz, btype="band", fs=sr, output="sos")
    y = sosfilt(sos, x).astype(np.float32)
    y8 = resample_poly(y, 1, sr // 8000).astype(np.float32)
    y8 = _mu_law(np.clip(y8, -1, 1)).astype(np.float32)
    n = int(8000 * frame_ms / 1000.0)
    n_frames = len(y8) // n
    drop = rng.random(n_frames) < dropout_p
    for i in np.where(drop)[0]:
        y8[i * n:(i + 1) * n] = 0.0
    y16 = resample_poly(y8, sr // 8000, 1).astype(np.float32)
    if len(y16) < len(x):
        y16 = np.pad(y16, (0, len(x) - len(y16)))
    return y16[: len(x)]


def limiter(x: np.ndarray, ceiling: float = 0.98, release_ms: float = 50.0, sr: int = SR) -> np.ndarray:
    """Demo-path safety limiter (not used in training). Instantaneous attack,
    slow release: no sample above `ceiling` ever reaches the model, and NaNs
    are zeroed. Zero added latency; it only trades peak for gain."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    a_rel = float(np.exp(-1.0 / (sr * release_ms / 1000.0)))
    env = 0.0
    out = np.empty_like(x)
    for i, v in enumerate(x):
        a = abs(float(v))
        env = max(a, a + a_rel * (env - a))
        out[i] = v * (ceiling / env) if env > ceiling else v
    return out.astype(np.float32)
