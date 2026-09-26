"""ITU-R BS.1770-4 loudness (mono) at any sample rate.

Why loudness and not RMS for the scene SNR: battlefield noise is bass-heavy
(engines, rotors, blasts). At equal RMS an engine sounds much quieter than
broadband noise, so an RMS-defined SNR under-doses low-frequency noise
perceptually and over-doses hiss. Scaper and EARS-WHAM set SNR in LUFS/LKFS
for the same reason (docs/NOISE_LAYERING.md section 2.2). The K-weighting
constants are the ones pyloudnorm derives for arbitrary rates.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .audio import EPS
from .config import SR


def _biquad_highshelf(fc: float, gain_db: float, q: float, sr: int):
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b0 = a * ((a + 1) + (a - 1) * cw + 2 * np.sqrt(a) * alpha)
    b1 = -2 * a * ((a - 1) + (a + 1) * cw)
    b2 = a * ((a + 1) + (a - 1) * cw - 2 * np.sqrt(a) * alpha)
    a0 = (a + 1) - (a - 1) * cw + 2 * np.sqrt(a) * alpha
    a1 = 2 * ((a - 1) - (a + 1) * cw)
    a2 = (a + 1) - (a - 1) * cw - 2 * np.sqrt(a) * alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _biquad_highpass(fc: float, q: float, sr: int):
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def k_weight(x: np.ndarray, sr: int = SR) -> np.ndarray:
    b1, a1 = _biquad_highshelf(1681.974450955533, 3.999843853973347, 0.7071752369554196, sr)
    b2, a2 = _biquad_highpass(38.13547087602444, 0.5003270373238773, sr)
    return lfilter(b2, a2, lfilter(b1, a1, np.asarray(x, dtype=np.float64)))


def loudness_lufs(x: np.ndarray, sr: int = SR) -> float:
    """Integrated loudness: K-weighting, 400 ms blocks with 75 % overlap,
    -70 LUFS absolute gate, -10 LU relative gate. Returns -70 for silence."""
    y = k_weight(x, sr)
    blk = int(0.4 * sr)
    hop = blk // 4
    if len(y) < blk:
        return float(-0.691 + 10 * np.log10(np.mean(y ** 2) + EPS))
    z = np.array([np.mean(y[i:i + blk] ** 2) for i in range(0, len(y) - blk + 1, hop)])
    lk = -0.691 + 10 * np.log10(z + EPS)
    keep = lk > -70.0
    if not keep.any():
        return -70.0
    rel = -0.691 + 10 * np.log10(np.mean(z[keep]) + EPS) - 10.0
    keep &= lk > rel
    if not keep.any():
        return -70.0
    return float(-0.691 + 10 * np.log10(np.mean(z[keep]) + EPS))


def gain_for_snr_lufs(speech: np.ndarray, noise: np.ndarray, snr_db: float, sr: int = SR) -> float:
    """Linear gain for `noise` so that L(speech) - L(noise * gain) = snr_db."""
    return float(10.0 ** ((loudness_lufs(speech, sr) - snr_db - loudness_lufs(noise, sr)) / 20.0))
