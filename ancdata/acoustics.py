"""Shared-space acoustics for the battlefield chain v4 (docs/NOISE_LAYERING.md section 8).

v3 mixed every layer dry, each with its own recording's room and noise floor,
and gave gunfire a frequency-flat noise tail. The ear hears that as "pasted
on" (Bregman's old-plus-new grouping; Traer & McDermott 2016 on reverb
statistics). v4 puts every *propagating* source of a clip into ONE drawn
environment, at its own distance:

    environment     open_field | forest | urban | cabin: mid-band RT60, echo count / spacing
    distance class  near 5-30 m | mid 30-150 m | far 150-1500 m; each sets level, direct-to-
                    reverberant ratio (DRR) and ISO 9613-1-like air absorption together
    IR              direct + ground reflection + 0-6 discrete echoes (terrain, tree line,
                    walls) + a diffuse tail whose decay is slowest at 150 Hz-4 kHz
                    (x0.7 below, x0.5 above), scaled to the drawn DRR

Also here: onset-aligned event crops with a raised-cosine fade-in and natural
decay, a downward expander that removes an event recording's own room tone,
Braun & Tashev (2020) random biquads per source, crossfaded concatenation
instead of looping, and a synthetic radio squelch.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

from .audio import EPS
from .config import SR

ENVIRONMENTS: dict[str, dict[str, Any]] = {
    # rt60: mid-band seconds; echoes: count range, delay range (s), level range (dB re direct)
    "open_field": {"rt60": (0.15, 0.4), "echoes": (2, 4), "echo_s": (0.06, 0.6), "echo_db": (-25.0, -12.0), "ground": True},
    "forest": {"rt60": (0.5, 1.2), "echoes": (1, 3), "echo_s": (0.04, 0.3), "echo_db": (-22.0, -12.0), "ground": True},
    "urban": {"rt60": (0.6, 1.6), "echoes": (3, 6), "echo_s": (0.02, 0.15), "echo_db": (-18.0, -8.0), "ground": True},
    "cabin": {"rt60": (0.08, 0.25), "echoes": (0, 0), "echo_s": (0.0, 0.0), "echo_db": (-30.0, -30.0), "ground": False},
}
DISTANCE_CLASSES: dict[str, dict[str, tuple[float, float]]] = {
    "near": {"m": (5.0, 30.0), "drr_db": (10.0, 20.0)},
    "mid": {"m": (30.0, 150.0), "drr_db": (0.0, 10.0)},
    "far": {"m": (150.0, 1500.0), "drr_db": (-10.0, 0.0)},
}


def draw_environment(kind: str, rng: np.random.Generator) -> dict[str, Any]:
    e = ENVIRONMENTS[kind]
    return {"kind": kind, "rt60_s": float(rng.uniform(*e["rt60"]))}


def draw_distance(cls: str, rng: np.random.Generator) -> dict[str, Any]:
    d = DISTANCE_CLASSES[cls]
    lo, hi = d["m"]
    return {"cls": cls, "m": float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), "drr_db": float(rng.uniform(*d["drr_db"]))}


# --------------------------------------------------------------------------
# Propagation
# --------------------------------------------------------------------------
def air_absorption_db(f_hz: np.ndarray, dist_m: float) -> np.ndarray:
    """Approximate ISO 9613-1 at 20 C / 50 % RH: ~0.5 dB/100 m at 1 kHz, 3 at 4 kHz, 10.5 at 8 kHz."""
    alpha = 0.03 * (np.maximum(f_hz, 1.0) / 4000.0) ** 1.8          # dB per metre
    return np.minimum(alpha * dist_m, 60.0)


def air_absorb(x: np.ndarray, dist_m: float, sr: int = SR) -> np.ndarray:
    if dist_m <= 1.0 or len(x) == 0:
        return x.astype(np.float32)
    n = int(2 ** np.ceil(np.log2(len(x) + 1)))
    X = np.fft.rfft(x, n)
    X *= 10 ** (-air_absorption_db(np.fft.rfftfreq(n, 1 / sr), dist_m) / 20)
    return np.fft.irfft(X, n)[: len(x)].astype(np.float32)


def outdoor_ir(env: dict[str, Any], drr_db: float, rng: np.random.Generator, sr: int = SR,
               ground: bool | None = None) -> np.ndarray:
    """One source position in the clip's environment: unit direct path first."""
    spec = ENVIRONMENTS[env["kind"]]
    rt60 = float(env["rt60_s"]) * float(rng.uniform(0.85, 1.15))        # position-to-position spread
    n = int((1.2 * rt60 + max(spec["echo_s"]) + 0.02) * sr)
    ir = np.zeros(n, dtype=np.float64)
    ir[0] = 1.0
    if spec["ground"] if ground is None else ground:
        d = int(rng.uniform(0.0005, 0.006) * sr)
        ir[max(1, d)] += float(rng.uniform(0.3, 0.7))
    rev = np.zeros(n, dtype=np.float64)
    lo, hi = spec["echoes"]
    for _ in range(int(rng.integers(lo, hi + 1))):
        k = int(rng.uniform(*spec["echo_s"]) * sr)
        if 0 < k < n:
            rev[k] += 10 ** (rng.uniform(*spec["echo_db"]) / 20) * rng.choice([-1.0, 1.0])
    if rev.any():                                                       # echoes lose top end on the way
        rev = lfilter([1 - np.exp(-2 * np.pi * 3500 / sr)], [1, -np.exp(-2 * np.pi * 3500 / sr)], rev)
    # diffuse tail, three bands with their own decay (Traer & McDermott: slowest mid band)
    pre = int(rng.uniform(0.004, 0.02) * sr)
    t = np.arange(n - pre) / sr
    tail = np.zeros(n - pre)
    for band, mult in (((None, 150.0), 0.7), ((150.0, 4000.0), 1.0), ((4000.0, None), 0.5)):
        w = rng.standard_normal(n - pre)
        if band[0] is None:
            w = sosfilt(butter(2, band[1], "lp", fs=sr, output="sos"), w)
        elif band[1] is None:
            w = sosfilt(butter(2, band[0], "hp", fs=sr, output="sos"), w)
        else:
            w = sosfilt(butter(2, band, "bp", fs=sr, output="sos"), w)
        tail += w * np.exp(-6.91 * t / (rt60 * mult))
    ramp = min(len(tail), int(0.005 * sr))
    tail[:ramp] *= np.linspace(0, 1, ramp)
    tail /= np.sqrt(np.sum(tail ** 2)) + EPS
    echo_e = float(np.sum(rev ** 2))
    rev[pre:] += tail * np.sqrt(max(echo_e, 0.05))                     # diffuse energy ~ echo energy, floor
    direct_e = float(np.sum(ir ** 2))
    rev *= np.sqrt(direct_e / (10 ** (drr_db / 10)) / (np.sum(rev ** 2) + EPS))
    return (ir + rev).astype(np.float32)


def propagate(x: np.ndarray, env: dict[str, Any], dist: dict[str, Any], rng: np.random.Generator, sr: int = SR,
              absorb: bool = True, ground: bool | None = None) -> np.ndarray:
    """Source recording -> what arrives at the listener: air absorption at the drawn
    distance, then this position's IR. Output is longer than the input by the tail."""
    y = air_absorb(x, dist["m"], sr) if absorb else x
    ir = outdoor_ir(env, dist["drr_db"], rng, sr, ground)
    return fftconvolve(y, ir).astype(np.float32)


def hull_transmission(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Exterior sound heard inside a vehicle: mass-law-ish low-pass, ~ -20 dB above 500 Hz."""
    y = sosfilt(butter(2, 400.0, "lp", fs=sr, output="sos"), x)
    return (y + 0.08 * x).astype(np.float32)


# --------------------------------------------------------------------------
# Event hygiene
# --------------------------------------------------------------------------
def _frame_db(x: np.ndarray, hop: int) -> np.ndarray:
    m = len(x) // hop
    if m == 0:
        return np.array([10 * np.log10(np.mean(x ** 2) + EPS)])
    return 10 * np.log10(np.mean(x[: m * hop].reshape(m, hop) ** 2, axis=1) + EPS)


def onset_crop(x: np.ndarray, max_s: float, sr: int = SR, search_from: int = 0, release_s: float = 0.15) -> np.ndarray:
    """Crop starting just before the first strong onset at/after `search_from`
    (level within 20 dB of the peak), 3 ms raised-cosine in, natural decay out:
    ends where the smoothed level falls 40 dB under the peak, with a short release."""
    x = np.asarray(x, dtype=np.float32)
    seg = x[search_from:] if search_from < len(x) - sr // 10 else x
    env = np.abs(seg)
    pk = float(env.max()) + EPS
    hit = np.where(env >= 0.1 * pk)[0]
    a = max(0, int(hit[0]) - int(0.008 * sr)) if len(hit) else 0
    y = seg[a: a + int(max_s * sr)].copy()
    hop = int(0.01 * sr)
    lv = _frame_db(y, hop)
    above = np.where(lv >= lv.max() - 40.0)[0]
    end = min(len(y), (int(above[-1]) + 1) * hop) if len(above) else len(y)
    y = y[:max(end, hop)]
    fi = min(len(y), int(0.003 * sr))
    y[:fi] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fi))).astype(np.float32)
    fo = min(len(y) // 2, int(release_s * sr))
    if fo > 0:
        y[-fo:] *= np.exp(-np.linspace(0, 5, fo)).astype(np.float32)
    return y


def expander(x: np.ndarray, sr: int = SR, over_db: float = 6.0, ratio: float = 4.0, max_cut_db: float = 30.0) -> np.ndarray:
    """Downward expander: frames under (L90 + over_db) are pushed down (ratio-1) dB per dB,
    so an event recording's own background does not ride in with the event."""
    hop = int(0.01 * sr)
    lv = _frame_db(x, hop)
    if len(lv) < 4:
        return x
    thr = float(np.percentile(lv, 10)) + over_db
    g_db = np.clip((lv - thr) * (ratio - 1.0), -max_cut_db, 0.0)
    g_db = np.where(lv < thr, g_db, 0.0)
    # 5 ms attack / 80 ms release on the frame gains, then to samples
    out = np.empty_like(g_db)
    prev = g_db[0]
    ka, kr = np.exp(-0.01 / 0.005), np.exp(-0.01 / 0.08)
    for i, g in enumerate(g_db):
        k = ka if g > prev else kr                 # opening (gain rising) fast, closing slow
        prev = k * prev + (1 - k) * g
        out[i] = prev
    gs = np.interp(np.arange(len(x)), np.arange(len(out)) * hop + hop / 2, out)
    return (x * 10 ** (gs / 20)).astype(np.float32)


def random_biquad(x: np.ndarray, rng: np.random.Generator, r: float = 3.0 / 8.0) -> np.ndarray:
    """Braun & Tashev (2020) spectral augmentation: H(z) = (1 + r1 z^-1 + r2 z^-2) / (1 + r3 z^-1 + r4 z^-2)."""
    c = rng.uniform(-r, r, 4)
    y = lfilter([1.0, c[0], c[1]], [1.0, c[2], c[3]], x)
    return (y * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(y ** 2)) + EPS))).astype(np.float32)


def concat_crossfade(pieces: list[np.ndarray], n: int, xfade_s: float = 0.6, sr: int = SR) -> np.ndarray:
    """Different segments joined with equal-power crossfades until n samples; never a loop of one clip
    unless the pieces run out (then the list is cycled -- the plan draws enough pieces to avoid it)."""
    xf = int(xfade_s * sr)
    out = np.zeros(0, dtype=np.float32)
    i = 0
    while len(out) < n and pieces:
        p = np.asarray(pieces[i % len(pieces)], dtype=np.float32)
        i += 1
        if len(p) < 2 * xf + 16:
            p = np.concatenate([p, p[::-1]]) if len(p) > 16 else np.zeros(2 * xf + 16, np.float32)
        p = p / (np.sqrt(np.mean(p ** 2)) + EPS)
        if len(out) == 0:
            out = p.copy()
            continue
        k = min(xf, len(out), len(p))
        w = np.linspace(0, np.pi / 2, k, dtype=np.float32)
        joint = out[-k:] * np.cos(w) + p[:k] * np.sin(w)
        out = np.concatenate([out[:-k], joint, p[k:]])
        if i > 64:
            break
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return out[:n].astype(np.float32)


def radio_squelch(rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Handset-speaker squelch burst: click, 0.1-0.35 s band-limited hiss, decaying tail."""
    dur = float(rng.uniform(0.1, 0.35))
    m = int(dur * sr)
    hiss = sosfilt(butter(4, (300.0, 3400.0), "bp", fs=sr, output="sos"), rng.standard_normal(m))
    env = np.ones(m)
    tail = int(min(m // 2, 0.08 * sr))
    env[-tail:] = np.exp(-np.linspace(0, 6, tail))
    y = hiss * env
    click = np.zeros(m)
    click[: int(0.002 * sr)] = rng.choice([-1.0, 1.0]) * 3.0 * np.hanning(int(0.002 * sr))
    y = y + sosfilt(butter(2, 3000.0, "lp", fs=sr, output="sos"), click)
    return (y / (np.abs(y).max() + EPS)).astype(np.float32)
