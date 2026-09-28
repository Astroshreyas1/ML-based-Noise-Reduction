"""Tactical radio link simulation (docs/research/RADIO_LINK_2026-09-26.md, docs/NOISE_LAYERING.md section 10).

Makes the boom signal sound like it came over a combat-net radio instead of a
studio mic. One link type per clip:

    cvsd16  0.55   16 kbit/s CVSD (SINCGARS / VINSON secure voice): MIL-STD-188-113 delta modulator,
                   3-bit run-of-threes, 5 ms syllabic filter, 16:1 step ratio, 1 ms leaky integrator;
                   Gilbert-Elliott burst bit errors on the bitstream (the leaky integrator is the concealment)
    fm      0.35   analog narrowband FM (plain voice): +6 dB/oct pre-emphasis, deviation clipping, splatter
                   filter, FM hiss at CNR 6-30 dB with Rayleigh fading and Rice clicks below threshold,
                   de-emphasis, squelch tail
    cvsd32  0.10   32 kbit/s CVSD (cleaner)

TX front end (all): VOX/PTT front-edge clipping of the first syllable (p 0.3) with a key-up click, AGC /
compressor, 300-3000 (FM) or 300-3400 Hz (CVSD) band-pass. RX: 300-3400 Hz band-pass.

Target rule (reading B of the research: receive-side enhancement): the target gets ONLY the linear
band-pass filters of the link -- linear-phase FIRs with their delay removed, so input and target stay
sample-aligned -- never the codec, clipping, noise or errors. A mask model cannot create 3.4-8 kHz
content the radio removed, so the target does not ask for it. `loss_mask` marks the VOX-muted head,
which no causal model can recover.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

EPS = 1e-9


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))

try:                                             # optional: 20x faster CVSD
    from numba import njit
except ImportError:                              # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

SR = 16000
SR_LINK = 8000
DEFAULT = dict(link_p={"cvsd16": 0.55, "fm": 0.35, "cvsd32": 0.10}, vox_p=0.3, squelch_p=0.4,
               agc_ratio=(4.0, 10.0), fm_cnr_db=(6.0, 30.0), ber_good=(1e-4, 1e-3), ber_bad=(0.05, 0.2),
               bad_occupancy=(0.0, 0.15), burst_ms=(20.0, 100.0))


# --------------------------------------------------------------------------
# Linear-phase filters (delay removed -> sample-aligned)
# --------------------------------------------------------------------------
def _fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    d = (len(h) - 1) // 2
    return np.convolve(x, h, mode="full")[d: d + len(x)].astype(np.float32)


def bandpass(x: np.ndarray, lo: float, hi: float, sr: int = SR_LINK, taps: int = 255) -> np.ndarray:
    return _fir(x, firwin(taps, [lo, hi], pass_zero=False, fs=sr))


def lowpass(x: np.ndarray, hi: float, sr: int = SR_LINK, taps: int = 127) -> np.ndarray:
    return _fir(x, firwin(taps, hi, fs=sr))


def link_filters(x16: np.ndarray, tx_hi: float) -> np.ndarray:
    """The linear part of the link, 16 kHz in and out: what the TARGET gets."""
    y = resample_poly(x16, 1, 2).astype(np.float32)
    y = bandpass(y, 300.0, tx_hi)
    y = bandpass(y, 300.0, 3400.0)
    return resample_poly(y, 2, 1)[: len(x16)].astype(np.float32)


# --------------------------------------------------------------------------
# CVSD (MIL-STD-188-113)
# --------------------------------------------------------------------------
@njit(cache=True)
def _cvsd_encode(x, beta_s, beta_p, d_min, d_max):
    n = len(x)
    bits = np.zeros(n, dtype=np.uint8)
    y = 0.0
    v = 0.0
    r0 = 0
    r1 = 0
    r2 = 0
    g = 1.0 - beta_s
    for i in range(n):
        b = 1 if x[i] >= y else 0
        r0, r1, r2 = r1, r2, b
        ovl = 1.0 if (r0 == r1 and r1 == r2) else 0.0
        v = beta_s * v + g * ovl
        step = d_min + (d_max - d_min) * v
        y = beta_p * y + (step if b == 1 else -step)
        bits[i] = b
    return bits


@njit(cache=True)
def _cvsd_decode(bits, beta_s, beta_p, d_min, d_max):
    n = len(bits)
    out = np.zeros(n, dtype=np.float32)
    y = 0.0
    v = 0.0
    r0 = 0
    r1 = 0
    r2 = 0
    g = 1.0 - beta_s
    for i in range(n):
        b = bits[i]
        r0, r1, r2 = r1, r2, b
        ovl = 1.0 if (r0 == r1 and r1 == r2) else 0.0
        v = beta_s * v + g * ovl
        step = d_min + (d_max - d_min) * v
        y = beta_p * y + (step if b == 1 else -step)
        out[i] = y
    return out


def gilbert_elliott(n: int, rate: int, rng: np.random.Generator, p: dict[str, Any]) -> np.ndarray:
    """Bit-error mask with bursts (ITU-T G.191 EID style): good / bad state, per-state BER."""
    occ = float(rng.uniform(*p["bad_occupancy"]))
    burst = float(rng.uniform(*p["burst_ms"])) / 1000.0 * rate           # mean bad-state length in bits
    p_bg = 1.0 / max(burst, 1.0)
    p_gb = p_bg * occ / max(1e-6, 1.0 - occ)
    ber_g, ber_b = float(rng.uniform(*p["ber_good"])), float(rng.uniform(*p["ber_bad"]))
    # state sequence by run lengths (geometric), then per-bit flips
    state = np.zeros(n, dtype=bool)
    i, bad = 0, False
    while i < n:
        q = p_bg if bad else max(p_gb, 1e-9)
        L = int(rng.geometric(min(q, 1.0)))
        state[i: i + L] = bad
        i += L
        bad = not bad
    ber = np.where(state, ber_b, ber_g)
    return rng.random(n) < ber


def cvsd_link(x8: np.ndarray, kbps: int, rng: np.random.Generator, p: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    fs = kbps * 1000
    up = fs // SR_LINK
    x = resample_poly(x8, up, 1).astype(np.float64)
    x = x / (np.abs(x).max() + EPS) * 0.9
    beta_s, beta_p = np.exp(-1 / (fs * 0.005)), np.exp(-1 / (fs * 0.001))
    d_min = 0.08 * (16000 / fs)                 # min step, calibrated (2026-09-27): clean speech SI-SNR 10.4 dB @16k,
                                                # 17.5 dB @32k (smaller steps slope-overload); 16:1 ratio (MIL-STD 12:1-21:1)
    d_max = 16.0 * d_min
    bits = _cvsd_encode(x, beta_s, beta_p, d_min, d_max)
    flips = gilbert_elliott(len(bits), fs, rng, p)
    bits = bits ^ flips.astype(np.uint8)
    y = _cvsd_decode(bits, beta_s, beta_p, d_min, d_max).astype(np.float32)
    y = resample_poly(lowpass(y, 3400.0, fs, 255), 1, up)[: len(x8)].astype(np.float32)
    return y, {"ber": float(flips.mean())}


# --------------------------------------------------------------------------
# Analog narrowband FM
# --------------------------------------------------------------------------
def fm_link(x8: np.ndarray, rng: np.random.Generator, p: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    pre_b = np.array([1.0, -0.8])               # +6 dB/oct over the voice band (~750 us)
    e = lfilter(pre_b, [1.0], x8).astype(np.float32)
    e = e / (np.abs(e).max() + EPS)
    clip = float(rng.uniform(0.5, 0.9))         # deviation limiter
    e = np.clip(e, -clip, clip) / clip
    e = lowpass(e, 3000.0)                      # splatter filter
    n = len(e)
    cnr = float(rng.uniform(*p["fm_cnr_db"]))
    # Rayleigh fading on the CNR (Doppler 2-40 Hz), per 5 ms block
    dop = float(rng.uniform(2.0, 40.0))
    blk = int(0.005 * SR_LINK)
    nb = n // blk + 2
    c = rng.standard_normal(nb) + 1j * rng.standard_normal(nb)
    k = max(1, int(1.0 / (dop * 0.005)))
    c = np.convolve(c, np.ones(k) / np.sqrt(k), mode="same")
    fade = np.abs(c) / (np.sqrt(np.mean(np.abs(c) ** 2)) + EPS)
    fade_db = np.repeat(20 * np.log10(fade + 1e-3), blk)[:n]
    inst_cnr = cnr + fade_db
    sig_rms = float(np.sqrt(np.mean(e ** 2))) + EPS
    noise = lfilter(pre_b, [1.0], rng.standard_normal(n)).astype(np.float32)   # flat hiss after de-emphasis
    noise *= sig_rms * 10 ** (-inst_cnr / 20) / (np.sqrt(np.mean(noise ** 2)) + EPS)
    y = e + noise
    # Rice clicks below FM threshold (~12 dB CNR)
    below = np.clip((12.0 - inst_cnr) / 12.0, 0.0, 1.0)
    rate = 40.0 * below / SR_LINK
    idx = np.where(rng.random(n) < rate)[0]
    for i in idx:
        L = int(rng.uniform(0.0003, 0.001) * SR_LINK) + 2
        y[i: i + L] += rng.choice([-1.0, 1.0]) * float(rng.uniform(0.3, 1.0)) * np.exp(-np.arange(min(L, n - i)) / (L / 3))
    y = lfilter([1.0], pre_b, y).astype(np.float32)                            # de-emphasis (exact inverse)
    return y, {"cnr_db": cnr, "doppler_hz": dop, "clicks": float(len(idx))}


@njit(cache=True)
def _rms_env(p, att, rel):
    env = np.empty_like(p)
    e = p[0]
    for i in range(len(p)):
        v = p[i]
        k = att if v > e else rel
        e = k * e + (1 - k) * v
        env[i] = e
    return env


# --------------------------------------------------------------------------
# Whole link
# --------------------------------------------------------------------------
def _compress(x: np.ndarray, rng: np.random.Generator, ratio_rng: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    ratio = float(rng.uniform(*ratio_rng))
    att, rel = np.exp(-1 / (0.003 * SR_LINK)), np.exp(-1 / (float(rng.uniform(0.08, 0.25)) * SR_LINK))
    p = x.astype(np.float64) ** 2
    env = _rms_env(p, att, rel)                 # RMS detector, attack / release
    lvl = 10 * np.log10(env + 1e-10)
    thr = 10 * np.log10(np.mean(p) + 1e-10) - 6.0
    g_db = np.where(lvl > thr, (thr - lvl) * (1 - 1 / ratio), 0.0)
    gain = (10 ** (g_db / 20)).astype(np.float32)
    return (x * gain).astype(np.float32), gain


def radio_link(x16: np.ndarray, speech16: np.ndarray, rng: np.random.Generator, params: dict[str, Any] | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """x16: boom signal (16 kHz, after the capsule channel). speech16: the target, used for the VOX
    onset and for the band-limited target. Returns (radio_input, target, loss_mask, info)."""
    p = {**DEFAULT, **(params or {})}
    n = len(x16)
    names = list(p["link_p"])
    w = np.array([float(p["link_p"][k]) for k in names])
    link = names[int(rng.choice(len(names), p=w / w.sum()))]
    tx_hi = 3000.0 if link == "fm" else 3400.0
    info: dict[str, Any] = {"link": link}
    x = resample_poly(x16, 1, 2).astype(np.float32)
    mask = np.ones(n, dtype=np.float32)
    # VOX / PTT front edge: the first syllable is lost, key-up click
    env = np.abs(speech16)
    on = np.where(env > 0.05 * (env.max() + EPS))[0]
    if len(on) and rng.random() < float(p["vox_p"]):
        a = int(on[0]) // 2
        m = int(rng.uniform(0.06, 0.25) * SR_LINK)
        x[a: a + m] = 0.0
        L = int(rng.uniform(0.002, 0.005) * SR_LINK)
        t = np.arange(L) / SR_LINK
        x[a: a + L] += (db_to_lin(float(rng.uniform(-20, -10))) * np.sin(2 * np.pi * rng.uniform(1000, 2000) * t)).astype(np.float32)
        mask[2 * a: 2 * (a + m)] = 0.0
        info["vox_ms"] = 1000.0 * m / SR_LINK
    x, comp_gain = _compress(x, rng, tuple(p["agc_ratio"]))
    x = bandpass(x, 300.0, tx_hi)
    if link == "fm":
        y, st = fm_link(x, rng, p)
    else:
        y, st = cvsd_link(x, 16 if link == "cvsd16" else 32, rng, p)
    info.update(st)
    # squelch tail at the end of the transmission (after the last speech)
    if len(on) and rng.random() < float(p["squelch_p"]) * (1.5 if link == "fm" else 0.6):
        b = min(len(y) - 1, int(on[-1]) // 2 + int(rng.uniform(0.05, 0.3) * SR_LINK))
        L = min(len(y) - b, int(rng.uniform(0.06, 0.2) * SR_LINK))
        if L > 0:
            hiss = rng.standard_normal(L).astype(np.float32) * db_to_lin(float(rng.uniform(-20, -8)))
            y[b: b + L] += hiss * np.hanning(L).astype(np.float32) ** 0.3
            info["squelch_ms"] = 1000.0 * L / SR_LINK
    y = bandpass(y, 300.0, 3400.0)
    out = resample_poly(y, 2, 1)[:n].astype(np.float32)
    # level: match the boom's loudness, then keep inside full scale
    out *= np.sqrt(np.mean(x16 ** 2) / (np.mean(out ** 2) + EPS))
    out = np.clip(out, -1.0, 1.0)
    # the TX compressor is a time-varying linear gain: shared with the target (same rule as the capsule AGC)
    g16 = np.repeat(comp_gain, 2)[:n]
    g16 = np.pad(g16, (0, n - len(g16)), mode="edge")
    target = link_filters(speech16 * g16, tx_hi)
    return out, target, mask, info
