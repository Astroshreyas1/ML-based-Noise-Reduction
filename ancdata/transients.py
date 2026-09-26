"""Transient events for the battlefield chain: gunfire bursts and blasts.

Three sources, mixed on purpose (docs/NOISE_LAYERING.md section 3, 6b):

    physics   Friedlander muzzle blast + N-wave crack + ground bounce + air
              absorption (``physics.synth_blast``); clean attacks, labelled
              standoff; the training majority.
    field     real range recordings (Zenodo edge-collected set): one shot
              cropped around its annotated time, repeated with gain jitter
              into a burst.
    mad       YouTube firefight bursts (MAD `gunshot`): lossy, but they carry
              the environment (echoes, multiple weapons) that no model gives.

Every burst gets a *landscape tail* -- a decaying, low-passed noise tail
convolved onto the event. Gunfire heard outdoors is never a bare impulse
(tree lines, buildings and ground scatter answer every shot); without the
tail the shot reads as "pasted on", which is what the first listening trial
found. Distant bursts are additionally low-passed (ISO 9613-1 direction).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import fftconvolve, lfilter

from .audio import EPS, load_mono, trim_silence
from .config import SR
from .physics import synth_blast


def lowpass1(x: np.ndarray, fc: float, sr: int = SR) -> np.ndarray:
    k = np.exp(-2 * np.pi * fc / sr)
    return lfilter([1 - k], [1, -k], x).astype(np.float32)


def outdoor_tail(x: np.ndarray, tau_s: float, level_db: float, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    n = int(3 * tau_s * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n).astype(np.float32) * np.exp(-t / tau_s)
    tail = lowpass1(tail, 2500.0, sr)
    tail *= 10 ** (level_db / 20) / (np.abs(tail).max() + EPS)
    ir = np.zeros(n + 1, dtype=np.float32)
    ir[0] = 1.0
    ir[1:] += tail
    return fftconvolve(x, ir)[: len(x) + n].astype(np.float32)


def load_gunshot(path: str, t_shot: float, sr: int = SR, seconds: float = 0.7) -> np.ndarray:
    """One annotated shot from a range recording, 10 ms pre-roll, faded out."""
    x = load_mono(path, sr)
    a = max(0, int((t_shot - 0.01) * sr))
    b = min(len(x), a + int(seconds * sr))
    seg = x[a:b].copy()
    fade = min(int(0.05 * sr), len(seg))
    if fade:
        seg[-fade:] *= np.linspace(1, 0, fade)
    return seg


def render_burst(b: dict[str, Any], rng: np.random.Generator, sr: int = SR) -> tuple[np.ndarray, dict[str, float]]:
    """b: source (physics|field|mad), n_rounds, spacing_s, distant, tail_tau_s,
    tail_db, file / t_shot for field, file for mad. Returns (peak-normalised
    waveform, physics params if any)."""
    params: dict[str, float] = {}
    if b["source"] == "physics":
        spec = dict(standoff_m={"dist": "loguniform", "low": 40.0, "high": 400.0} if b.get("distant") else
                    {"dist": "loguniform", "low": 5.0, "high": 80.0},
                    T_ms={"dist": "uniform", "low": 0.4, "high": 2.0},
                    burst={"dist": "uniform_int", "low": b["n_rounds"], "high": b["n_rounds"]},
                    cyclic_ms={"dist": "uniform", "low": b["spacing_s"] * 1000, "high": b["spacing_s"] * 1000},
                    supersonic_p=0.7)
        wave, params = synth_blast(rng, spec, sr)
    elif b["source"] == "mad":
        one = trim_silence(load_mono(b["file"], sr))
        pk = int(np.argmax(np.abs(one)))
        a0 = max(0, pk - int(0.02 * sr))
        wave = one[a0: a0 + int(1.5 * sr)].copy()
        fade = min(int(0.1 * sr), len(wave))
        wave[-fade:] *= np.linspace(1, 0, fade)
    else:
        one = load_gunshot(b["file"], float(b["t_shot"]), sr)
        step = int(b["spacing_s"] * sr)
        wave = np.zeros(step * (b["n_rounds"] - 1) + len(one) + step, dtype=np.float32)
        for r in range(b["n_rounds"]):
            s0 = max(0, r * step + (int(rng.integers(-step // 20, step // 20 + 1)) if r else 0))
            wave[s0: s0 + len(one)] += one * float(rng.uniform(0.7, 1.0))
    if b.get("distant"):
        wave = lowpass1(lowpass1(wave, 1800.0, sr), 1800.0, sr)
    wave = outdoor_tail(wave, b["tail_tau_s"], b["tail_db"], rng, sr)
    return wave / (np.abs(wave).max() + EPS), params


def render_blast(bl: dict[str, Any], rng: np.random.Generator, sr: int = SR) -> tuple[np.ndarray, dict[str, float]]:
    """A shelling / explosion event: long-positive-phase physics blast with a
    long tail, or a MAD shelling clip cropped at its loudest point."""
    params: dict[str, float] = {}
    if bl["source"] == "physics":
        wave, params = synth_blast(rng, dict(standoff_m={"dist": "loguniform", "low": 100.0, "high": 600.0},
                                             T_ms={"dist": "uniform", "low": 2.5, "high": 4.0}, supersonic_p=0.0), sr)
        wave = outdoor_tail(wave, float(rng.uniform(0.3, 0.7)), -float(rng.uniform(4, 10)), rng, sr)
    else:
        x = trim_silence(load_mono(bl["file"], sr))
        pk = int(np.argmax(np.abs(x)))
        a0 = max(0, pk - int(0.03 * sr))
        wave = x[a0: a0 + int(1.5 * sr)].copy()
        fade = int(0.2 * sr)
        if len(wave) > fade:
            wave[-fade:] *= np.linspace(1, 0, fade)
    return wave / (np.abs(wave).max() + EPS), params


def audible_span(x: np.ndarray, start: int, rel_db: float = -40.0, sr: int = SR) -> tuple[int, int]:
    """[start, end) of the part of an event above rel_db of its own peak (the
    tail below that is inaudible under the scene and would inflate the label)."""
    env = np.abs(x)
    pk = float(env.max()) + EPS
    idx = np.where(env >= pk * 10 ** (rel_db / 20))[0]
    if len(idx) == 0:
        return start, start + len(x)
    return start + int(idx[0]), start + int(idx[-1]) + 1
