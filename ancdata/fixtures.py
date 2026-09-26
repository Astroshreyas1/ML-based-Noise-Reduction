"""Synthetic fixture sources so the whole pipeline can be exercised with no
downloads (smoke test, CI, first run on a new machine).

Speech stand-in: glottal pulse train with a wandering F0 through two or three
formant resonators, gated into syllables with pauses. It is not speech, but it
has the properties the chain relies on: harmonic, voiced/unvoiced structure,
pauses, a defined speaker identity (F0 range + formant set).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import lfilter, butter, sosfilt

from .audio import write_wav
from .config import SR
from .paths import pile_dir
from .physics import synth_blast, synth_rotor


def _formant(x: np.ndarray, f: float, bw: float, sr: int) -> np.ndarray:
    r = np.exp(-np.pi * bw / sr)
    a = [1.0, -2.0 * r * np.cos(2 * np.pi * f / sr), r * r]
    return lfilter([1.0 - r], a, x)


def fake_speech(rng: np.random.Generator, seconds: float, speaker: int, sr: int = SR) -> np.ndarray:
    n = int(sr * seconds)
    t = np.arange(n) / sr
    f0_base = 100.0 + 15.0 * speaker
    f0 = f0_base * (1.0 + 0.15 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)))
    phase = np.cumsum(f0) / sr
    pulses = (np.diff(np.floor(phase), prepend=0) > 0).astype(np.float32)
    formants = [(500 + 40 * speaker, 80), (1500 + 60 * speaker, 120), (2500 + 30 * speaker, 160)]
    y = np.zeros(n)
    for f, bw in formants:
        y += _formant(pulses, f, bw, sr)
    # syllable gating: 4 Hz envelope, plus sentence pauses
    env = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t)) ** 2
    pause = np.ones(n)
    for s in np.arange(1.2, seconds, 1.6):
        i0, i1 = int(s * sr), int((s + 0.35) * sr)
        pause[i0:i1] = 0.0
    y = y * env * pause
    # a little breath noise for "consonants"
    y += 0.02 * rng.standard_normal(n) * (1 - env) * pause
    y = y / (np.abs(y).max() + 1e-9) * 0.5
    return y.astype(np.float32)


def fake_noise(rng: np.random.Generator, kind: str, seconds: float, sr: int = SR) -> np.ndarray:
    n = int(sr * seconds)
    if kind == "helicopter":
        x, _ = synth_rotor(rng, n, sr)
    elif kind == "siren":
        t = np.arange(n) / sr
        f = 700 + 300 * np.sin(2 * np.pi * 0.5 * t)
        x = np.sin(2 * np.pi * np.cumsum(f) / sr) + 0.1 * rng.standard_normal(n)
    elif kind == "wind":
        sos = butter(2, 400, btype="low", fs=sr, output="sos")
        x = sosfilt(sos, rng.standard_normal(n)) * (1 + 0.8 * np.sin(2 * np.pi * 0.3 * np.arange(n) / sr))
    else:
        raise ValueError(kind)
    x = x / (np.abs(x).max() + 1e-9) * 0.7
    return x.astype(np.float32)


def fake_hardneg(rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Clap-like burst: 5 ms noise attack, 60 ms decay."""
    n = int(sr * 0.25)
    t = np.arange(n) / sr
    env = np.exp(-t / 0.03)
    x = rng.standard_normal(n) * env
    sos = butter(2, 1500, btype="high", fs=sr, output="sos")
    x = sosfilt(sos, x)
    return (x / (np.abs(x).max() + 1e-9) * 0.9).astype(np.float32)


def make_fixtures(seed: int = 0, n_speakers: int = 20, files_per_speaker: int = 2,
                  noise_per_kind: int = 10, n_hardneg: int = 8, n_gunshot: int = 6,
                  verbose: bool = True) -> None:
    rng = np.random.default_rng(seed)
    sp = pile_dir("fixture_speech")
    for s in range(n_speakers):
        for k in range(files_per_speaker):
            write_wav(sp / f"spk{s:02d}-{k:03d}.wav", fake_speech(rng, 6.0, s))
    nz = pile_dir("fixture_noise")
    for kind in ("helicopter", "siren", "wind"):
        for k in range(noise_per_kind):
            write_wav(nz / f"{kind}_{k:03d}.wav", fake_noise(rng, kind, 5.0))
    hn = pile_dir("fixture_hardneg")
    for k in range(n_hardneg):
        write_wav(hn / f"clap_{k:03d}.wav", fake_hardneg(rng))
    gs = pile_dir("fixture_gunshot")
    for k in range(n_gunshot):
        x, _ = synth_blast(rng)
        tail = np.exp(-np.arange(int(sr_tail := SR * 0.3)) / (SR * 0.05)) * rng.standard_normal(int(sr_tail)) * 0.05
        y = np.zeros(int(sr_tail), dtype=np.float32)
        y[: len(x)] += x
        y += tail.astype(np.float32)
        write_wav(gs / f"gunshot_{k:03d}.wav", y / (np.abs(y).max() + 1e-9) * 0.95)
    if verbose:
        print(f"fixtures written under {sp.parent.parent}")
