"""Physics-based noise generators.

Two parametric sources live here:

* :func:`synth_blast` — ballistic / blast impulse. Friedlander overpressure
  wave (muzzle blast), optional supersonic N-wave crack arriving first,
  ground-bounce reflection, distance-dependent air absorption. Every event
  returns its physical parameters (standoff, positive-phase duration, height)
  so training examples carry labels no scraped clip can provide.

* :func:`synth_rotor` — continuous rotorcraft / turbine model. Blade-passage
  harmonics with slow amplitude modulation, high-Q turbine whine, pink
  fuselage turbulence. Used as a *minority* ambience source alongside real
  recordings; a network learns a pure synthetic rotor in minutes and then
  fails on real blade slap and doppler, so real clips carry most weight.

The blast model is first-order: no atmospheric turbulence, no barrel geometry,
no weapon-specific signature. Say so in the writeup.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

from .config import SR, sample

C_AIR = 343.0  # m/s


def friedlander(peak: float, T_ms: float, sr: int = SR, n_tau: float = 10.0) -> np.ndarray:
    """Ideal free-field blast wave p(t) = P (1 - t/T) exp(-t/T).

    Instantaneous rise at t=0, zero crossing at t=T, negative phase after.
    The negative phase is physical; keep it.
    """
    T = T_ms / 1000.0
    t = np.arange(0.0, n_tau * T, 1.0 / sr)
    return (peak * (1.0 - t / T) * np.exp(-t / T)).astype(np.float32)


def ballistic_crack(sr: int = SR, dur_ms: float = 1.2, peak: float = 1.0) -> np.ndarray:
    """Supersonic projectile N-wave. Precedes the muzzle blast for rifle shots."""
    n = max(4, int(sr * dur_ms / 1000.0))
    t = np.linspace(-1.0, 1.0, n)
    return (peak * -t * np.exp(-3.0 * t ** 2)).astype(np.float32)


def ground_reflection(x: np.ndarray, standoff_m: float, height_m: float,
                      sr: int = SR, coeff: float = 0.7) -> np.ndarray:
    """Delayed, attenuated copy from the ground bounce."""
    direct = float(standoff_m)
    reflected = float(np.hypot(standoff_m, 2.0 * height_m))
    delay = int(max(1, round((reflected - direct) / C_AIR * sr)))
    y = np.zeros(len(x) + delay, dtype=np.float32)
    y[: len(x)] += x
    y[delay:] += coeff * x * (direct / reflected)
    return y


def air_absorption(x: np.ndarray, standoff_m: float, sr: int = SR) -> np.ndarray:
    """Distance-dependent high-frequency loss: why a far shot is a boom and a
    near one a crack. Second-order low-pass whose cutoff falls with distance."""
    cutoff = float(np.clip(18000.0 * np.exp(-standoff_m / 150.0), 300.0, 0.45 * sr))
    sos = butter(2, cutoff, btype="low", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def synth_blast(rng: np.random.Generator, params: dict[str, Any] | None = None,
                sr: int = SR) -> tuple[np.ndarray, dict[str, float]]:
    """One parameterised blast event, peak-normalised to 1.0.

    `params` may override the sampling specs (see configs/train.yaml
    ``impulsive.sources[parametric].params``).
    """
    p = params or {}
    standoff = sample(p.get("standoff_m", {"dist": "loguniform", "low": 10.0, "high": 500.0}), rng)
    T_ms = sample(p.get("T_ms", {"dist": "uniform", "low": 0.3, "high": 4.0}), rng)
    height = sample(p.get("height_m", {"dist": "uniform", "low": 1.0, "high": 2.0}), rng)
    supersonic_p = float(p.get("supersonic_p", 0.6))
    burst = p.get("burst", {"dist": "uniform_int", "low": 1, "high": 1})
    n_rounds = int(sample(burst, rng))
    cyclic_ms = sample(p.get("cyclic_ms", {"dist": "uniform", "low": 80.0, "high": 140.0}), rng)

    wave = friedlander(1.0, T_ms, sr)
    supersonic = rng.random() < supersonic_p
    if supersonic:
        crack = ballistic_crack(sr, peak=float(rng.uniform(0.5, 1.2)))
        gap = int(sr * rng.uniform(0.002, 0.030))  # crack arrives first
        combined = np.zeros(gap + max(len(crack), len(wave)), dtype=np.float32)
        combined[: len(crack)] += crack
        combined[gap: gap + len(wave)] += wave
        wave = combined

    wave = ground_reflection(wave, standoff, height, sr)
    wave = air_absorption(wave, standoff, sr)

    if n_rounds > 1:
        step = int(sr * cyclic_ms / 1000.0)
        jit = max(1, step // 20)
        out = np.zeros(step * (n_rounds - 1) + jit + len(wave), dtype=np.float32)
        for k in range(n_rounds):
            jitter = int(rng.integers(-jit, jit + 1)) if k else 0
            s = max(0, k * step + jitter)
            out[s: s + len(wave)] += wave * float(rng.uniform(0.85, 1.0))
        wave = out

    wave = wave / (np.abs(wave).max() + 1e-9)
    meta = {
        "standoff_m": float(standoff),
        "T_ms": float(T_ms),
        "height_m": float(height),
        "supersonic": float(supersonic),
        "n_rounds": float(n_rounds),
    }
    return wave.astype(np.float32), meta


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise via spectral shaping of white noise."""
    white = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if n > 1 else 1.0
    spec = spec / np.sqrt(f)
    out = np.fft.irfft(spec, n=n)
    return (out / (np.abs(out).max() + 1e-9)).astype(np.float32)


def synth_rotor(rng: np.random.Generator, n: int, sr: int = SR,
                params: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, float]]:
    """Continuous rotor + turbine + turbulence ambience, peak-normalised.

    f_bpf = RPM * N_blades / 60 (18-25 Hz for a main rotor). Harmonics carry
    slow AM to mimic blade slap; turbine whine is a set of high-Q sinusoids;
    turbulence is pink noise. Parameters returned as labels.
    """
    p = params or {}
    t = np.arange(n) / sr
    rpm = sample(p.get("rpm", {"dist": "uniform", "low": 250.0, "high": 400.0}), rng)
    n_blades = int(sample(p.get("n_blades", {"dist": "uniform_int", "low": 2, "high": 5}), rng))
    f_bpf = rpm * n_blades / 60.0
    n_harm = int(p.get("n_harmonics", 12))

    rotor = np.zeros(n, dtype=np.float32)
    am = 1.0 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.3, 1.5) * t + rng.uniform(0, 2 * np.pi))
    for k in range(1, n_harm + 1):
        amp = 1.0 / (k ** float(rng.uniform(0.8, 1.4)))
        rotor += (amp * np.sin(2 * np.pi * f_bpf * k * t + rng.uniform(0, 2 * np.pi))).astype(np.float32)
    rotor *= am.astype(np.float32)

    # turbine whine: harmonics with slow random-walk FM (+-1 %) and AM so the lines are not
    # dead straight, which a real turbine never is
    whine = np.zeros(n, dtype=np.float32)
    base = sample(p.get("turbine_hz", {"dist": "uniform", "low": 900.0, "high": 1600.0}), rng)
    walk = np.cumsum(rng.standard_normal(64))
    walk = np.interp(np.linspace(0, 63, n), np.arange(64), (walk - walk.mean()) / (walk.std() + 1e-9))
    fm = 1.0 + 0.01 * walk
    am = 1.0 + 0.3 * np.interp(np.linspace(0, 63, n), np.arange(64), rng.uniform(-1, 1, 64))
    for k, amp in ((1, 1.0), (2, 0.5), (3, 0.25)):
        whine += (amp * np.sin(2 * np.pi * np.cumsum(base * k * fm) / sr)).astype(np.float32)
    whine *= am.astype(np.float32)

    # blade slap: short broadband bursts locked to the blade passage, random strength
    slap = np.zeros(n, dtype=np.float32)
    period = int(sr / f_bpf)
    burst_len = int(sr * 0.004)
    win = np.hanning(burst_len).astype(np.float32)
    for s0 in range(int(rng.integers(0, period)), n - burst_len, period):
        slap[s0:s0 + burst_len] += win * rng.standard_normal(burst_len).astype(np.float32) * float(rng.uniform(0.2, 1.0))

    turb = _pink(n, rng)

    w_rotor = float(rng.uniform(0.5, 1.0))
    w_whine = float(rng.uniform(0.02, 0.15))
    w_slap = float(rng.uniform(0.1, 0.5))
    w_turb = float(rng.uniform(0.2, 0.6))
    out = w_rotor * rotor / (np.abs(rotor).max() + 1e-9)         + w_whine * whine / (np.abs(whine).max() + 1e-9)         + w_slap * slap / (np.abs(slap).max() + 1e-9)         + w_turb * turb
    out = out / (np.abs(out).max() + 1e-9)
    meta = {"rpm": float(rpm), "n_blades": float(n_blades), "f_bpf_hz": float(f_bpf),
            "turbine_hz": float(base)}
    return out.astype(np.float32), meta
