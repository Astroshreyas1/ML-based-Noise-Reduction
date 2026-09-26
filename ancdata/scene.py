"""Noise scene composition: several concurrent, time-varying sources.

A battlefield is not one 4-second clip at a fixed level. It is a rotor that
approaches, wind that gusts, an engine idling behind, and a burst of fire on
top. The chain therefore builds each example's noise as a *scene*:

    1-3 layers, each drawn from the ambience pool (real clip or the parametric
    rotor), each with its own relative level and its own time envelope:

        steady     constant (default for machinery)
        gust       slow random-walk gain, +-3..10 dB, 0.2-1 Hz   (wind, rain, fire, sea)
        approach   linear dB ramp of +6..15 dB over the clip + a small Doppler
                   drift (vehicles, aircraft, sirens)
        recede     the mirror of approach
        passby     Gaussian bump in dB, +6..12 dB, centred at a random time

    Relative levels: the dominant layer at 0 dB, the others 2-12 dB below.
    The chain then scales the whole scene to the requested SNR against the
    dry target, so SNR keeps its meaning (total noise vs speech).

Each layer is returned separately so the pipeline can write stems for
inspection (``materialize --stems``) and log per-layer metadata. Layers that
are capsule-local (wind) stay dry; the rest go through the room's noise path.

References: multi-source room simulation with 1-3 noise sources at levels
spread over ~10 dB (hearing-aid enhancement literature); moving-source
simulators (SonicSim, DynamicSound) for the Doppler / approach rationale.
Here the Doppler is a small resample-rate drift, not a full trajectory: the
model only needs to see pitch drift, not learn geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from .audio import EPS, tile_or_crop
from .config import SR, sample

# category -> envelope preference (probabilities over envelope types)
ENVELOPE_PRIORS: dict[str, dict[str, float]] = {
    "wind":            {"gust": 0.85, "steady": 0.15},
    "rain":            {"gust": 0.6, "steady": 0.4},
    "sea_waves":       {"gust": 0.7, "steady": 0.3},
    "crackling_fire":  {"gust": 0.6, "steady": 0.4},
    "thunderstorm":    {"gust": 0.7, "steady": 0.3},
    "helicopter":      {"steady": 0.35, "approach": 0.25, "recede": 0.2, "passby": 0.2},
    "synthetic_rotor": {"steady": 0.35, "approach": 0.25, "recede": 0.2, "passby": 0.2},
    "airplane":        {"steady": 0.3, "approach": 0.25, "recede": 0.2, "passby": 0.25},
    "engine":          {"steady": 0.5, "approach": 0.2, "recede": 0.15, "passby": 0.15},
    "train":           {"steady": 0.3, "approach": 0.25, "recede": 0.2, "passby": 0.25},
    "siren":           {"steady": 0.4, "approach": 0.2, "recede": 0.2, "passby": 0.2},
    "car_horn":        {"steady": 0.6, "passby": 0.4},
    "chainsaw":        {"steady": 0.6, "gust": 0.4},
}
DEFAULT_PRIOR = {"steady": 0.7, "gust": 0.15, "passby": 0.15}


@dataclass
class Layer:
    audio_boom: np.ndarray          # (T,) already enveloped, at its relative level (before scene->SNR scaling)
    audio_ref: np.ndarray           # (T,) same for the reference mic (may be an independent crop)
    category: str
    dry: bool
    envelope: str
    rel_db: float
    meta: dict[str, Any] = field(default_factory=dict)


def _pick(prior: dict[str, float], rng: np.random.Generator) -> str:
    keys = list(prior)
    p = np.array([prior[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _gust_envelope(n: int, rng: np.random.Generator, depth_db: float, rate_hz: float, sr: int = SR) -> np.ndarray:
    """Smoothed random walk in dB, zero-mean, +-depth."""
    n_ctrl = max(4, int(rate_hz * n / sr * 4))
    ctrl = np.cumsum(rng.standard_normal(n_ctrl))
    ctrl = (ctrl - ctrl.mean()) / (ctrl.std() + EPS)
    t = np.linspace(0, n_ctrl - 1, n)
    env_db = depth_db * np.interp(t, np.arange(n_ctrl), ctrl) / 2.0
    return (10.0 ** (env_db / 20.0)).astype(np.float32)


def _ramp_envelope(n: int, delta_db: float) -> np.ndarray:
    return (10.0 ** (np.linspace(-delta_db / 2, delta_db / 2, n) / 20.0)).astype(np.float32)


def _passby_envelope(n: int, rng: np.random.Generator, peak_db: float) -> np.ndarray:
    c = rng.uniform(0.2, 0.8) * n
    w = rng.uniform(0.12, 0.3) * n
    env_db = peak_db * np.exp(-0.5 * ((np.arange(n) - c) / w) ** 2) - peak_db / 2
    return (10.0 ** (env_db / 20.0)).astype(np.float32)


def _doppler_drift(x: np.ndarray, rng: np.random.Generator, max_pct: float = 2.0) -> np.ndarray:
    """Approximate pitch drift of a moving source: resample by a factor that
    moves linearly from 1+d to 1-d (or the reverse) across the clip. Done in
    two halves with polyphase resampling; good enough for a few percent."""
    d = rng.uniform(0.5, max_pct) / 100.0
    sign = 1.0 if rng.random() < 0.5 else -1.0
    n = len(x)
    halves = []
    for k, f in enumerate((1.0 + sign * d, 1.0 - sign * d)):
        seg = x[k * n // 2:(k + 1) * n // 2]
        up = int(round(1000 * f))
        halves.append(resample_poly(seg, up, 1000).astype(np.float32))
    y = np.concatenate(halves)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n]


def apply_envelope(x: np.ndarray, kind: str, rng: np.random.Generator, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    n = len(x)
    if kind == "steady":
        return x, {}
    if kind == "gust":
        depth = sample(cfg.get("gust_depth_db", {"dist": "uniform", "low": 3, "high": 10}), rng)
        rate = sample(cfg.get("gust_rate_hz", {"dist": "uniform", "low": 0.2, "high": 1.0}), rng)
        return x * _gust_envelope(n, rng, depth, rate), {"depth_db": depth, "rate_hz": rate}
    if kind in ("approach", "recede"):
        delta = sample(cfg.get("ramp_db", {"dist": "uniform", "low": 6, "high": 15}), rng)
        delta = delta if kind == "approach" else -delta
        y = _doppler_drift(x, rng, float(cfg.get("doppler_max_pct", 2.0)))
        return y * _ramp_envelope(n, delta), {"ramp_db": delta}
    if kind == "passby":
        peak = sample(cfg.get("passby_peak_db", {"dist": "uniform", "low": 6, "high": 12}), rng)
        y = _doppler_drift(x, rng, float(cfg.get("doppler_max_pct", 2.0)))
        return y * _passby_envelope(n, rng, peak), {"peak_db": peak}
    raise ValueError(f"unknown envelope {kind!r}")


def compose_scene(draw_ambience, rng: np.random.Generator, cfg: dict[str, Any], n: int,
                  dry_categories: set[str]) -> list[Layer]:
    """draw_ambience(rng) -> (audio_full, category, meta). Returns 1-3 layers
    at their relative levels; the caller scales the sum to the target SNR."""
    n_layers = int(sample(cfg.get("layers", {"dist": "uniform_int", "low": 1, "high": 3}), rng))
    layers: list[Layer] = []
    seen: set[str] = set()
    for k in range(n_layers):
        audio_full, cat, meta = draw_ambience(rng)
        if cat in seen and n_layers > 1 and k > 0:
            audio_full, cat, meta = draw_ambience(rng)  # one retry for variety
        seen.add(cat)
        boom = tile_or_crop(audio_full, n, rng).astype(np.float32)
        dry = cat in dry_categories
        # dry (capsule-local) layers are uncorrelated between the two mics
        ref = tile_or_crop(audio_full, n, rng).astype(np.float32) if dry else boom.copy()
        kind = _pick(ENVELOPE_PRIORS.get(cat, DEFAULT_PRIOR), rng)
        boom, env_meta = apply_envelope(boom, kind, rng, cfg)
        if dry:
            ref, _ = apply_envelope(ref, kind, rng, cfg)
        else:
            ref = boom.copy()
        # level: normalise each layer to unit RMS, then the relative level
        g = 1.0 / (np.sqrt(np.mean(boom ** 2)) + EPS)
        rel_db = 0.0 if k == 0 else -sample(cfg.get("secondary_below_db", {"dist": "uniform", "low": 2, "high": 12}), rng)
        g *= 10.0 ** (rel_db / 20.0)
        layers.append(Layer(boom * g, ref * g * (float(rng.uniform(1.0, 1.5)) if dry else 1.0),
                            cat, dry, kind, float(rel_db), {**meta, **env_meta}))
    return layers
