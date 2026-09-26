"""Configuration loading and distribution sampling.

The YAML config *is* the specification of the generation chain: every
probability, range and source weight lives here, not in code. This module
turns the YAML into a validated dict and provides `sample(spec, rng)` which
draws from a distribution spec such as
``{dist: triangular, low: -5, mode: 2, high: 20}``.

Paths are never stored in the config. Source piles are named, and the name is
resolved under ``ANC_DATA_ROOT`` by :mod:`ancdata.paths`.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SR = 16000
SEG_SECONDS = 4.0
SEG_SAMPLES = int(SR * SEG_SECONDS)
STFT_WIN = 512
STFT_HOP = 128  # 8 ms at 16 kHz — must equal the deployment frame

REQUIRED_TOP_LEVEL = ("seed", "sample_rate", "segment_seconds", "sources",
                      "lombard", "reverb", "mix_ambience", "impulsive",
                      "hard_negative", "adc", "codec", "channels", "target")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a chain config. Raises on anything malformed."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a mapping")
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if int(cfg["sample_rate"]) != SR:
        raise ValueError(f"{path}: sample_rate {cfg['sample_rate']} != {SR} (locked decision)")
    if float(cfg["segment_seconds"]) != SEG_SECONDS:
        raise ValueError(f"{path}: segment_seconds {cfg['segment_seconds']} != {SEG_SECONDS} (locked decision)")
    if cfg["target"] != "dry":
        raise ValueError(f"{path}: target must be 'dry' (matches dry-calm eval reference)")
    if list(cfg["channels"]) != ["boom", "ref"]:
        raise ValueError(f"{path}: channels must be [boom, ref]")
    for key in ("lombard", "reverb", "impulsive", "hard_negative", "adc", "codec"):
        p = cfg[key].get("probability")
        if p is None or not (0.0 <= float(p) <= 1.0):
            raise ValueError(f"{path}: {key}.probability must be in [0, 1]")
    src = cfg["sources"]
    for pile in ("speech", "ambience"):
        if pile not in src or not src[pile]:
            raise ValueError(f"{path}: sources.{pile} must list at least one source")
    cfg["_path"] = str(path)
    return cfg


def with_overrides(cfg: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return a deep copy of cfg with dotted-key overrides, e.g. ``codec.probability=1.0``."""
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = out
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def sample(spec: Any, rng: np.random.Generator) -> float:
    """Draw one value from a distribution spec.

    Accepted forms::

        3.5                                   -> constant
        {dist: uniform,     low, high}
        {dist: loguniform,  low, high}        -> log-uniform (positive bounds)
        {dist: triangular,  low, mode, high}
        {dist: normal,      mean, std}
        {dist: uniform_int, low, high}        -> inclusive integer range
    """
    if isinstance(spec, (int, float)):
        return float(spec)
    if not isinstance(spec, dict) or "dist" not in spec:
        raise ValueError(f"bad distribution spec: {spec!r}")
    d = spec["dist"]
    if d == "uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    if d == "loguniform":
        lo, hi = float(spec["low"]), float(spec["high"])
        if lo <= 0 or hi <= 0:
            raise ValueError(f"loguniform bounds must be positive: {spec}")
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if d == "triangular":
        return float(rng.triangular(spec["low"], spec["mode"], spec["high"]))
    if d == "normal":
        return float(rng.normal(spec["mean"], spec["std"]))
    if d == "uniform_int":
        return int(rng.integers(int(spec["low"]), int(spec["high"]) + 1))
    raise ValueError(f"unknown dist {d!r} in {spec}")


def child_rng(seed: int, index: int) -> np.random.Generator:
    """Deterministic per-example RNG. Same (seed, index) -> same stream, forever."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(index)]))


def source_list(entry: Any) -> list[dict[str, Any]]:
    """Normalise a `sources.<pile>` entry to a list of {name, weight, ...} dicts.

    Accepts ``[esc50, mad_vehicle]`` or
    ``[{type: file, name: esc50, weight: 0.5}, {type: parametric, generator: friedlander, weight: 0.5}]``.
    """
    out = []
    for item in entry:
        if isinstance(item, str):
            out.append({"type": "file", "name": item, "weight": 1.0})
        elif isinstance(item, dict):
            d = dict(item)
            d.setdefault("type", "file")
            d.setdefault("weight", 1.0)
            if d["type"] == "file" and "name" not in d:
                raise ValueError(f"file source without name: {item}")
            if d["type"] == "parametric" and "generator" not in d:
                raise ValueError(f"parametric source without generator: {item}")
            out.append(d)
        else:
            raise ValueError(f"bad source entry: {item!r}")
    return out
