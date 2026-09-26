"""Data-root resolution. The only place that knows where data lives.

Everything on disk is addressed relative to ``ANC_DATA_ROOT`` (env var,
default ``./data`` under the current working directory). Moving the project
to another machine means copying that directory and setting one variable.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "ANC_DATA_ROOT"

# Pile name -> path under <root>/sources. Names are what configs refer to.
PILES: dict[str, tuple[str, str]] = {
    # name: (relative dir, sclass)
    "librispeech":     ("speech/librispeech",      "speech"),
    "lombardgrid":     ("speech/lombardgrid",      "speech"),        # real Lombard speech; speaker-disjoint split
    "local_speakers":  ("speech/local",            "speech"),        # eval only
    "esc50":           ("noise/esc50",             "ambience"),
    "mad_vehicle":     ("noise/mad/vehicle",       "ambience"),
    "mad_helicopter":  ("noise/mad/helicopter",    "ambience"),
    "mad_fighter":     ("noise/mad/fighter",       "ambience"),
    "musan_noise":     ("noise/musan",             "ambience"),
    "esc50_gunshot":   ("impulsive/esc50_gunshot", "impulsive"),     # eval only
    "field_gunshots":  ("impulsive/field",         "impulsive"),     # Zenodo 7004819 range set; eval only
    "mad_shelling":    ("impulsive/mad/shelling",  "impulsive"),     # eval only / holdout
    "mad_gunshot":     ("impulsive/mad/gunshot",   "impulsive"),     # eval only
    "local_transients": ("hard_negative/local",    "hard_negative"),
    "mad_footsteps":   ("hard_negative/mad/footsteps", "hard_negative"),
    "realcheck":       ("realcheck",               "realcheck"),     # eval only, no reference
    "fixture_speech":  ("speech/fixture",          "speech"),        # smoke test
    "fixture_noise":   ("noise/fixture",           "ambience"),      # smoke test
    "fixture_hardneg": ("hard_negative/fixture",   "hard_negative"), # smoke test
    "fixture_gunshot": ("impulsive/fixture",       "impulsive"),     # smoke test
}

EVAL_ONLY_PILES = frozenset({
    "local_speakers", "esc50_gunshot", "field_gunshots", "mad_shelling", "mad_gunshot", "realcheck",
})


def data_root() -> Path:
    return Path(os.environ.get(ENV_VAR, "data")).expanduser().resolve()


def sources_dir() -> Path:
    return data_root() / "sources"


def pile_dir(name: str) -> Path:
    if name not in PILES:
        raise KeyError(f"unknown pile {name!r}; known: {sorted(PILES)}")
    return sources_dir() / PILES[name][0]


def pile_sclass(name: str) -> str:
    return PILES[name][1]


def manifest_path() -> Path:
    return data_root() / "manifest.parquet"


def rir_bank_path() -> Path:
    return data_root() / "sources" / "rir" / "bank.npz"


def eval_dir(name: str) -> Path:
    return data_root() / "eval" / name


def quarantine_dir() -> Path:
    return data_root() / "quarantine"


def to_relative(p: Path) -> str:
    """Store manifest paths relative to the data root, POSIX-style, so the
    manifest survives a move between Windows and Linux."""
    return Path(p).resolve().relative_to(data_root()).as_posix()


def from_relative(rel: str) -> Path:
    return data_root() / Path(rel)
