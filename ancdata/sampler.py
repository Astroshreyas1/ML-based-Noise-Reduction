"""Weighted source drawing that honours the registry's split columns.

The sampler is the only code that turns a config ``sources`` entry into a
manifest row. It filters on ``split``, never draws ``eval_only`` rows for
training, and applies the holdout policy (exclude the held-out noise group
during training; draw only from it for the generalisation eval).
"""
from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from .audio import load_mono
from .config import source_list
from .paths import from_relative


class AudioCache:
    """Bounded LRU of decoded 16 kHz mono arrays keyed by relative path."""

    def __init__(self, max_items: int = 2000):
        self.max_items = max_items
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, rel: str) -> np.ndarray:
        if rel in self._d:
            self._d.move_to_end(rel)
            return self._d[rel]
        audio = load_mono(from_relative(rel))
        self._d[rel] = audio
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)
        return audio


class Sampler:
    def __init__(self, manifest: pd.DataFrame, cfg: dict[str, Any], split: str,
                 cache: AudioCache | None = None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"bad split {split!r}")
        self.split = split
        self.cfg = cfg
        self.cache = cache or AudioCache()
        self.holdout_mode = cfg.get("holdout", {}).get("mode", "exclude")
        self.holdout_group = cfg.get("holdout", {}).get("group", "gen")
        self._pools: dict[str, list[tuple[dict[str, Any], pd.DataFrame]]] = {}
        self._build_pools(manifest)

    def _filter(self, df: pd.DataFrame, sclass: str, pile: str,
                categories: list[str] | None = None) -> pd.DataFrame:
        m = (df.sclass == sclass) & (df.pile == pile)
        # held-out noise classes never enter training, so the generalisation eval may use
        # every fold of them; everything else is split-filtered
        if not (self.holdout_mode == "only" and sclass in ("ambience", "impulsive")):
            m &= df.split == self.split
        if categories:
            m &= df.category.isin(categories)
        if self.split == "train":
            m &= ~df.eval_only.astype(bool)
        sub = df[m]
        if sclass in ("ambience", "impulsive"):
            is_ho = sub.holdout_group == self.holdout_group
            if self.holdout_mode == "exclude":
                sub = sub[~is_ho]
            elif self.holdout_mode == "only":
                sub = sub[is_ho]
            elif self.holdout_mode != "include":
                raise ValueError(f"bad holdout.mode {self.holdout_mode!r}")
        return sub.reset_index(drop=True)

    def _build_pools(self, df: pd.DataFrame) -> None:
        for sclass in ("speech", "ambience", "impulsive", "hard_negative"):
            entries = source_list(self.cfg["sources"].get(sclass, []))
            pool = []
            for e in entries:
                if e["type"] == "parametric":
                    pool.append((e, pd.DataFrame()))
                    continue
                sub = self._filter(df, sclass, e["name"], e.get("categories"))
                if len(sub) == 0:
                    if float(e["weight"]) > 0:
                        # absent pile (not downloaded / not screened yet): warn, renormalise the rest
                        warnings.warn(f"source {e['name']!r} ({sclass}) has no rows for split={self.split} "
                                      f"holdout={self.holdout_mode}; drawing from the remaining sources",
                                      stacklevel=2)
                    continue
                pool.append((e, sub))
            self._pools[sclass] = pool
        for sclass in ("speech", "ambience"):
            if not self._pools.get(sclass):
                raise RuntimeError(f"no {sclass} sources with rows for split {self.split}; run `ancdata registry`")

    def has(self, sclass: str) -> bool:
        return bool(self._pools.get(sclass))

    def draw_entry(self, sclass: str, rng: np.random.Generator) -> tuple[dict[str, Any], pd.DataFrame]:
        pool = self._pools.get(sclass, [])
        if not pool:
            raise RuntimeError(f"no {sclass} sources available")
        w = np.array([float(e["weight"]) for e, _ in pool], dtype=np.float64)
        if w.sum() <= 0:
            raise RuntimeError(f"all {sclass} source weights are zero")
        i = int(rng.choice(len(pool), p=w / w.sum()))
        return pool[i]

    def draw_row(self, sclass: str, rng: np.random.Generator) -> tuple[dict[str, Any], pd.Series | None]:
        """Returns (entry, row). row is None for parametric entries."""
        entry, sub = self.draw_entry(sclass, rng)
        if entry["type"] == "parametric":
            return entry, None
        return entry, sub.iloc[int(rng.integers(0, len(sub)))]

    def audio(self, row: pd.Series) -> np.ndarray:
        return self.cache.get(str(row.path))

    def counts(self) -> dict[str, int]:
        return {k: int(sum(len(df) for e, df in v if e["type"] == "file")) for k, v in self._pools.items()}
