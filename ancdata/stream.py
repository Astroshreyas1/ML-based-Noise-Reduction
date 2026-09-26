"""Two entry points, one code path.

``stream(cfg, split)``      infinite iterator of Pair -> training
``materialize(...)``        (in materialize.py) calls the same Chain and writes N pairs to disk

Training never touches the disk except for the source files themselves. Only
frozen evaluation sets are materialised.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .chain import Chain, Pair
from .config import load_config
from .registry import load_manifest
from .rir_gen import RirBank
from .sampler import AudioCache, Sampler


def build_chain(cfg: dict[str, Any] | str | Path, split: str, manifest_path: Path | None = None,
                bank_path: Path | None = None, cache_items: int = 2000) -> Chain:
    if not isinstance(cfg, dict):
        cfg = load_config(cfg)
    manifest = load_manifest(manifest_path)
    sampler = Sampler(manifest, cfg, split, cache=AudioCache(cache_items))
    bank = RirBank(bank_path)
    return Chain(cfg, sampler, bank)


def stream(cfg: dict[str, Any] | str | Path, split: str = "train", start_index: int = 0,
           chain: Chain | None = None) -> Iterator[Pair]:
    """Infinite, deterministic iterator. Example k is always generated from
    rng(seed, k), so any worker layout yields the same pairs."""
    chain = chain or build_chain(cfg, split)
    k = start_index
    while True:
        yield chain.generate(k)
        k += 1
