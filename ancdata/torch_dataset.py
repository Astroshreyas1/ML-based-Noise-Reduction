"""Optional torch wrapper. Import only where torch is installed.

    from ancdata.torch_dataset import PairDataset, BattlefieldDataset, MaterializedDataset
    ds = PairDataset("configs/train.yaml", "train")                       # old 4 s chain (v2)
    ds = BattlefieldDataset("configs/battlefield.yaml", "train")           # v3 stream, infinite
    ds = MaterializedDataset("data/battlefield_v3/train")                  # v3 dump on disk, map-style
    dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=4)

Each worker generates a disjoint arithmetic progression of indices, so the
union over workers is exactly the sequence a single process would produce.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset, get_worker_info
except ImportError as e:  # pragma: no cover
    raise ImportError("torch not installed: pip install -r requirements-gpu.txt") from e

from .config import load_config
from .stream import build_chain


class PairDataset(IterableDataset):
    def __init__(self, cfg: dict[str, Any] | str | Path, split: str = "train",
                 mono: bool = False, with_events: bool = False, cache_items: int = 2000):
        self.cfg = cfg if isinstance(cfg, dict) else load_config(cfg)
        self.split = split
        self.mono = mono
        self.with_events = with_events
        self.cache_items = cache_items
        self._chain = None

    def _get_chain(self):
        if self._chain is None:
            self._chain = build_chain(self.cfg, self.split, cache_items=self.cache_items)
        return self._chain

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        chain = self._get_chain()
        k = wid
        while True:
            p = chain.generate(k)
            noisy = p.noisy[0] if self.mono else p.noisy
            item = {"noisy": torch.from_numpy(np.ascontiguousarray(noisy)),
                    "clean": torch.from_numpy(p.clean), "index": k}
            if self.with_events:
                mask = np.zeros(len(p.clean), dtype=np.float32)
                for e in p.events:
                    mask[e.start:e.end] = 1.0
                item["event_mask"] = torch.from_numpy(mask)
            yield item
            k += nw


class BattlefieldDataset(IterableDataset):
    """Infinite, worker-disjoint stream from the battlefield v3 chain
    (``configs/battlefield.yaml``): items are ``noisy`` (2, T) or (T,) if mono,
    ``clean`` (T,), ``index``, optional ``event_mask`` (1 inside gunfire / blast
    spans). Indices beyond ``max_examples`` keep streaming: the speech repeats,
    every scene is new."""

    def __init__(self, cfg: dict[str, Any] | str | Path = "configs/battlefield.yaml", split: str = "train",
                 mono: bool = False, with_events: bool = False, cache_mb: float | None = None):
        from .battlefield import load_battlefield_config
        self.cfg = cfg if isinstance(cfg, dict) else load_battlefield_config(cfg)
        self.split, self.mono, self.with_events, self.cache_mb = split, mono, with_events, cache_mb
        self._chain = None

    def _get_chain(self):
        if self._chain is None:
            from .battlefield import build_battlefield_chain
            self._chain = build_battlefield_chain(self.cfg, self.split, cache_mb=self.cache_mb)
        return self._chain

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        chain = self._get_chain()
        k = wid
        while True:
            p = chain.generate(k)
            noisy = p.noisy[0] if self.mono else p.noisy
            item = {"noisy": torch.from_numpy(np.ascontiguousarray(noisy)), "clean": torch.from_numpy(p.clean), "index": k}
            if self.with_events:
                mask = np.zeros(len(p.clean), dtype=np.float32)
                for e in p.events:
                    if e.category != "hard_negative":
                        mask[e.start:e.end] = 1.0
                item["event_mask"] = torch.from_numpy(mask)
            yield item
            k += nw


class MaterializedDataset(Dataset):
    """Map-style reader of a built split (``data/battlefield_v3/<split>``):
    noisy / clean FLAC or WAV + meta.jsonl. Random access, shuffle-friendly."""

    def __init__(self, root: str | Path, mono: bool = False, with_events: bool = False, with_meta: bool = False):
        import json
        import soundfile as sf  # noqa: F401
        self.root = Path(root)
        self.mono, self.with_events, self.with_meta = mono, with_events, with_meta
        self.meta = [json.loads(l) for l in (self.root / "meta.jsonl").open(encoding="utf-8") if l.strip()]
        self.ids = [m["id"] for m in self.meta]

    def __len__(self) -> int:
        return len(self.ids)

    def _find(self, sub: str, stem: str) -> Path:
        for ext in (".flac", ".wav"):
            p = self.root / sub / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"{self.root / sub / stem}.flac|.wav")

    def __getitem__(self, i: int):
        import soundfile as sf
        m = self.meta[i]
        noisy, _ = sf.read(str(self._find("noisy", m["id"])), dtype="float32", always_2d=True)
        clean, _ = sf.read(str(self._find("clean", m["id"])), dtype="float32")
        noisy = noisy.T
        item = {"noisy": torch.from_numpy(np.ascontiguousarray(noisy[0] if self.mono else noisy)),
                "clean": torch.from_numpy(np.ascontiguousarray(clean)), "index": int(m["id"])}
        if self.with_events:
            mask = np.zeros(len(clean), dtype=np.float32)
            for e in m.get("events", []):
                if e["category"] != "hard_negative":
                    mask[e["start"]:e["end"]] = 1.0
            item["event_mask"] = torch.from_numpy(mask)
        if self.with_meta:
            item["meta"] = m
        return item
