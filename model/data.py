"""Training / validation sets over the built battlefield_v3 splits.

Each item: noisy (2, N) float32 [boom, ref], clean (N,), frame_class (N/64,) long
(0 none / hard negative, 1 gunfire, 2 blast), ev_window (N,) float (1 inside
[-50 ms, +250 ms] around every gunfire / blast span), plus the pair id. Train
items are random crops (default 4 s) of the 6 s pairs; val items are the full
6 s. Event spans come from meta.jsonl and are shifted with the crop.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SR = 16000
HOP = 64
CLASS_OF = {"gunfire_auto": 1, "gunfire_semi": 1, "gunfire_distant": 1, "blast": 2, "hard_negative": 0}
PRE_MS, POST_MS = 50.0, 250.0


class PairSet(Dataset):
    def __init__(self, root: str | Path, crop_seconds: float | None = 4.0, limit: int | None = None, seed: int = 0):
        self.root = Path(root)
        self.meta = [json.loads(l) for l in (self.root / "meta.jsonl").open(encoding="utf-8") if l.strip()]
        if limit:
            rng = random.Random(seed)
            self.meta = rng.sample(self.meta, min(limit, len(self.meta)))
        self.crop = int(crop_seconds * SR) if crop_seconds else None

    def __len__(self) -> int:
        return len(self.meta)

    def _find(self, sub: str, stem: str) -> Path:
        for ext in (".flac", ".wav"):
            p = self.root / sub / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(stem)

    def __getitem__(self, i: int):
        m = self.meta[i]
        noisy, _ = sf.read(str(self._find("noisy", m["id"])), dtype="float32", always_2d=True)
        clean, _ = sf.read(str(self._find("clean", m["id"])), dtype="float32")
        noisy = noisy.T                                            # (2, N)
        n = noisy.shape[1]
        off = 0
        if self.crop and n > self.crop:
            off = random.randint(0, n - self.crop)
            noisy, clean = noisy[:, off: off + self.crop], clean[off: off + self.crop]
            n = self.crop
        t = n // HOP
        frame_class = np.zeros(t, dtype=np.int64)
        ev_window = np.zeros(n, dtype=np.float32)
        pre, post = int(PRE_MS * SR / 1000), int(POST_MS * SR / 1000)
        for e in m.get("events", []):
            c = CLASS_OF.get(e["category"], 0)
            if c == 0:
                continue
            a, b = e["start"] - off, e["end"] - off
            a0, b0 = max(0, a), min(n, b)
            if b0 <= a0:
                continue
            fa, fb = a0 // HOP, min(t, (b0 + HOP - 1) // HOP)
            frame_class[fa:fb] = np.maximum(frame_class[fa:fb], c)
            ev_window[max(0, a0 - pre): min(n, b0 + post)] = 1.0
        return {"noisy": torch.from_numpy(np.ascontiguousarray(noisy)), "clean": torch.from_numpy(np.ascontiguousarray(clean)),
                "frame_class": torch.from_numpy(frame_class), "ev_window": torch.from_numpy(ev_window),
                "id": m["id"], "scenario": m["scenario"], "snr": float(m["snr_lufs_db"])}
