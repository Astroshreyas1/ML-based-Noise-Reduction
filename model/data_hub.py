"""Hub training / eval pairs over data/battlefield_v31.

Item: noisy (N,) = the boom channel as received over the radio link (channel 0; the hub has no
ear-cup ref), clean (N,) = the radio-band target (studio clips are band-limited here the same
way, since the model outputs 0-4 kHz only), valid (N,) = 0 inside the VOX-muted head of a
transmission (no causal model can restore speech the radio never sent), kw (N,) = 1 inside
keyword spans (military / radio vocabulary, +-50 ms), from the snippet's whisper word timings.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ancdata.radio import link_filters  # noqa: E402

SR = 16000


def load_words(data_root: Path) -> dict[str, list[dict]]:
    """snippet id -> words (from every snippet set's words.parquet)."""
    out: dict[str, list[dict]] = {}
    for f in (data_root / "snippets").glob("*/words.parquet"):
        for r in pd.read_parquet(f).itertuples(index=False):
            out[str(r.id)] = json.loads(r.words)
    return out


class HubPairs(Dataset):
    def __init__(self, root: str | Path, crop_seconds: float | None = 2.0, limit: int | None = None, seed: int = 0,
                 words: dict[str, list[dict]] | None = None):
        self.root = Path(root)
        self.meta = [json.loads(l) for l in (self.root / "meta.jsonl").open(encoding="utf-8") if l.strip()]
        if limit:
            self.meta = random.Random(seed).sample(self.meta, min(limit, len(self.meta)))
        self.crop = int(crop_seconds * SR) if crop_seconds else None
        self.words = words if words is not None else load_words(ROOT / "data")

    def __len__(self) -> int:
        return len(self.meta)

    def _read(self, sub: str, stem: str) -> np.ndarray:
        for ext in (".flac", ".wav"):
            p = self.root / sub / f"{stem}{ext}"
            if p.exists():
                x, _ = sf.read(str(p), dtype="float32", always_2d=True)
                return x
        raise FileNotFoundError(stem)

    def item(self, i: int, off: int | None = None) -> dict:
        m = self.meta[i]
        noisy = self._read("noisy", m["id"])[:, 0]
        clean = self._read("clean", m["id"])[:, 0]
        radio = m.get("radio")
        if not radio:                                   # studio-band clip: same radio-band target
            clean = link_filters(clean, 3400.0)
        n = len(noisy)
        valid = np.ones(n, np.float32)
        if radio and radio.get("mute_span"):
            a, b = radio["mute_span"]
            valid[a:b] = 0.0
        kw = np.zeros(n, np.float32)
        pad = int(0.05 * SR)
        for w in self.words.get(str(m.get("speech_id")), []):
            if w.get("kw"):
                kw[max(0, int(w["start"] * SR) - pad): min(n, int(w["end"] * SR) + pad)] = 1.0
        if self.crop and n > self.crop:
            off = random.randint(0, n - self.crop) if off is None else off
            sl = slice(off, off + self.crop)
            noisy, clean, valid, kw = noisy[sl], clean[sl], valid[sl], kw[sl]
        return {"noisy": noisy, "clean": clean, "valid": valid, "kw": kw, "id": m["id"],
                "link": (radio or {}).get("link", "studio"), "snr": float(m["snr_lufs_db"]), "scenario": m["scenario"]}

    def __getitem__(self, i: int) -> dict:
        d = self.item(i)
        return {k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v) for k, v in d.items()}
