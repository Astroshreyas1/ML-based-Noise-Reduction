"""Training / eval pairs for HubNet, self-contained (no `ancdata` import).

Layout expected under --data (copied from the laptop's data/battlefield_v31):
    <data>/{train,val,test}/{noisy,clean}/*.flac|wav + meta.jsonl
    <data>/words/*.parquet           (lombard6s.parquet, radio6s.parquet: word timings + keyword flags)

Item: noisy (N,) boom channel as received over the radio link (channel 0), clean (N,) radio-band target
(studio-band clips are band-limited here with the same linear-phase filters the link uses), valid (N,) =
0 inside the VOX-muted head, kw (N,) = 1 inside military / radio keyword spans (+-50 ms).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import firwin, resample_poly
from torch.utils.data import Dataset

SR = 16000
SR_LINK = 8000


# ---- the link's linear part (copy of ancdata/radio.py link_filters) --------------------------
def _fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    d = (len(h) - 1) // 2
    return np.convolve(x, h, mode="full")[d: d + len(x)].astype(np.float32)


def _bandpass(x: np.ndarray, lo: float, hi: float, taps: int = 255) -> np.ndarray:
    return _fir(x, firwin(taps, [lo, hi], pass_zero=False, fs=SR_LINK))


def link_filters(x16: np.ndarray, tx_hi: float = 3400.0) -> np.ndarray:
    y = resample_poly(x16, 1, 2).astype(np.float32)
    y = _bandpass(_bandpass(y, 300.0, tx_hi), 300.0, 3400.0)
    return resample_poly(y, 2, 1)[: len(x16)].astype(np.float32)


def load_words(words_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in Path(words_dir).glob("*.parquet"):
        for r in pd.read_parquet(f).itertuples(index=False):
            out[str(r.id)] = json.loads(r.words)
    return out


HARD = {"urban_patrol", "artillery", "helo_lz"}      # weakest scenarios in the step-16k diagnosis


class HubPairs(Dataset):
    """target="ref": radio clips use the ideal-channel radio reference (<split>/ref, hubnet.make_radio_ref);
    target="clean": the pre-radio band-limited speech (runs before 2026-09-29)."""

    def __init__(self, root: str | Path, crop_seconds: float | None = 2.0, limit: int | None = None, seed: int = 0,
                 words: dict[str, list[dict]] | None = None, target: str = "ref", hard_weight: float = 1.5):
        self.root = Path(root)
        self.target = target
        self.hard_weight = hard_weight
        self.meta = [json.loads(l) for l in (self.root / "meta.jsonl").open(encoding="utf-8") if l.strip()]
        if limit:
            self.meta = random.Random(seed).sample(self.meta, min(limit, len(self.meta)))
        self.crop = int(crop_seconds * SR) if crop_seconds else None
        self.words = words or {}

    def __len__(self) -> int:
        return len(self.meta)

    def _read(self, sub: str, stem: str) -> np.ndarray:
        for ext in (".flac", ".wav"):
            p = self.root / sub / f"{stem}{ext}"
            if p.exists():
                x, fs = sf.read(str(p), dtype="float32", always_2d=True)
                if fs != SR:                               # repacked at 8 kHz for a small disk: back to 16 kHz
                    x = resample_poly(x, SR // np.gcd(SR, fs), fs // np.gcd(SR, fs), axis=0).astype(np.float32)
                return x
        raise FileNotFoundError(f"{self.root / sub / stem}")

    def item(self, i: int, off: int | None = None) -> dict:
        m = self.meta[i]
        noisy = self._read("noisy", m["id"])[:, 0]
        radio = m.get("radio")
        if radio and self.target == "ref":
            clean = self._read("ref", m["id"])[:, 0]
        else:
            clean = self._read("clean", m["id"])[:, 0]
            if not radio:
                clean = link_filters(clean)
        n0 = min(len(noisy), len(clean))
        noisy, clean = noisy[:n0], clean[:n0]
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
            if off is None:                         # bias crops toward speech: up to 4 tries for >= 25 % active
                fr = 320
                e = np.convolve(clean ** 2, np.ones(fr) / fr, mode="same")
                act = e > 1e-4 * (e.max() + 1e-12)
                for _ in range(4):
                    off = random.randint(0, n - self.crop)
                    if act[off: off + self.crop].mean() >= 0.25:
                        break
            sl = slice(off, off + self.crop)
            noisy, clean, valid, kw = noisy[sl], clean[sl], valid[sl], kw[sl]
        w = self.hard_weight if (m.get("scenario") in HARD or m.get("gunfire")) else 1.0
        return {"noisy": noisy, "clean": clean, "valid": valid, "kw": kw, "w": np.float32(w), "id": m["id"],
                "link": (radio or {}).get("link", "studio"), "snr": float(m["snr_lufs_db"]), "scenario": m["scenario"],
                "corpus": m.get("speech_corpus")}

    def __getitem__(self, i: int) -> dict:
        d = self.item(i)
        return {k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else
                    torch.tensor(v) if isinstance(v, np.floating) else v) for k, v in d.items()}


def collate(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        out[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else v
    return out
