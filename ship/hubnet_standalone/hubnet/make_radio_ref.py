"""Radio-clean reference targets: what the same radio link would deliver over an ideal channel with no
battlefield noise -- same link type, TX compressor, VOX clipping, band limits, codec (CVSD16/32 quantisation,
FM pre/de-emphasis and deviation clipping), but no acoustic noise, no RF hiss / clicks, no bit errors and no
squelch tail. Those are noise; the hub removes them. (Replaying the link *with* its RF impairments was tried
and rejected: CVSD bit errors act on the encoded signal, so the reference gets its own unpredictable damage.)

The v3.1 link draws from its own stream, seeded by the clip index alone (ancdata/battlefield.py:
rng([seed, index, 53]) -> on, link seed), so each clip's link type, VOX clipping and FM deviation clip are
replayed exactly on the stored clean target. The replay is checked against meta.jsonl (link and VOX length
must match).

Studio clips (no radio link, 10 %) keep their band-limited clean target. Output: <data>/<split>/ref/<id>.flac.

    python -m hubnet.make_radio_ref --data ~/anc_work/data/battlefield_v31_8k --workers 40
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

from .radio_v31 import EPS, SR_LINK, _compress, _cvsd_decode, _cvsd_encode, bandpass, db_to_lin, lowpass

SR = 16000
LINK_P = {"cvsd16": 0.55, "fm": 0.35, "cvsd32": 0.10}
VOX_P = 0.3


def ideal_link(c16: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """radio_v31.radio_link's random draws up to the codec (link, VOX, compressor, FM deviation clip), in the same
    order, then an error-free codec. The compressor gain is already in the stored target: its draws are consumed,
    its gain is not applied."""
    n = len(c16)
    names = list(LINK_P)
    w = np.array([LINK_P[k] for k in names])
    link = names[int(rng.choice(len(names), p=w / w.sum()))]
    tx_hi = 3000.0 if link == "fm" else 3400.0
    info = {"link": link}
    x = resample_poly(c16, 1, 2).astype(np.float32)
    env = np.abs(c16)
    on = np.where(env > 0.05 * (env.max() + EPS))[0]
    if len(on) and rng.random() < VOX_P:
        a = int(on[0]) // 2
        m = int(rng.uniform(0.06, 0.25) * SR_LINK)
        x[a: a + m] = 0.0
        L = int(rng.uniform(0.002, 0.005) * SR_LINK)
        t = np.arange(L) / SR_LINK
        x[a: a + L] += (db_to_lin(float(rng.uniform(-20, -10))) * np.sin(2 * np.pi * rng.uniform(1000, 2000) * t)).astype(np.float32)
        info["vox_ms"] = 1000.0 * m / SR_LINK
    _compress(x, rng, (4.0, 10.0))                       # draws only
    x = bandpass(x, 300.0, tx_hi)
    if link == "fm":
        pre_b = np.array([1.0, -0.8])
        e = lfilter(pre_b, [1.0], x).astype(np.float32)
        e = e / (np.abs(e).max() + EPS)
        clip = float(rng.uniform(0.5, 0.9))
        e = np.clip(e, -clip, clip) / clip
        e = lowpass(e, 3000.0)
        y = lfilter([1.0], pre_b, e).astype(np.float32)
    else:
        fs = (16 if link == "cvsd16" else 32) * 1000
        up = fs // SR_LINK
        z = resample_poly(x, up, 1).astype(np.float64)
        z = z / (np.abs(z).max() + EPS) * 0.9
        beta_s, beta_p = np.exp(-1 / (fs * 0.005)), np.exp(-1 / (fs * 0.001))
        d_min = 0.08 * (16000 / fs)
        bits = _cvsd_encode(z, beta_s, beta_p, d_min, 16.0 * d_min)
        y = _cvsd_decode(bits, beta_s, beta_p, d_min, 16.0 * d_min).astype(np.float32)
        y = resample_poly(lowpass(y, 3400.0, fs, 255), 1, up)[: len(x)].astype(np.float32)
    y = bandpass(y, 300.0, 3400.0)
    out = resample_poly(y, 2, 1)[:n].astype(np.float32)
    out *= np.sqrt(np.mean(c16 ** 2) / (np.mean(out ** 2) + EPS))
    return out, info


def _read(p: Path) -> tuple[np.ndarray, int]:
    for ext in (".flac", ".wav"):
        q = p.with_suffix(ext)
        if q.exists():
            x, fs = sf.read(str(q), dtype="float32", always_2d=True)
            return x[:, 0], fs
    raise FileNotFoundError(p)


def one(job: tuple[str, dict]) -> dict:
    root, m = job
    root = Path(root)
    r = m.get("radio")
    if not r:
        return {"id": m["id"], "radio": False}
    rr = np.random.default_rng([int(m["seed"]), int(m["index"]), 53])
    on = bool(rr.random() < 0.9)
    seed = int(rr.integers(2 ** 31))
    c, fs = _read(root / "clean" / m["id"])
    c16 = c if fs == SR else resample_poly(c, SR // fs, 1).astype(np.float32)
    y, info = ideal_link(c16, np.random.default_rng(seed))
    ok = on and info["link"] == r["link"] and abs(info.get("vox_ms", -1) - r.get("vox_ms", -1)) < 1e-6
    out = root / "ref" / f"{m['id']}.flac"
    y8 = resample_poly(y, 1, 2).astype(np.float32)
    sf.write(str(out), np.clip(y8, -1, 1), 8000, subtype="PCM_16")
    return {"id": m["id"], "radio": True, "link": info["link"], "match": bool(ok)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--splits", default="test,val,train")
    a = ap.parse_args()
    for split in a.splits.split(","):
        root = a.data / split
        (root / "ref").mkdir(exist_ok=True)
        meta = [json.loads(l) for l in (root / "meta.jsonl").open(encoding="utf-8") if l.strip()]
        n = mism = radio = 0
        with ProcessPoolExecutor(a.workers) as ex:
            for res in ex.map(one, [(str(root), m) for m in meta], chunksize=32):
                n += 1
                if res["radio"]:
                    radio += 1
                    mism += not res["match"]
        print(f"{split}: {n} pairs, {radio} radio refs written, {mism} replay mismatches", flush=True)


if __name__ == "__main__":
    main()
