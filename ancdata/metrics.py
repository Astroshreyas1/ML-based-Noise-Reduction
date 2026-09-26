"""Evaluation metrics and the per-condition results table.

Report broken out by noise category and by transient SNR, never one average:
a single number over 25 %-impulsive data hides total failure on gunshots
inside a respectable-looking mean.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .audio import load_mono
from .config import SR, STFT_HOP


def si_snr_db(est: np.ndarray, ref: np.ndarray) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    s_target = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12) * ref
    e_noise = est - s_target
    return float(10.0 * np.log10((np.sum(s_target ** 2) + 1e-12) / (np.sum(e_noise ** 2) + 1e-12)))


def snr_db(est: np.ndarray, ref: np.ndarray) -> float:
    return float(10.0 * np.log10((np.sum(ref ** 2) + 1e-12) / (np.sum((est - ref) ** 2) + 1e-12)))


def stoi(est: np.ndarray, ref: np.ndarray, sr: int = SR) -> float:
    from pystoi import stoi as _stoi
    return float(_stoi(ref, est, sr, extended=False))


def pesq_wb(est: np.ndarray, ref: np.ndarray, sr: int = SR) -> float:
    from pesq import pesq
    return float(pesq(sr, ref, est, "wb"))


def chunked(enhance: Callable[[np.ndarray], np.ndarray], hop: int = STFT_HOP) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap an enhancer so it is fed causally in `hop`-sample blocks (8 ms at
    16 kHz), the way the live demo feeds it. A stateless enhancer that peeks
    at the whole utterance will score differently here than on the full
    signal; that difference is the latency lie a judge would catch. The
    enhancer must accept a block and return a block of the same length."""
    def run(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        for i in range(0, len(x), hop):
            blk = x[i:i + hop]
            y = enhance(blk)
            if len(y) != len(blk):
                raise ValueError(f"chunked enhancer returned {len(y)} samples for a {len(blk)}-sample block")
            out[i:i + hop] = y
        return out
    return run


def nonintrusive_snr_db(x: np.ndarray, sr: int = SR, frame_ms: float = 20.0, thresh_db: float = -45.0) -> float:
    """Reference-free SNR estimate for live recordings (row 3b): energy of the
    loudest 30 % of frames (speech) over the quietest 20 % (noise floor).
    Crude, honest, and labelled as such on the slide."""
    n = int(sr * frame_ms / 1000.0)
    frames = x[: len(x) - len(x) % n].reshape(-1, n)
    e = np.sort(np.mean(frames ** 2, axis=1))
    lo = e[: max(1, int(0.2 * len(e)))].mean() + 1e-12
    hi = e[int(0.7 * len(e)):].mean() + 1e-12
    return float(10.0 * np.log10(hi / lo))


def scorer_selftest(sr: int = SR, with_pesq: bool = True, with_stoi: bool = True) -> dict[str, float]:
    """Known-answer test of the metric implementations. Raises on failure.

    identical pair            -> SI-SNR very high, STOI ~1.0, PESQ-WB ~4.64
    white noise at +10 dB SNR -> measured SNR within 0.5 dB of 10
    chunked() identity        -> bit-identical to the input
    """
    rng = np.random.default_rng(0)
    t = np.arange(int(sr * 3.0)) / sr
    # speech-like reference: AM-modulated harmonic series (PESQ rejects pure tones / silence)
    f0 = 140.0 * (1 + 0.1 * np.sin(2 * np.pi * 0.5 * t))
    ph = 2 * np.pi * np.cumsum(f0) / sr
    ref = sum(np.sin(k * ph) / k for k in range(1, 8)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)) ** 2
    ref = (0.3 * ref / np.abs(ref).max()).astype(np.float32)
    out: dict[str, float] = {}
    out["si_snr_identical"] = si_snr_db(ref, ref)
    if out["si_snr_identical"] < 60:
        raise AssertionError(f"SI-SNR of identical signals is {out['si_snr_identical']:.1f} dB, expected > 60")
    if with_stoi:
        out["stoi_identical"] = stoi(ref, ref, sr)
        if out["stoi_identical"] < 0.99:
            raise AssertionError(f"STOI of identical signals is {out['stoi_identical']:.3f}, expected ~1")
    if with_pesq:
        out["pesq_identical"] = pesq_wb(ref, ref, sr)
        if out["pesq_identical"] < 4.5:
            raise AssertionError(f"PESQ-WB of identical signals is {out['pesq_identical']:.2f}, expected ~4.64")
    noise = rng.standard_normal(len(ref)).astype(np.float32)
    noise *= np.sqrt(np.mean(ref ** 2)) / np.sqrt(np.mean(noise ** 2)) * 10 ** (-10 / 20)
    out["snr_plus10"] = snr_db(ref + noise, ref)
    if abs(out["snr_plus10"] - 10.0) > 0.5:
        raise AssertionError(f"SNR of a +10 dB mixture measured {out['snr_plus10']:.2f} dB")
    y = chunked(lambda b: b)(ref)
    if not np.array_equal(y, ref):
        raise AssertionError("chunked() identity is not bit-exact")
    return out


def evaluate_set(eval_root: Path, enhance: Callable[[np.ndarray], np.ndarray] | None = None,
                 with_pesq: bool = True, with_stoi: bool = True, channel: int = 0) -> pd.DataFrame:
    """Score every pair in a materialised set. `enhance` maps noisy boom
    channel -> estimate; None scores the unprocessed input (the baseline row)."""
    eval_root = Path(eval_root)
    rows = []
    with (eval_root / "meta.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            nfile = _find(eval_root / "noisy", m["id"])
            noisy = load_mono(nfile) if channel is None else _load_channel(nfile, channel)
            clean = load_mono(_find(eval_root / "clean", m["id"]))
            est = enhance(noisy) if enhance else noisy
            r = {
                "id": m["id"], "ambience": m.get("ambience"), "snr_db": m.get("snr_db"),
                "n_events": len(m.get("events", [])),
                "min_local_snr_db": min([e["local_snr_db"] for e in m.get("events", [])], default=np.nan),
                "si_snr": si_snr_db(est, clean), "snr": snr_db(est, clean),
            }
            if with_stoi:
                r["stoi"] = stoi(est, clean)
            if with_pesq:
                try:
                    r["pesq"] = pesq_wb(est, clean)
                except Exception:  # pesq raises on silent/degenerate frames
                    r["pesq"] = np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def _find(d: Path, stem: str) -> Path:
    for ext in (".wav", ".flac"):
        if (d / f"{stem}{ext}").exists():
            return d / f"{stem}{ext}"
    raise FileNotFoundError(f"{d / stem}.wav|.flac")


def _load_channel(path: Path, ch: int) -> np.ndarray:
    import soundfile as sf
    a, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(a[:, ch])


def results_table(df: pd.DataFrame) -> str:
    cols = [c for c in ("si_snr", "stoi", "pesq") if c in df.columns]
    lines = []
    overall = df[cols].mean()
    lines.append(("all", *overall.values))
    for cat, sub in df.groupby("ambience"):
        lines.append((str(cat), *sub[cols].mean().values))
    imp = df[df.n_events > 0]
    if len(imp):
        lines.append(("with transients (all)", *imp[cols].mean().values))
        hi = imp[imp.min_local_snr_db > 6]
        lo = imp[imp.min_local_snr_db < 0]
        if len(hi):
            lines.append(("  transient SNR > 6 dB", *hi[cols].mean().values))
        if len(lo):
            lines.append(("  transient SNR < 0 dB", *lo[cols].mean().values))
    w = max(len(l[0]) for l in lines) + 2
    header = " " * w + "".join(f"{c:>9}" for c in cols)
    body = "\n".join(f"{l[0]:<{w}}" + "".join(f"{v:9.2f}" for v in l[1:]) for l in lines)
    return header + "\n" + body
