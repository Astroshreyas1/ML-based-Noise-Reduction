"""In-house recordings: the bridge from synthetic data to the real headset.

Three things live here, all fed by the 90-minute recording session
(scripts/record_session.md):

1. :func:`make_sweep` / :func:`fit_mic_ir` — own-voice / microphone transfer
   function. Play an exponential sine sweep through a small speaker at the
   mouth position, record it on the boom mic, deconvolve. The resulting short
   impulse response (32 ms) is applied to the boom channel of every synthetic
   example so training audio carries *this* capsule's colouration
   (hearable own-voice-reconstruction literature does exactly this).

2. :func:`mix_realcheck` — evaluation row "3a": the presenter recorded on the
   boom mic in a quiet room is the clean reference; real noise recorded at
   the venue (babble, claps, PA hum) and real gunshot files are mixed in
   digitally through the same ADC stage. Alignment is exact by construction,
   so STOI/PESQ are valid — and every source is real.

3. :func:`realcheck_stats` — reference-free statistics of fully live takes
   (row "3b": loudspeaker noise while speaking) for the reality-gap plot and
   the non-intrusive SNR column.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from .adc import simulate_adc
from .audio import EPS, active_rms_db, load_mono, normalise_rms, peak, place_or_crop, rms, tile_or_crop, trim_silence, write_wav
from .config import SEG_SAMPLES, SR, child_rng, sample
from .metrics import nonintrusive_snr_db

MIC_IR_TAPS = int(SR * 0.032)


# ------------------------------------------------------------------ mic IR
def make_sweep(seconds: float = 8.0, f0: float = 40.0, f1: float = 7800.0, sr: int = SR,
               pad_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Exponential sine sweep and its inverse filter (Farina). Returns (sweep, inverse)."""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    R = np.log(f1 / f0)
    sweep = np.sin(2 * np.pi * f0 * seconds / R * (np.exp(t * R / seconds) - 1.0))
    # amplitude envelope for the inverse: compensates the 6 dB/oct energy tilt of the ESS
    inv = sweep[::-1] * np.exp(-t * R / seconds)
    inv /= np.abs(fftconvolve(sweep, inv)).max()
    pad = np.zeros(int(sr * pad_s))
    return np.concatenate([pad, 0.5 * sweep, pad]).astype(np.float32), inv.astype(np.float32)


def fit_mic_ir(recorded: np.ndarray, inverse: np.ndarray, taps: int = MIC_IR_TAPS) -> np.ndarray:
    """Deconvolve the recorded sweep; keep the linear response around the main peak."""
    h = fftconvolve(recorded, inverse)
    pk = int(np.argmax(np.abs(h)))
    ir = h[pk: pk + taps]          # peak at index 0: colouration without added delay
    ir = ir / (np.abs(ir).max() + EPS)
    return ir.astype(np.float32)


def save_mic_ir(ir: np.ndarray, out: Path, note: str = "") -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ir=ir.astype(np.float32), sr=np.int32(SR), note=np.array(note))


def load_mic_ir(path: str | Path | None) -> np.ndarray | None:
    if path is None or not Path(path).exists():
        return None
    z = np.load(path)
    if int(z["sr"]) != SR:
        raise ValueError(f"{path}: mic IR sample rate {int(z['sr'])} != {SR}")
    return z["ir"].astype(np.float32)


def apply_mic_ir(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Causal convolution, level-preserving (unity gain at the IR peak)."""
    return fftconvolve(x, ir, mode="full")[: len(x)].astype(np.float32)


# -------------------------------------------------------------- row 3a mix
def _audio_files(d: Path) -> list[Path]:
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in (".wav", ".flac"))


def mix_realcheck(cfg: dict[str, Any], speech_dir: Path, noise_dir: Path, n: int, out: Path,
                  impulsive_dir: Path | None = None, seed: int = 7, verbose: bool = True) -> Path:
    """Real speech (quiet-room boom-mic takes) x real venue noise x real transients.

    Uses the config's SNR distribution, peak-ratio range, ADC stage and
    impulsive probability so the mixture statistics match training; nothing
    synthetic enters. Writes the same layout as materialize().
    """
    speech = [(p, load_mono(p)) for p in _audio_files(speech_dir)]
    noise = [(p, load_mono(p)) for p in _audio_files(noise_dir)]
    imps = [(p, load_mono(p)) for p in _audio_files(impulsive_dir)] if impulsive_dir else []
    if not speech or not noise:
        raise FileNotFoundError("need at least one speech file and one noise file")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists; eval sets are frozen")
    (out / "noisy").mkdir(parents=True); (out / "clean").mkdir(parents=True)
    imp_cfg, adc_cfg = cfg["impulsive"], cfg["adc"]
    with (out / "meta.jsonl").open("w", encoding="utf-8") as fh:
        for k in range(n):
            rng = child_rng(seed, k)
            sp_path, sp = speech[int(rng.integers(len(speech)))]
            for _ in range(50):
                seg = place_or_crop(sp, SEG_SAMPLES, rng)
                if active_rms_db(seg) > -35.0:
                    break
            clean = normalise_rms(seg, -25.0)
            nz_path, nz = noise[int(rng.integers(len(noise)))]
            nz = tile_or_crop(trim_silence(nz), SEG_SAMPLES, rng)
            snr_db = sample(cfg["mix_ambience"]["snr_db"], rng)
            noisy = clean + nz * (rms(clean) / (rms(nz) + EPS)) * 10 ** (-snr_db / 20)
            events = []
            if imps and rng.random() < float(imp_cfg["probability"]):
                ip, ev = imps[int(rng.integers(len(imps)))]
                ev = ev / peak(ev)
                if len(ev) >= SEG_SAMPLES:
                    ev = ev[: SEG_SAMPLES // 2]
                ratio = sample(imp_cfg["peak_ratio_db"], rng)
                ev = ev * peak(clean) * 10 ** (ratio / 20)
                s = int(rng.integers(0, SEG_SAMPLES - len(ev) + 1))
                noisy[s: s + len(ev)] += ev
                events.append({"start": s, "end": s + len(ev), "category": "real_" + ip.stem.split("_")[0],
                               "ratio_db": float(ratio)})
            meta: dict[str, Any] = {"id": f"{k:06d}", "speech_path": str(sp_path), "noise_path": str(nz_path),
                                    "ambience": nz_path.stem.split("_")[0], "snr_db": float(snr_db), "events": events,
                                    "real_sources": True}
            if rng.random() < float(adc_cfg["probability"]):
                hd = sample(adc_cfg["headroom_db"], rng)
                gain = float(10.0 ** (-hd / 20.0) / (np.abs(noisy).max() + 1e-9))
                noisy = simulate_adc(noisy, hd)
                meta["adc"] = {"headroom_db": float(hd), "gain": gain}
            else:
                gain = 0.99 / float(np.abs(noisy).max()) if np.abs(noisy).max() > 0.99 else 1.0
                noisy = noisy * gain
            clean = (clean * gain).astype(np.float32)   # target tracks the linear gain
            write_wav(out / "noisy" / f"{k:06d}.wav", np.stack([noisy, noisy * 0.0]).astype(np.float32))
            write_wav(out / "clean" / f"{k:06d}.wav", clean)
            fh.write(json.dumps(meta) + "\n")
    if verbose:
        print(f"wrote {n} real-source pairs -> {out}")
    return out


# ---------------------------------------------------------------- row 3b
def realcheck_stats(live_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for p in _audio_files(live_dir):
        x = load_mono(p)
        rows.append({"file": p.name, "seconds": len(x) / SR, "peak_dbfs": 20 * np.log10(peak(x)),
                     "nonintrusive_snr_db": nonintrusive_snr_db(x)})
    return rows
