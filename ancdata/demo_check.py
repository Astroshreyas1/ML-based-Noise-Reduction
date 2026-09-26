"""Demo-day guards.

The most likely live failure is not the model: it is the demo microphone
chain (gain, AGC, OS resampler, a laptop's built-in noise suppression)
putting the input outside the training distribution. Two tools:

* :func:`registered_ranges` — computed when an eval set is materialised and
  written to ``<eval>/ranges.json``: peak dBFS, RMS dBFS, noise floor and DC
  offset percentiles of the *training-style* input.
* :func:`demo_capture_check` — take a 5 s capture through the exact demo
  chain (a wav file, or record it if `sounddevice` is installed) and assert
  it falls inside those ranges. Run it on the demo laptop at the venue,
  before the talk. If it fails, fix the gain, not the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio import EPS, load_mono
from .config import SR


def _stats(x: np.ndarray, sr: int = SR) -> dict[str, float]:
    n = int(sr * 0.02)
    frames = x[: len(x) - len(x) % n].reshape(-1, n)
    e_db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    return {
        "peak_dbfs": float(20 * np.log10(np.abs(x).max() + EPS)),
        "rms_dbfs": float(10 * np.log10(np.mean(x ** 2) + EPS)),
        "noise_floor_dbfs": float(np.percentile(e_db, 10)),
        "dc_offset": float(abs(x.mean())),
        "clip_frac": float(np.mean(np.abs(x) > 0.999)),
    }


def registered_ranges(noisy_files: list[Path], out: Path, sr: int = SR, channel: int = 0) -> dict:
    """Percentile envelope of the training-style input, written as JSON."""
    import soundfile as sf

    rows = []
    for p in noisy_files:
        a, _ = sf.read(str(p), dtype="float32", always_2d=True)
        rows.append(_stats(np.ascontiguousarray(a[:, channel]), sr))
    keys = rows[0].keys()
    rng = {k: {"p02": float(np.percentile([r[k] for r in rows], 2)),
               "p98": float(np.percentile([r[k] for r in rows], 98))} for k in keys}
    rng["n"] = len(rows)
    rng["sample_rate"] = sr
    out.write_text(json.dumps(rng, indent=2), encoding="utf-8")
    return rng


def record(seconds: float = 5.0, sr: int = SR, device: int | str | None = None) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install sounddevice, or pass a wav captured through the demo chain") from e
    x = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    return np.ascontiguousarray(x[:, 0])


def demo_capture_check(ranges_json: Path, wav: Path | None = None, seconds: float = 5.0,
                       margin_db: float = 3.0) -> dict:
    """Assert a live capture sits inside the registered training ranges.
    Returns the capture stats; raises AssertionError with the offending fields."""
    ranges = json.loads(Path(ranges_json).read_text(encoding="utf-8"))
    if wav is not None:
        import soundfile as sf
        info = sf.info(str(wav))
        native_sr = info.samplerate
        x = load_mono(wav)
    else:
        native_sr = SR
        x = record(seconds)
    st = _stats(x)
    st["native_sample_rate"] = native_sr
    problems = []
    for k in ("peak_dbfs", "rms_dbfs", "noise_floor_dbfs"):
        lo, hi = ranges[k]["p02"] - margin_db, ranges[k]["p98"] + margin_db
        if not (lo <= st[k] <= hi):
            problems.append(f"{k}={st[k]:.1f} dB outside [{lo:.1f}, {hi:.1f}]")
    if st["dc_offset"] > max(0.01, 2 * ranges["dc_offset"]["p98"]):
        problems.append(f"dc_offset={st['dc_offset']:.4f} (coupling / bad ADC?)")
    if st["clip_frac"] > max(0.02, 2 * ranges["clip_frac"]["p98"]):
        problems.append(f"clip_frac={st['clip_frac']:.3f}: input is clipping far more than training data")
    if native_sr not in (16000, 32000, 44100, 48000):
        problems.append(f"native sample rate {native_sr} is unusual; check the OS audio settings")
    if problems:
        raise AssertionError("demo capture OUTSIDE training distribution:\n  " + "\n  ".join(problems)
                             + "\nFix the gain / device settings, not the model.")
    return st
