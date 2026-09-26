"""Diagnostic plots that justify the data design.

1. crest factor (fixed 200 ms window around the peak, so a 30 ms synthetic
   event and a 5 s field recording are comparable) + attack time: synthetic
   blasts (raw AND after the ADC stage) vs real gunshot clips. Pass = the
   synthetic cloud encloses the real one.
   Measured post-ADC because that is the signal the network sees; a
   pre-clip comparison validates the wrong object.
2. average spectrum: speech vs gunshot on the same axes. They overlap, so no
   fixed filter can separate them — the visual case for using ML at all.
3. spectrogram triptych: clean / noisy / (optional) enhanced for one pair.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .adc import simulate_adc
from .audio import attack_time_ms, crest_factor_db, event_crest_db, load_mono
from .config import SR
from .physics import synth_blast
from .registry import load_manifest
from .paths import from_relative


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _real_gunshots(pile: str, limit: int = 200) -> list[np.ndarray]:
    m = load_manifest()
    rows = m[(m.pile == pile)]
    return [load_mono(from_relative(r)) for r in rows.path.head(limit)]


def plot_crest_attack(cfg: dict[str, Any], out: Path, n_synth: int = 200,
                      real_pile: str = "esc50_gunshot") -> Path:
    plt = _mpl()
    rng = np.random.default_rng(int(cfg["seed"]) + 1)
    params = None
    for e in cfg["sources"].get("impulsive", []):
        if isinstance(e, dict) and e.get("type") == "parametric":
            params = e.get("params")
    raw, post, room = [], [], []
    try:
        from .rir_gen import RirBank
        bank = RirBank()
    except (FileNotFoundError, ValueError):
        bank = None
    for _ in range(n_synth):
        x, _ = synth_blast(rng, params)
        raw.append((event_crest_db(x), attack_time_ms(x)))
        y = simulate_adc(x * 0.3, headroom_db=float(rng.uniform(-3, 6)))
        post.append((event_crest_db(y), attack_time_ms(y)))
        if bank is not None:
            # what the chain actually injects 70 % of the time: the blast through the room's
            # far-field noise path, then the ADC
            _, rirs, _ = bank.draw(rng, "train")
            z = np.convolve(x, rirs[2])[: len(x) + 4000]
            z = simulate_adc(z / (np.abs(z).max() + 1e-9) * 0.3, headroom_db=float(rng.uniform(-3, 6)))
            room.append((event_crest_db(z), attack_time_ms(z)))
    try:
        real = [(event_crest_db(x), attack_time_ms(x)) for x in _real_gunshots(real_pile)]
    except (FileNotFoundError, KeyError):
        real = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for data, label, mk in ((raw, "synthetic (raw)", "o"), (post, "synthetic (post-ADC)", "s"),
                            (room, "synthetic (through room + ADC)", "^"), (real, f"real ({real_pile})", "x")):
        if data:
            a = np.asarray(data)
            ax.scatter(a[:, 1], a[:, 0], label=f"{label}, n={len(a)}", marker=mk, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("10-90 % attack time (ms)")
    ax.set_ylabel("crest factor over a 200 ms window (dB)")
    ax.set_title("Impulse realism: synthetic must enclose real")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _avg_spectrum(clips: list[np.ndarray], n_fft: int = 1024) -> np.ndarray:
    acc = np.zeros(n_fft // 2 + 1)
    k = 0
    for x in clips:
        for i in range(0, len(x) - n_fft, n_fft // 2):
            acc += np.abs(np.fft.rfft(x[i:i + n_fft] * np.hanning(n_fft))) ** 2
            k += 1
    return 10 * np.log10(acc / max(k, 1) + 1e-12)


def plot_spectrum_overlap(speech_clips: list[np.ndarray], gun_clips: list[np.ndarray], out: Path) -> Path:
    plt = _mpl()
    f = np.fft.rfftfreq(1024, 1.0 / SR)
    s = _avg_spectrum(speech_clips)
    g = _avg_spectrum(gun_clips)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(f, s - s.max(), label="speech")
    ax.plot(f, g - g.max(), label="gunshot / blast")
    ax.set_xlabel("Hz"); ax.set_ylabel("dB (peak-normalised)")
    ax.set_title("Average spectra overlap: no fixed filter separates them")
    ax.legend(); ax.grid(alpha=0.3)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def plot_triptych(clean: np.ndarray, noisy: np.ndarray, out: Path, enhanced: np.ndarray | None = None,
                  title: str = "") -> Path:
    plt = _mpl()
    panels = [("clean (target)", clean), ("noisy (boom)", noisy)]
    if enhanced is not None:
        panels.append(("enhanced", enhanced))
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    floor = 1e-6 * np.random.default_rng(0).standard_normal(len(clean)).astype(np.float32)  # avoids log(0) on digital silence
    for ax, (name, x) in zip(axes, panels):
        ax.specgram(x + floor, NFFT=512, Fs=SR, noverlap=384, cmap="magma", vmin=-120, vmax=-20)
        ax.set_title(name); ax.set_xlabel("s")
    axes[0].set_ylabel("Hz")
    if title:
        fig.suptitle(title)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def _lta_spectrum_db(clips: list[np.ndarray], n_fft: int = 1024) -> np.ndarray:
    s = _avg_spectrum(clips, n_fft)
    return s - s.max()


def plot_reality_gap(synth_root: Path, real_files: list[Path], out: Path, channel: int = 0,
                     max_clips: int = 300) -> Path:
    """THE standout figure: synthetic training-style input vs real headset
    recordings on the same axes — long-term average spectrum, peak-level
    histogram and crest-factor histogram. If the real cloud sits inside the
    synthetic one, the pipeline models the device; if not, the plot says
    exactly which axis is wrong."""
    import soundfile as sf
    plt = _mpl()
    synth = []
    for p in sorted((Path(synth_root) / "noisy").glob("*.wav"))[:max_clips]:
        a, _ = sf.read(str(p), dtype="float32", always_2d=True)
        synth.append(np.ascontiguousarray(a[:, channel]))
    real = [load_mono(p) for p in real_files[:max_clips]]
    if not synth or not real:
        raise FileNotFoundError("need both a materialised synthetic set and real recordings")
    f = np.fft.rfftfreq(1024, 1.0 / SR)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(f, _lta_spectrum_db(synth), label=f"synthetic (n={len(synth)})")
    axes[0].plot(f, _lta_spectrum_db(real), label=f"real headset (n={len(real)})")
    axes[0].set_xscale("log"); axes[0].set_xlim(50, 8000)
    axes[0].set_xlabel("Hz"); axes[0].set_ylabel("dB rel. max"); axes[0].set_title("long-term average spectrum")
    axes[0].legend(); axes[0].grid(alpha=0.3, which="both")
    for data, label in ((synth, "synthetic"), (real, "real headset")):
        pk = [20 * np.log10(np.abs(x).max() + 1e-9) for x in data]
        cf = [crest_factor_db(x) for x in data]
        axes[1].hist(pk, bins=30, alpha=0.5, label=label, density=True)
        axes[2].hist(cf, bins=30, alpha=0.5, label=label, density=True)
    axes[1].set_xlabel("peak level (dBFS)"); axes[1].set_title("input level distribution"); axes[1].legend()
    axes[2].set_xlabel("crest factor (dB)"); axes[2].set_title("peakiness distribution"); axes[2].legend()
    fig.suptitle("Reality gap: does the synthetic pipeline model the device?")
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out
