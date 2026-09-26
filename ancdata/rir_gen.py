"""Room impulse response bank.

Recorded RIRs for "armoured vehicle cabin" or "bunker" do not exist publicly,
so rooms are simulated with pyroomacoustics (image-source method).

Each room yields FOUR impulse responses because the headset has two capture
points and the scene has two sources:

    speech -> boom   near-field, mouth to boom mic (~2-3 cm). Direct path dominates
    speech -> ref    mouth to external reference mic on the ear-cup (~8-12 cm)
    noise  -> boom   far-field ambient source to boom mic
    noise  -> ref    far-field ambient source to reference mic

Convolving speech and noise with RIRs from the *same* room keeps the mixture
physically consistent: a reverberant noise with dry speech (or vice versa) is
a cue the network would learn instead of learning to denoise.

The bank is stored as a single float32 ``.npz`` — portable, no pickles.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import SR
from .paths import rir_bank_path

# name -> (dim ranges m, absorption range). Low absorption = hard/reflective.
PRESETS: dict[str, tuple[list[tuple[float, float]], tuple[float, float]]] = {
    "vehicle_cabin": ([(1.5, 2.5), (1.5, 3.0), (1.2, 1.8)], (0.05, 0.15)),
    "bunker":        ([(2.5, 5.0), (2.5, 5.0), (2.0, 3.0)], (0.05, 0.20)),
    "room":          ([(3.0, 7.0), (3.0, 7.0), (2.4, 3.2)], (0.25, 0.60)),
    "corridor":      ([(1.5, 2.5), (6.0, 15.0), (2.4, 3.0)], (0.10, 0.30)),
    "hangar":        ([(10.0, 25.0), (10.0, 25.0), (5.0, 12.0)], (0.10, 0.30)),
}
RIR_LEN = int(SR * 1.0)  # 1 s is enough for every preset at these absorptions
KEYS = ("speech_boom", "speech_ref", "noise_boom", "noise_ref")


def _place(rng: np.random.Generator, dims: list[float], margin: float = 0.4) -> np.ndarray:
    return np.array([rng.uniform(margin, d - margin) for d in dims])


def _simulate_room(rng: np.random.Generator, preset: str, sr: int = SR,
                   max_order: int = 10) -> tuple[np.ndarray, dict[str, float]]:
    import pyroomacoustics as pra  # heavy import, only when generating

    dims_rng, absorp_rng = PRESETS[preset]
    dims = [float(rng.uniform(lo, hi)) for lo, hi in dims_rng]
    a = float(rng.uniform(*absorp_rng))
    # Mouth position; boom mic 2.5 cm in front, ref mic ~10 cm to the side (ear-cup).
    mouth = _place(rng, dims, margin=0.6)
    forward = np.array([1.0, 0.0, 0.0])
    side = np.array([0.0, 1.0, 0.0])
    boom = mouth + 0.025 * forward
    ref = mouth + float(rng.uniform(0.08, 0.12)) * side
    # noise source at least min_sep from the mouth; min_sep shrinks with the room so a
    # 1.5 m vehicle cabin cannot spin forever looking for an impossible placement
    min_sep = min(0.8, 0.3 * float(np.linalg.norm(dims)))
    noise_src = _place(rng, dims, margin=0.3)
    for _ in range(200):
        if np.linalg.norm(noise_src - mouth) >= min_sep:
            break
        noise_src = _place(rng, dims, margin=0.3)

    out = np.zeros((4, RIR_LEN), dtype=np.float32)
    for si, src in enumerate((mouth, noise_src)):
        room = pra.ShoeBox(dims, fs=sr, materials=pra.Material(a), max_order=max_order)
        room.add_source(src.tolist())
        room.add_microphone_array(pra.MicrophoneArray(np.stack([boom, ref], axis=1), sr))
        room.compute_rir()
        for mi in range(2):
            h = np.asarray(room.rir[mi][0], dtype=np.float32)
            h = h[:RIR_LEN]
            out[si * 2 + mi, : len(h)] = h
    # Normalise all four by the speech->boom direct-path peak so relative
    # levels between channels are preserved (ref is quieter than boom).
    scale = float(np.abs(out[0]).max() + 1e-9)
    out /= scale
    meta = {
        "dim_x": dims[0], "dim_y": dims[1], "dim_z": dims[2], "absorption": a,
        "noise_dist_m": float(np.linalg.norm(noise_src - mouth)),
    }
    return out, meta


def drr_db(h: np.ndarray, sr: int = SR, direct_ms: float = 2.5) -> float:
    """Direct-to-reverberant ratio around the first peak."""
    pk = int(np.argmax(np.abs(h)))
    w = int(sr * direct_ms / 1000.0)
    d = h[max(0, pk - w): pk + w]
    r = np.concatenate([h[: max(0, pk - w)], h[pk + w:]])
    return float(10.0 * np.log10((np.sum(d ** 2) + 1e-12) / (np.sum(r ** 2) + 1e-12)))


def generate_bank(n_per_preset: int = 400, seed: int = 0, out: Path | None = None,
                  presets: dict | None = None, sr: int = SR, verbose: bool = True) -> Path:
    """Simulate `n_per_preset` rooms per preset and save one .npz bank."""
    from tqdm import tqdm

    presets = presets or PRESETS
    rng = np.random.default_rng(seed)
    rirs, names, metas = [], [], []
    for name in presets:
        for _ in tqdm(range(n_per_preset), desc=name, disable=not verbose):
            h, m = _simulate_room(rng, name, sr)
            rirs.append(h)
            names.append(name)
            metas.append([m["dim_x"], m["dim_y"], m["dim_z"], m["absorption"], m["noise_dist_m"],
                          drr_db(h[0], sr), drr_db(h[2], sr)])
    out = Path(out) if out else rir_bank_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rirs=np.stack(rirs), preset=np.array(names),
                        meta=np.asarray(metas, dtype=np.float32),
                        meta_cols=np.array(["dim_x", "dim_y", "dim_z", "absorption",
                                            "noise_dist_m", "drr_speech_boom_db", "drr_noise_boom_db"]),
                        keys=np.array(KEYS), sr=np.int32(sr))
    if verbose:
        print(f"wrote {len(rirs)} rooms x 4 RIRs -> {out}")
    return out


def synthetic_bank(n: int = 20, seed: int = 0, out: Path | None = None, sr: int = SR) -> Path:
    """Fast stand-in bank with no pyroomacoustics (exponentially decaying noise
    tails with a direct-path spike). Used by the smoke test only."""
    rng = np.random.default_rng(seed)
    rirs = np.zeros((n, 4, RIR_LEN), dtype=np.float32)
    t = np.arange(RIR_LEN) / sr
    names, metas = [], []
    for i in range(n):
        rt60 = float(rng.uniform(0.15, 0.9))
        tail = np.exp(-6.9 * t / rt60)
        for k in range(4):
            h = rng.standard_normal(RIR_LEN).astype(np.float32) * tail
            h *= 0.05 if k == 0 else 0.3  # boom-speech nearly dry; others reverberant
            delay = 1 if k == 0 else int(rng.integers(3, 40))
            h[delay] += 1.0 if k == 0 else 0.6
            rirs[i, k] = h
        rirs[i] /= np.abs(rirs[i, 0]).max() + 1e-9
        names.append("synthetic")
        metas.append([3, 3, 2.5, 0.3, 2.0, drr_db(rirs[i, 0], sr), drr_db(rirs[i, 2], sr)])
    out = Path(out) if out else rir_bank_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rirs=rirs, preset=np.array(names), meta=np.asarray(metas, dtype=np.float32),
                        meta_cols=np.array(["dim_x", "dim_y", "dim_z", "absorption", "noise_dist_m",
                                            "drr_speech_boom_db", "drr_noise_boom_db"]),
                        keys=np.array(KEYS), sr=np.int32(sr))
    return out


class RirBank:
    """Loaded bank with a split column so eval rooms never appear in training."""

    def __init__(self, path: Path | None = None, test_frac: float = 0.1):
        path = Path(path) if path else rir_bank_path()
        if not path.exists():
            raise FileNotFoundError(f"RIR bank missing: {path}. Run `ancdata rir` first.")
        z = np.load(path)
        if int(z["sr"]) != SR:
            raise ValueError(f"{path}: bank sample rate {int(z['sr'])} != {SR}")
        self.rirs: np.ndarray = self._align(z["rirs"])
        self.preset: np.ndarray = z["preset"]
        self.meta: np.ndarray = z["meta"]
        self.meta_cols = [str(c) for c in z["meta_cols"]]
        n = len(self.rirs)
        # deterministic split by index hash so it is stable across machines
        h = (np.arange(n) * 2654435761 % 2**32) / 2**32
        self.split = np.where(h < test_frac, "test", "train")
        self.path = path

    @staticmethod
    def _align(rirs: np.ndarray) -> np.ndarray:
        """pyroomacoustics prepends a ~40-sample fractional-delay filter, so the
        speech->boom direct path lands at index ~41. Shift every room's four
        RIRs left by the same amount so the direct path sits at index 0 and the
        input stays sample-aligned with the dry target; relative delays between
        channels and sources are preserved."""
        out = np.zeros_like(rirs)
        for i in range(len(rirs)):
            d = int(np.argmax(np.abs(rirs[i, 0])))
            if d > 0:
                out[i, :, :-d] = rirs[i, :, d:]
            else:
                out[i] = rirs[i]
        return out

    def draw(self, rng: np.random.Generator, split: str) -> tuple[int, np.ndarray, dict[str, float]]:
        idx = np.where(self.split == split)[0]
        if len(idx) == 0:
            raise RuntimeError(f"no rooms for split {split!r} in {self.path}")
        i = int(rng.choice(idx))
        meta = {c: float(v) for c, v in zip(self.meta_cols, self.meta[i])}
        meta["preset"] = str(self.preset[i])
        return i, self.rirs[i], meta
