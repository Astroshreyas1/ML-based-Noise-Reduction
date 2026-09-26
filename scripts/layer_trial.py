"""Noise-layering listening trial: the same Lombard snippets through four
layering recipes, so a listener can say which one sounds like a real
battlefield boom-mic capture. Background: docs/NOISE_LAYERING.md.

    .venv/Scripts/python.exe scripts/layer_trial.py --n 10 --out outputs/layering_trial

Every sample k draws ONE plan (scenario, speech snippet, noise files, crop
offsets, envelope parameters, event times, SNR, level) from rng(seed, k);
the four recipes consume the same plan, so the only thing that differs
between the columns of the listening sheet is the layering recipe:

    naive        one static noise clip, whole-clip RMS SNR, nothing else.
                 What a first-cut pipeline does. Control.
    scene_lufs   Scaper / WHAM style: bed + 1-3 foreground events with time
                 envelopes and Doppler drift, SNR defined on K-weighted
                 loudness (ITU-R BS.1770) against the active speech level,
                 real gunshots at a peak ratio, wind dry at the capsule.
                 No propagation model, no room.
    scene_room   The current ancdata chain: the same scene, speech through a
                 near-dry boom RIR, far-field layers through the same
                 simulated room, active-level RMS SNR, ADC hard clip.
    outdoor      No room. Every source gets a distance and a reference SPL
                 from the military-noise literature; level at the capsule
                 follows 1/r, ISO 9613-1 air absorption (distance-dependent
                 low-pass), a ground-reflection comb for ground sources;
                 speech level from ISO 9921 vocal effort at 2.5 cm; the
                 ambient level is coupled to the talker's effort (Lombard
                 slope); 0 dBFS = 130 dB SPL; gunshots at their physical peak,
                 so near shots clip. SNR is a *consequence*, not a draw.

Outputs: <out>/speech/k.wav (dry), <out>/mix/<recipe>/k.wav, plan.json,
key.json (blind code -> recipe), listen.html (sheet with blind codes).

Status: trials 1 and 2 are done (verdicts in docs/NOISE_LAYERING.md section 6b-6c).
The chosen recipe (grain2) was ported into the package -- ancdata/loudness.py,
transients.py, channel.py, battlefield.py -- with the reduced gunfire ratios;
this script is kept as the A/B listening tool and still renders the trial
variants bit-for-bit (its own copies of the grain functions are frozen here).
Use `ancdata battlefield-listen` to audition the production chain.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ancdata.audio import EPS, active_rms_db, db_to_lin, load_mono, rms, tile_or_crop, trim_silence, write_wav  # noqa: E402
from ancdata.config import SR  # noqa: E402
from ancdata.scene import _doppler_drift, _gust_envelope, _passby_envelope, _ramp_envelope  # noqa: E402

DATA = ROOT / "data"
RAW = DATA / "raw"
SECONDS = 6.0
N = int(SR * SECONDS)
FULL_SCALE_SPL = 130.0            # boom capsule: 0 dBFS = 130 dB SPL (outdoor recipe)

# ---------------------------------------------------------------------------
# Source pools (raw corpora, indexed once)
# ---------------------------------------------------------------------------
MAD_LABELS = {1: "gunshot", 2: "footsteps", 3: "shelling", 4: "vehicle", 5: "helicopter", 6: "fighter"}


def build_index(cache: Path) -> dict[str, list[str]]:
    if cache.exists():
        return json.loads(cache.read_text())
    idx: dict[str, list[str]] = {}
    mad = RAW / "mad" / "archive" / "MAD_dataset"
    with (mad / "training.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lab = int(row["label"])
            if lab in MAD_LABELS:
                p = mad / row["path"]
                if lab == 2 or sf.info(str(p)).duration >= 5.0:   # short clips would loop audibly (footsteps are events)
                    idx.setdefault(f"mad_{MAD_LABELS[lab]}", []).append(str(p))
    for env in ("NFIELD", "NPARK", "NRIVER", "SPSQUARE", "STRAFFIC", "TBUS", "TCAR", "TMETRO"):
        p = RAW / "demand" / env / "ch01.wav"
        if p.exists():
            idx[f"demand_{env.lower()}"] = [str(p)]
    esc = DATA / "sources" / "noise" / "esc50" / "ESC-50-master"
    with (esc / "meta" / "esc50.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["category"] in ("wind", "rain", "helicopter", "engine", "airplane", "chainsaw", "siren",
                                   "thunderstorm", "crackling_fire", "footsteps", "door_wood_knock", "can_opening",
                                   "clapping", "glass_breaking"):
                idx.setdefault(f"esc50_{row['category']}", []).append(str(esc / "audio" / row["filename"]))
    field = DATA / "sources" / "impulsive" / "field"
    shots = {}
    with (field / "gunshot-audio-all-metadata.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            locs = [float(v) for v in row["gunshot_location_in_seconds"].strip("[]").split()]   # numpy repr
            shots[row["filename"]] = locs
    for p in sorted((field / "edge-collected-gunshot-audio").glob("*/*.wav")):
        key = p.stem.replace("_chan0", "")                # most files are <uuid>_v<k>.wav; a few have _chanN
        if "_chan" in p.stem and "_chan0" not in p.stem:
            continue
        if key in shots and shots[key]:
            idx.setdefault("field_gunshot", []).append(json.dumps([str(p), shots[key][0]]))
    idmt = RAW / "idmt_traffic" / "IDMT_Traffic" / "audio"
    for p in sorted(idmt.glob("*_D_T[LR]_ME_CH12.wav")):
        idx.setdefault("idmt_truck", []).append(str(p))
    for p in sorted(idmt.glob("*_D_C[LR]_ME_CH12.wav")):
        idx.setdefault("idmt_car", []).append(str(p))
    for p in sorted((RAW / "drone_audio" / "Binary_Drone_Audio" / "yes_drone").glob("*.wav")):
        idx.setdefault("drone", []).append(str(p))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(idx))
    return idx


# ---------------------------------------------------------------------------
# Scenarios: what a battlefield boom mic hears. bed = continuous, events =
# foreground layers with envelopes, impulses = gunfire; wind is capsule-local.
# ---------------------------------------------------------------------------
SCENARIOS = [
    dict(name="convoy_idle",    bed=("mad_vehicle", "steady"),     events=[("idmt_truck", "real_passby")],          impulses=0, wind=0.6),
    dict(name="helo_lz",        bed=("esc50_wind", "gust"),        events=[("mad_helicopter", "approach")],         impulses=0, wind=1.0),
    dict(name="firefight",      bed=("demand_nfield", "steady"),   events=[("mad_vehicle", "recede")],              impulses=3, wind=0.5),
    dict(name="artillery",      bed=("esc50_wind", "gust"),        events=[("mad_shelling", "steady"), ("mad_shelling", "steady")], impulses=0, wind=1.0),
    dict(name="jet_flyover",    bed=("demand_nfield", "steady"),   events=[("mad_fighter", "passby")],              impulses=0, wind=0.7),
    dict(name="drone_overhead", bed=("demand_npark", "steady"),    events=[("drone", "gust")],                      impulses=1, wind=0.4),
    dict(name="urban_patrol",   bed=("demand_straffic", "steady"), events=[("esc50_siren", "passby"), ("idmt_car", "real_passby")], impulses=2, wind=0.3),
    dict(name="armour_moving",  bed=("mad_vehicle", "steady"),     events=[("mad_vehicle", "passby"), ("mad_helicopter", "recede")], impulses=1, wind=0.5),
    dict(name="rain_ambush",    bed=("esc50_rain", "gust"),        events=[("esc50_thunderstorm", "gust")],         impulses=3, wind=0.6),
    dict(name="engine_room",    bed=("esc50_engine", "steady"),    events=[("esc50_chainsaw", "gust")],             impulses=0, wind=0.0),
]

# Reference levels for the outdoor recipe. Continuous: dB SPL (A-ish) RMS at
# ref distance; impulsive: dB peak at ref distance. Sources: NAP "Noise and
# Military Service" ch.3 (M16 157 dBP at shooter, M1A2 93-117 dBA interior,
# UH-60 106 dBA cockpit, 105 mm howitzer 183 dBP at gunner), IDMT-Traffic
# recording geometry, small-quadrotor measurements (~80 dBA at 1 m).
SPL_REF = {
    "mad_vehicle":     dict(spl=100.0, r0=10.0,  r=(10.0, 300.0),  ground=True,  kind="cont"),
    "idmt_truck":      dict(spl=85.0,  r0=10.0,  r=(10.0, 120.0),  ground=True,  kind="cont"),
    "idmt_car":        dict(spl=78.0,  r0=10.0,  r=(10.0, 120.0),  ground=True,  kind="cont"),
    "mad_helicopter":  dict(spl=105.0, r0=50.0,  r=(50.0, 800.0),  ground=False, kind="cont"),
    "mad_fighter":     dict(spl=120.0, r0=300.0, r=(200.0, 2000.0), ground=False, kind="cont"),
    "mad_shelling":    dict(spl=115.0, r0=100.0, r=(100.0, 3000.0), ground=True,  kind="cont"),
    "drone":           dict(spl=80.0,  r0=1.0,   r=(3.0, 60.0),    ground=False, kind="cont"),
    "esc50_helicopter": dict(spl=105.0, r0=50.0, r=(50.0, 800.0),  ground=False, kind="cont"),
    "esc50_siren":     dict(spl=110.0, r0=10.0,  r=(20.0, 500.0),  ground=True,  kind="cont"),
    "esc50_chainsaw":  dict(spl=105.0, r0=1.0,   r=(5.0, 100.0),   ground=True,  kind="cont"),
    "esc50_thunderstorm": dict(spl=95.0, r0=100.0, r=(100.0, 3000.0), ground=False, kind="cont"),
    "field_gunshot":   dict(spl=157.0, r0=1.0,   r=(5.0, 400.0),   ground=True,  kind="peak"),
}
DIFFUSE_SPL = {   # beds with no meaningful distance: ambient dB SPL range
    "demand_nfield": (50.0, 65.0), "demand_npark": (55.0, 68.0), "demand_nriver": (60.0, 72.0),
    "demand_straffic": (65.0, 78.0), "demand_spsquare": (60.0, 72.0), "demand_tbus": (70.0, 82.0),
    "demand_tcar": (68.0, 80.0), "demand_tmetro": (72.0, 85.0),
    "esc50_wind": (60.0, 85.0), "esc50_rain": (60.0, 78.0), "esc50_engine": (80.0, 100.0),
}
# Ambient level a talker at this effort would have been induced by (Lombard
# slope ~0.5 dB/dB above ~45 dBA; ISO 9921 loud 72 / very loud 78 / shout 84 dBA at 1 m;
# GRID Lombard takes were induced by 80 dB SPL SSN).
# Battlefield ambient at the ear runs 80-115 dB SPL (vehicles 93-117 dBA, UH-60 106 dBA);
# the Lombard response saturates around 85-90 dB ambient, so a "very loud" talker
# is consistent with anything from a loud engine to a tank interior.
EFFORT_AMBIENT = {"lombard": (78.0, 100.0), "loud": (72.0, 95.0), "veryloud": (88.0, 115.0)}
EFFORT_SPL_1M = {"lombard": (68.0, 76.0), "loud": (70.0, 76.0), "veryloud": (77.0, 86.0)}
MOUTH_TO_BOOM_GAIN_DB = 28.0      # 1 m -> 2.5 cm: close-talk boom levels (~95 dB SPL for normal speech), not pure inverse-square (32 dB)


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
# BS.1770 loudness now lives in the package (identical constants); kept importable here for the sheet.
from ancdata.loudness import loudness_lufs  # noqa: E402


def iso9613_alpha(f: np.ndarray, T: float = 293.15, hr: float = 50.0, pa: float = 101.325) -> np.ndarray:
    """Pure-tone atmospheric absorption, dB/m (ISO 9613-1:1993 eq. 3-5)."""
    pr, T0, T01 = 101.325, 293.15, 273.16
    psat = pr * 10 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)
    h = hr * psat / pa
    frO = pa / pr * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = pa / pr * (T / T0) ** -0.5 * (9 + 280 * h * np.exp(-4.17 * ((T / T0) ** (-1 / 3) - 1)))
    f2 = f ** 2
    return 8.686 * f2 * (1.84e-11 * (pa / pr) ** -1 * (T / T0) ** 0.5 + (T / T0) ** -2.5 * (
        0.01275 * np.exp(-2239.1 / T) / (frO + f2 / frO) + 0.1068 * np.exp(-3352.0 / T) / (frN + f2 / frN)))


def air_absorb(x: np.ndarray, r_m: float, sr: int = SR) -> np.ndarray:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    g = 10 ** (-iso9613_alpha(np.maximum(f, 1.0)) * r_m / 20)
    return np.fft.irfft(X * g, n=len(x)).astype(np.float32)


def ground_comb(x: np.ndarray, r_m: float, rng: np.random.Generator, sr: int = SR,
                h_mic: float = 1.7) -> np.ndarray:
    """Direct + one ground reflection (soft ground, |R| ~ 0.5, low-passed)."""
    h_src = float(rng.uniform(0.5, 2.0))
    d_direct = np.sqrt(r_m ** 2 + (h_mic - h_src) ** 2)
    d_refl = np.sqrt(r_m ** 2 + (h_mic + h_src) ** 2)
    delay = int(round((d_refl - d_direct) / 343.0 * sr))
    if delay < 1:
        return x
    refl = np.zeros_like(x)
    refl[delay:] = x[:-delay]
    k = np.exp(-2 * np.pi * 1500.0 / sr)
    refl = lfilter([1 - k], [1, -k], refl).astype(np.float32)
    return (x + 0.5 * (d_direct / d_refl) * refl).astype(np.float32)


def crop_at(x: np.ndarray, n: int, offset: int) -> np.ndarray:
    if len(x) < n:
        reps = int(np.ceil(n / len(x))) + 1
        x = np.tile(x, reps)
    offset = offset % (len(x) - n + 1)
    return x[offset:offset + n].astype(np.float32)


def envelope(x: np.ndarray, kind: str, p: dict, sr: int = SR) -> np.ndarray:
    n = len(x)
    rng = np.random.default_rng(int(p["env_seed"]))
    if kind == "steady":
        return x
    if kind == "gust":
        return x * _gust_envelope(n, rng, p["depth_db"], p["rate_hz"], sr)
    if kind == "approach":
        return _doppler_drift(x, rng, p["doppler_pct"]) * _ramp_envelope(n, p["ramp_db"])
    if kind == "recede":
        return _doppler_drift(x, rng, p["doppler_pct"]) * _ramp_envelope(n, -p["ramp_db"])
    if kind == "passby":
        return _doppler_drift(x, rng, p["doppler_pct"]) * _passby_envelope(n, rng, p["peak_db"])
    if kind == "real_passby":     # a 2 s real pass-by placed at a time (the clip has its own Doppler)
        n = N
        out = np.zeros(n, dtype=np.float32)
        m = min(len(x), n)
        t = int(p["t0"] * sr)
        t = min(t, n - m)
        fade = int(0.05 * sr)
        seg = x[:m].copy()
        seg[:fade] *= np.linspace(0, 1, fade)
        seg[-fade:] *= np.linspace(1, 0, fade)
        out[t:t + m] = seg
        return out
    raise ValueError(kind)


def place_impulses(speech_peak: float, shots: list[dict], n: int, gain_fn) -> np.ndarray:
    """shots: [{audio, t0, ratio_db or spl_peak}]. gain_fn(shot, peak) -> linear gain."""
    out = np.zeros(n, dtype=np.float32)
    for s in shots:
        a = s["audio"]
        g = gain_fn(s, float(np.abs(a).max()) + EPS)
        t = int(s["t0"] * SR)
        m = min(len(a), n - t)
        if m > 0:
            out[t:t + m] += a[:m] * g
    return out


# ---------------------------------------------------------------------------
# Plan: everything random, drawn once per sample
# ---------------------------------------------------------------------------
def load_gunshot(entry: str) -> np.ndarray:
    path, t_shot = json.loads(entry)
    x = load_mono(path)
    a = max(0, int((t_shot - 0.01) * SR))
    b = min(len(x), a + int(0.7 * SR))
    seg = x[a:b].copy()
    fade = int(0.05 * SR)
    seg[-fade:] *= np.linspace(1, 0, fade)
    return seg


def make_plan(k: int, seed: int, idx: dict, speech_meta: list[dict], rng_split: str = "train") -> dict:
    rng = np.random.default_rng([seed, k])
    sc = SCENARIOS[k % len(SCENARIOS)]
    cands = [m for m in speech_meta if m["split"] == rng_split and m["corpus"] == ("lombardgrid" if k % 2 == 0 else "avid")]
    sp = cands[int(rng.integers(len(cands)))]
    plan = dict(k=k, scenario=sc["name"], speech=sp["path"], speech_id=sp["id"], effort=sp["effort"],
                snr_db=float(rng.uniform(-5, 15)), speech_level_db=float(rng.uniform(-32, -18)),
                layers=[], impulses=[], wind=None, room_seed=int(rng.integers(2 ** 31)))

    def layer(pool: str, kind: str, rel_db: float) -> dict:
        files = idx[pool]
        f = files[int(rng.integers(len(files)))]
        p = dict(pool=pool, file=f, kind=kind, rel_db=rel_db, offset=int(rng.integers(0, 10 ** 7)),
                 env_seed=int(rng.integers(2 ** 31)), depth_db=float(rng.uniform(3, 10)),
                 rate_hz=float(rng.uniform(0.2, 1.0)), ramp_db=float(rng.uniform(6, 15)),
                 peak_db=float(rng.uniform(6, 12)), doppler_pct=float(rng.uniform(0.5, 2.0)),
                 t0=float(rng.uniform(0.0, SECONDS - 2.0)))
        ref = SPL_REF.get(pool)
        if ref:
            lo, hi = ref["r"]
            p["dist_m"] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            lo, hi = DIFFUSE_SPL.get(pool, (60.0, 80.0))
            p["spl_db"] = float(rng.uniform(lo, hi))
        return p

    plan["layers"].append(layer(*sc["bed"], 0.0))
    for pool, kind in sc["events"]:
        plan["layers"].append(layer(pool, kind, -float(rng.uniform(2, 12))))
    if sc["wind"] > 0 and rng.random() < sc["wind"]:
        w = layer("esc50_wind", "gust", -float(rng.uniform(0, 10)))
        w["offset_ref"] = int(rng.integers(0, 10 ** 7))
        plan["wind"] = w
    n_imp = int(rng.integers(1, sc["impulses"] + 1)) if sc["impulses"] else 0
    if n_imp:
        t = float(rng.uniform(0.3, SECONDS - 1.5))
        dist = float(np.exp(rng.uniform(np.log(5.0), np.log(400.0))))
        own = rng.random() < 0.15
        files = idx["field_gunshot"]
        for j in range(n_imp):
            plan["impulses"].append(dict(file=files[int(rng.integers(len(files)))], t0=t,
                                         ratio_db=float(rng.uniform(-6, 18)),
                                         dist_m=1.0 if own else dist, own_weapon=bool(own)))
            t += float(rng.uniform(0.09, 0.25))
    plan["ambient_target_spl"] = float(rng.uniform(*EFFORT_AMBIENT[sp["effort"]]))
    plan["speech_spl_1m"] = float(rng.uniform(*EFFORT_SPL_1M[sp["effort"]]))
    return plan


TEXTURE_POOLS = ["esc50_footsteps", "esc50_door_wood_knock", "esc50_can_opening", "mad_footsteps", "esc50_clapping"]


def make_grain_plan(plan: dict, seed: int, idx: dict) -> dict:
    """Second set of draws for the 'grain' recipes (trial 2): a rougher second
    bed, texture events, capsule buffets, gunfire in EVERY sample, blasts."""
    rng = np.random.default_rng([seed, plan["k"], 99])
    g: dict = {}
    bed_pool = plan["layers"][0]["pool"]
    files = idx[bed_pool]
    g["bed2"] = dict(pool=bed_pool, file=files[int(rng.integers(len(files)))], offset=int(rng.integers(0, 10 ** 7)),
                     env_seed=int(rng.integers(2 ** 31)), depth_db=float(rng.uniform(4, 12)),
                     rate_hz=float(rng.uniform(0.8, 3.0)), rel_db=-float(rng.uniform(3, 8)))
    g["texture"] = []
    for _ in range(int(rng.integers(3, 9))):
        pool = TEXTURE_POOLS[int(rng.integers(len(TEXTURE_POOLS)))]
        files = idx[pool]
        g["texture"].append(dict(pool=pool, file=files[int(rng.integers(len(files)))], offset=int(rng.integers(0, 10 ** 6)),
                                 dur_s=float(rng.uniform(0.3, 1.5)), t0=float(rng.uniform(0, SECONDS - 0.3)),
                                 rel_db=-float(rng.uniform(8, 20))))
    g["buffets"] = [dict(t0=float(rng.uniform(0, SECONDS - 0.4)), dur_s=float(rng.uniform(0.08, 0.4)),
                         rel_db=float(rng.uniform(-6, 6)), seed=int(rng.integers(2 ** 31)))
                    for _ in range(int(rng.integers(2, 7)))]
    g["bursts"] = []
    for j in range(int(rng.integers(1, 3))):
        auto = rng.random() < 0.6
        n_rounds = int(rng.integers(4, 10)) if auto else int(rng.integers(2, 5))
        spacing = float(rng.uniform(0.067, 0.092)) if auto else float(rng.uniform(0.15, 0.4))
        src = rng.choice(["physics", "field", "mad"], p=[0.5, 0.3, 0.2])
        b = dict(t0=float(rng.uniform(0.2, SECONDS - 1.2)), n_rounds=n_rounds, spacing_s=spacing, auto=bool(auto),
                 source=str(src), ratio_db=float(rng.uniform(0, 14)), seed=int(rng.integers(2 ** 31)),
                 tail_tau_s=float(rng.uniform(0.08, 0.25)), tail_db=-float(rng.uniform(10, 18)),
                 file=None)
        if src == "field":
            b["file"] = idx["field_gunshot"][int(rng.integers(len(idx["field_gunshot"])))]
        elif src == "mad":
            b["file"] = idx["mad_gunshot"][int(rng.integers(len(idx["mad_gunshot"])))]
        g["bursts"].append(b)
    if rng.random() < 0.6:       # a distant burst: dull, quiet, longer spacing
        g["bursts"].append(dict(t0=float(rng.uniform(0.2, SECONDS - 1.5)), n_rounds=int(rng.integers(3, 8)),
                                spacing_s=float(rng.uniform(0.08, 0.12)), auto=True, source="physics",
                                ratio_db=-float(rng.uniform(0, 12)), seed=int(rng.integers(2 ** 31)),
                                tail_tau_s=float(rng.uniform(0.15, 0.35)), tail_db=-float(rng.uniform(6, 12)),
                                file=None, distant=True))
    g["blasts"] = []
    for _ in range(int(rng.integers(0, 3))):
        src = "mad" if rng.random() < 0.5 else "physics"
        bl = dict(t0=float(rng.uniform(0.1, SECONDS - 1.0)), source=src, ratio_db=float(rng.uniform(-4, 10)),
                  seed=int(rng.integers(2 ** 31)), file=None)
        if src == "mad":
            bl["file"] = idx["mad_shelling"][int(rng.integers(len(idx["mad_shelling"])))]
        g["blasts"].append(bl)
    g["channel_seed"] = int(rng.integers(2 ** 31))
    return g


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------
class Sources:
    """Loads + crops every layer of a plan once, so recipes share identical material."""

    def __init__(self, plan: dict, snippet_root: Path):
        self.plan = plan
        sp = load_mono(snippet_root / plan["speech"])
        self.speech = sp[:N] if len(sp) >= N else np.pad(sp, (0, N - len(sp)))
        self.layers = []
        for L in plan["layers"]:
            x = trim_silence(load_mono(L["file"]))
            if L["kind"] == "real_passby":
                self.layers.append(x[:N])
            else:
                self.layers.append(crop_at(x, N, L["offset"]))
        self.wind = None
        if plan["wind"]:
            w = trim_silence(load_mono(plan["wind"]["file"]))
            self.wind = (crop_at(w, N, plan["wind"]["offset"]), crop_at(w, N, plan["wind"]["offset_ref"]))
        self.shots = [dict(audio=load_gunshot(s["file"]), **s) for s in plan["impulses"]]

    def unit_layers(self) -> list[np.ndarray]:
        """Enveloped layers at unit active RMS. A placed pass-by is normalised by
        the clip itself, not by the 6 s frame it sits in (mostly zeros)."""
        out = []
        for L, x in zip(self.plan["layers"], self.layers):
            y = envelope(x, L["kind"], L)
            ref = x if L["kind"] == "real_passby" else y
            out.append(y / (db_to_lin(active_rms_db(ref)) + EPS))
        return out


def _unit_active(x: np.ndarray) -> np.ndarray:
    return x / (db_to_lin(active_rms_db(x)) + EPS)


def recipe_naive(src: Sources) -> np.ndarray:
    p = src.plan
    s = src.speech * db_to_lin(p["speech_level_db"]) / (rms(src.speech) + EPS)
    n = src.layers[0]
    n = n * (rms(s) / (rms(n) + EPS)) * db_to_lin(-p["snr_db"])
    return dict(mix=np.clip(s + n, -1, 1), speech=s, noise=n)


def _scene_sum(src: Sources) -> tuple[np.ndarray, np.ndarray | None]:
    tot = np.zeros(N, dtype=np.float32)
    for L, y in zip(src.plan["layers"], src.unit_layers()):
        tot += y * db_to_lin(L["rel_db"])
    wind = None
    if src.wind is not None:
        w = src.plan["wind"]
        wb = envelope(src.wind[0], "gust", w)
        wind = _unit_active(wb) * db_to_lin(w["rel_db"])
    return tot, wind


def recipe_scene_lufs(src: Sources) -> np.ndarray:
    p = src.plan
    s = src.speech * db_to_lin(p["speech_level_db"] - active_rms_db(src.speech))
    tot, wind = _scene_sum(src)
    if wind is not None:
        tot = tot + wind
    # SNR on K-weighted loudness of the speech vs the noise scene
    g = db_to_lin(loudness_lufs(s) - p["snr_db"] - loudness_lufs(tot))
    noise = tot * g
    sp_peak = float(np.abs(s).max())
    imp = place_impulses(sp_peak, src.shots, N, lambda sh, pk: sp_peak * db_to_lin(sh["ratio_db"]) / pk)
    return dict(mix=np.clip(s + noise + imp, -1, 1), speech=s, noise=noise + imp)


_BANK = None


def _bank():
    global _BANK
    if _BANK is None:
        from ancdata.rir_gen import RirBank
        _BANK = RirBank()
    return _BANK


def _conv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve
    return fftconvolve(x, h)[:len(x)].astype(np.float32)


def recipe_scene_room(src: Sources) -> np.ndarray:
    p = src.plan
    rng = np.random.default_rng(p["room_seed"])
    _, rirs, meta = _bank().draw(rng, "train")
    h_sb, h_nb = rirs[0], rirs[2]          # speech->boom, noise->boom
    s_dry = src.speech * db_to_lin(p["speech_level_db"] - active_rms_db(src.speech))
    # proximity low shelf on the near-field path
    k = np.exp(-2 * np.pi * 150.0 / SR)
    low = lfilter([1 - k], [1, -k], s_dry).astype(np.float32)
    s = _conv(s_dry + (db_to_lin(float(rng.uniform(0, 8))) - 1) * low, h_sb)
    tot, wind = _scene_sum(src)
    noise = _conv(tot, h_nb)
    if wind is not None:
        noise = noise + wind                      # capsule-local, dry
    g = db_to_lin(active_rms_db(s) - p["snr_db"] - active_rms_db(noise))
    noise = noise * g
    sp_peak = float(np.abs(s).max())
    imp = place_impulses(sp_peak, [dict(sh, audio=_conv(sh["audio"], h_nb)) for sh in src.shots], N,
                         lambda sh, pk: sp_peak * db_to_lin(sh["ratio_db"]) / pk)
    head = db_to_lin(float(rng.uniform(-1, 6)))
    mix = np.clip((s + noise + imp) * head, -1, 1)
    return dict(mix=(np.round(mix * 32767) / 32767).astype(np.float32), speech=s * head, noise=(noise + imp) * head,
                room=meta["preset"])


def recipe_outdoor(src: Sources) -> np.ndarray:
    p = src.plan
    rng = np.random.default_rng(p["room_seed"] + 1)
    fs = db_to_lin(-FULL_SCALE_SPL)                       # Pa-equivalent -> FS: 1.0 = 130 dB SPL
    spl_boom = p["speech_spl_1m"] + MOUTH_TO_BOOM_GAIN_DB
    s = _unit_active(src.speech) * db_to_lin(spl_boom) * fs
    total = np.zeros(N, dtype=np.float32)
    # the first point source is placed at the distance that puts it at the effort-coupled
    # ambient level; diffuse beds sit 6 dB under it; the other sources keep their drawn distance
    solved = False
    for i, (L, y) in enumerate(zip(p["layers"], src.unit_layers())):
        ref = SPL_REF.get(L["pool"])
        dominant = i == 0
        if ref:
            r = L["dist_m"]
            if not solved:
                r = float(np.clip(ref["r0"] * db_to_lin(ref["spl"] - p["ambient_target_spl"]), *ref["r"]))
                solved = dominant = True
            L["dist_used_m"] = r
            spl_mic = ref["spl"] - 20 * np.log10(r / ref["r0"])      # spherical spreading beyond r0
            y = air_absorb(y, max(0.0, r - ref["r0"]))              # extra path only: a low-pass, no re-normalise
            if ref["ground"]:
                y = ground_comb(y, r, rng)
        else:
            spl_mic = L["spl_db"] if i else max(L["spl_db"], p["ambient_target_spl"] - 6.0)
        L["spl_at_mic_db"] = float(spl_mic + (0.0 if dominant else L["rel_db"]))
        total += y * db_to_lin(L["spl_at_mic_db"]) * fs
    if src.wind is not None:
        w = p["wind"]
        wb = _unit_active(envelope(src.wind[0], "gust", w))
        total += wb * db_to_lin(p["ambient_target_spl"] - 3.0 + w["rel_db"]) * fs
    shots = []
    for sh in src.shots:
        r = sh["dist_m"]
        a = sh["audio"]
        if r > 1.0:
            a = ground_comb(air_absorb(a, r - 1.0), r, rng)
        shots.append(dict(sh, audio=a, spl_peak=SPL_REF["field_gunshot"]["spl"] - 20 * np.log10(r)))
    imp = place_impulses(1.0, shots, N, lambda sh, pk: db_to_lin(sh["spl_peak"]) * fs / pk)
    mix = np.clip(s + total + imp, -1, 1)
    return dict(mix=(np.round(mix * 32767) / 32767).astype(np.float32), speech=s, noise=total + imp)


# ---------------------------------------------------------------------------
# Trial 2: "grain" recipes = scene_lufs + rougher beds + texture + gunfire everywhere + channel
# ---------------------------------------------------------------------------
def _biquad_peaking(fc: float, gain_db: float, q: float, sr: int):
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b = np.array([1 + alpha * a, -2 * cw, 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * cw, 1 - alpha / a])
    return b / den[0], den / den[0]


def _lowpass1(x: np.ndarray, fc: float, sr: int = SR) -> np.ndarray:
    k = np.exp(-2 * np.pi * fc / sr)
    return lfilter([1 - k], [1, -k], x).astype(np.float32)


def outdoor_tail(x: np.ndarray, tau_s: float, level_db: float, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Cheap landscape reverb for a transient: the shot plus a decaying, low-passed
    noise tail (tree lines, buildings, ground scatter). Gunfire heard outdoors is
    never a bare impulse."""
    from scipy.signal import fftconvolve
    n = int(3 * tau_s * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n).astype(np.float32) * np.exp(-t / tau_s)
    tail = _lowpass1(tail, 2500.0, sr)
    tail *= db_to_lin(level_db) / (np.abs(tail).max() + EPS)
    ir = np.zeros(n + 1, dtype=np.float32)
    ir[0] = 1.0
    ir[1:] += tail
    return fftconvolve(x, ir)[: len(x) + n].astype(np.float32)


def render_burst(b: dict, rng: np.random.Generator) -> np.ndarray:
    """One burst of gunfire, peak-normalised, with per-round gain jitter and a tail."""
    from ancdata.physics import synth_blast
    if b["source"] == "physics":
        params = dict(standoff_m={"dist": "loguniform", "low": 40.0, "high": 400.0} if b.get("distant") else
                      {"dist": "loguniform", "low": 5.0, "high": 80.0},
                      T_ms={"dist": "uniform", "low": 0.4, "high": 2.0},
                      burst={"dist": "uniform_int", "low": b["n_rounds"], "high": b["n_rounds"]},
                      cyclic_ms={"dist": "uniform", "low": b["spacing_s"] * 1000, "high": b["spacing_s"] * 1000},
                      supersonic_p=0.7)
        wave, _ = synth_blast(rng, params)
    elif b["source"] == "mad":
        # MAD gunfire clips are already bursts with their environment; take 1.5 s from the loudest point
        one = trim_silence(load_mono(b["file"]))
        pk = int(np.argmax(np.abs(one)))
        a0 = max(0, pk - int(0.02 * SR))
        wave = one[a0: a0 + int(1.5 * SR)].copy()
        fade = min(int(0.1 * SR), len(wave))
        wave[-fade:] *= np.linspace(1, 0, fade)
    else:
        one = load_gunshot(b["file"])
        step = int(b["spacing_s"] * SR)
        wave = np.zeros(step * (b["n_rounds"] - 1) + len(one) + step, dtype=np.float32)
        for r in range(b["n_rounds"]):
            s0 = max(0, r * step + (int(rng.integers(-step // 20, step // 20 + 1)) if r else 0))
            wave[s0: s0 + len(one)] += one * float(rng.uniform(0.7, 1.0))
    if b.get("distant"):
        wave = _lowpass1(_lowpass1(wave, 1800.0), 1800.0)
    wave = outdoor_tail(wave, b["tail_tau_s"], b["tail_db"], rng)
    return wave / (np.abs(wave).max() + EPS)


def render_blast(bl: dict, rng: np.random.Generator) -> np.ndarray:
    from ancdata.physics import synth_blast
    if bl["source"] == "physics":
        wave, _ = synth_blast(rng, dict(standoff_m={"dist": "loguniform", "low": 100.0, "high": 600.0},
                                        T_ms={"dist": "uniform", "low": 2.5, "high": 4.0}, supersonic_p=0.0))
        wave = outdoor_tail(wave, float(rng.uniform(0.3, 0.7)), -float(rng.uniform(4, 10)), rng)
    else:
        x = trim_silence(load_mono(bl["file"]))
        pk = int(np.argmax(np.abs(x)))
        a0 = max(0, pk - int(0.03 * SR))
        wave = x[a0: a0 + int(1.5 * SR)].copy()
        fade = int(0.2 * SR)
        if len(wave) > fade:
            wave[-fade:] *= np.linspace(1, 0, fade)
    return wave / (np.abs(wave).max() + EPS)


def _place(out: np.ndarray, x: np.ndarray, t0: float) -> None:
    t = int(t0 * SR)
    m = min(len(x), len(out) - t)
    if m > 0:
        out[t: t + m] += x[:m]


def grain_noise(src: Sources, g: dict) -> np.ndarray:
    """Continuous scene: scene_lufs (bed + events + wind) plus a second, rougher
    bed, short texture events and low-frequency capsule buffets."""
    tot, wind = _scene_sum(src)
    b2 = g["bed2"]
    x2 = crop_at(trim_silence(load_mono(b2["file"])), N, b2["offset"])
    x2 = envelope(x2, "gust", b2) / (db_to_lin(active_rms_db(x2)) + EPS)
    tot = tot + x2 * db_to_lin(b2["rel_db"])
    for tx in g["texture"]:
        x = trim_silence(load_mono(tx["file"]))
        n = int(tx["dur_s"] * SR)
        seg = crop_at(x, n, tx["offset"]).copy()
        fade = min(int(0.02 * SR), len(seg) // 2)
        seg[:fade] *= np.linspace(0, 1, fade)
        seg[-fade:] *= np.linspace(1, 0, fade)
        seg = seg / (db_to_lin(active_rms_db(seg)) + EPS) * db_to_lin(tx["rel_db"])
        _place(tot, seg, tx["t0"])
    buf = np.zeros(N, dtype=np.float32)
    for bf in g["buffets"]:
        r = np.random.default_rng(bf["seed"])
        n = int(bf["dur_s"] * SR)
        e = np.hanning(n).astype(np.float32) ** 0.5
        seg = _lowpass1(_lowpass1(r.standard_normal(n).astype(np.float32), 150.0), 150.0) * e
        seg = seg / (rms(seg) + EPS) * db_to_lin(bf["rel_db"])
        _place(buf, seg, bf["t0"])
    return tot + (wind + buf if wind is not None else buf * 0.5)


def grain_impulses(g: dict, sp_peak: float) -> np.ndarray:
    imp = np.zeros(N, dtype=np.float32)
    for b in g["bursts"]:
        w = render_burst(b, np.random.default_rng(b["seed"]))
        _place(imp, w * sp_peak * db_to_lin(b["ratio_db"]), b["t0"])
    for bl in g["blasts"]:
        w = render_blast(bl, np.random.default_rng(bl["seed"]))
        _place(imp, w * sp_peak * db_to_lin(bl["ratio_db"]), bl["t0"])
    return imp


def channel(x: np.ndarray, rng: np.random.Generator, radio: bool = False) -> np.ndarray:
    """Everything the capsule hears goes through the same electronics: boom-mic EQ,
    self-noise, soft saturation, an AGC/limiter that ducks the whole channel after a
    shot, hard clip, 16-bit. Shared non-linearity is what fuses speech and noise."""
    from scipy.signal import butter, sosfilt
    from ancdata.physics import _pink
    y = sosfilt(butter(2, 90.0, "hp", fs=SR, output="sos"), x).astype(np.float32)
    b, a = _biquad_peaking(3000.0, 3.0, 1.0, SR)
    y = lfilter(b, a, y).astype(np.float32)
    y = sosfilt(butter(2, 6800.0, "lp", fs=SR, output="sos"), y).astype(np.float32)
    floor = _pink(N, rng).astype(np.float32)
    floor = floor / (rms(floor) + EPS) * db_to_lin(-50.0 if radio else -55.0)
    y = y + floor
    drive = 1.4
    y = (np.tanh(y * drive) / np.tanh(drive)).astype(np.float32)          # soft saturation, mild below -6 dBFS
    # AGC / limiter: instantaneous attack, ~250 ms release, threshold -8 dBFS
    thr = db_to_lin(-8.0)
    rel = np.exp(-1.0 / (0.25 * SR))
    ay = np.abs(y)
    env = np.empty(N, dtype=np.float32)
    e = 0.0
    for i in range(N):
        e = ay[i] if ay[i] > e else e * rel
        env[i] = e
    gain = np.minimum(1.0, thr / (env + EPS)).astype(np.float32)
    gain = _lowpass1(gain, 800.0)                                        # ~0.2 ms smoothing, no zipper
    y = y * gain * db_to_lin(6.0)                                        # make-up
    if radio:
        from ancdata.adc import radio_codec
        y = radio_codec(y, rng, band_hz=(250.0, 4000.0), dropout_p=0.01)
        crackle = np.zeros(N, dtype=np.float32)
        for t in rng.integers(0, N, int(rng.integers(20, 80))):
            crackle[t] = float(rng.uniform(-1, 1)) * db_to_lin(-28.0)
        y = y + _lowpass1(crackle, 4000.0)
    y = np.clip(y, -1, 1)
    return (np.round(y * 32767) / 32767).astype(np.float32)


def _grain_base(src: Sources, g: dict):
    p = src.plan
    s = src.speech * db_to_lin(p["speech_level_db"] - active_rms_db(src.speech))
    tot = grain_noise(src, g)
    gn = db_to_lin(loudness_lufs(s) - p["snr_db"] - loudness_lufs(tot))
    noise = tot * gn
    imp = grain_impulses(g, float(np.abs(s).max()))
    return s, noise, imp


def recipe_grain1(src: Sources, g: dict) -> dict:
    s, noise, imp = _grain_base(src, g)
    return dict(mix=np.clip(s + noise + imp, -1, 1), speech=s, noise=noise + imp, cont=noise)


def recipe_grain2(src: Sources, g: dict) -> dict:
    s, noise, imp = _grain_base(src, g)
    mix = channel(s + noise + imp, np.random.default_rng(g["channel_seed"]))
    return dict(mix=mix, speech=s, noise=noise + imp, cont=noise)


def recipe_grain3(src: Sources, g: dict) -> dict:
    s, noise, imp = _grain_base(src, g)
    mix = channel(s + noise + imp, np.random.default_rng(g["channel_seed"]), radio=True)
    return dict(mix=mix, speech=s, noise=noise + imp, cont=noise)


RECIPES = {"naive": recipe_naive, "scene_lufs": recipe_scene_lufs, "scene_room": recipe_scene_room,
           "outdoor": recipe_outdoor}
GRAIN_RECIPES = {"grain1": recipe_grain1, "grain2": recipe_grain2, "grain3": recipe_grain3}


# ---------------------------------------------------------------------------
# Listening sheet
# ---------------------------------------------------------------------------
def write_sheet(out: Path, plans: list[dict], codes: dict[str, str], embed: bool = False,
                anchor: str | None = None, title: str = "Noise layering trial") -> None:
    import base64

    def src(rel: str) -> str:
        if not embed:
            return rel
        return "data:audio/wav;base64," + base64.b64encode((out / rel).read_bytes()).decode("ascii")

    order = sorted(codes)                                    # blind code order
    cols = ([anchor] if anchor else []) + order
    rows = []
    for p in plans:
        k = p["k"]
        cells = "".join(
            f'<td><audio controls preload="none" src="{src(f"mix/{codes.get(c, c)}/{k:02d}.wav")}"></audio></td>' for c in cols)
        rows.append(f'<tr><td class="k">{k:02d}</td><td class="sc">{p["scenario"]}<br>'
                    f'<small>{p["speech_id"]} · {p["effort"]}</small></td>'
                    f'<td><audio controls preload="none" src="{src(f"speech/{k:02d}.wav")}"></audio></td>{cells}</tr>')
    heads = "".join(f"<th>{c}</th>" for c in cols)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}}
table{{border-collapse:collapse}}td,th{{padding:6px 10px;border-bottom:1px solid #ddd;vertical-align:middle}}
th{{text-align:left;background:#eee}}td.k{{font-weight:600}}td.sc{{min-width:160px}}audio{{width:230px}}
p{{max-width:900px}}</style></head><body>
<h2>Which column sounds like a real battlefield boom-mic recording?</h2>
<p>Same speech, same noise material, same scenario in every row; only the layering recipe differs between
R1-R4 (blind; mapping in <code>key.json</code>). Listen at a fixed, moderate volume. Judge realism, not
pleasantness: does the noise sit <em>around</em> the talker like a real capture, are levels believable, do
gunshots sound like gunshots, does anything sound pasted on. Then reply with a ranking per row or overall.</p>
<table><tr><th>#</th><th>scenario</th><th>dry speech</th>{heads}</tr>{''.join(rows)}</table>
</body></html>"""
    (out / ("listen_embedded.html" if embed else "listen.html")).write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "layering_trial")
    ap.add_argument("--snippets", type=Path, default=DATA / "snippets" / "lombard6s")
    ap.add_argument("--recipes", nargs="*", default=list(RECIPES))
    ap.add_argument("--lufs", type=float, default=-23.0, help="playback loudness of the listening copies")
    ap.add_argument("--grain", action="store_true", help="trial 2: scene_lufs anchor + grain1-3 (blind)")
    a = ap.parse_args()
    if a.grain:
        a.recipes = list(GRAIN_RECIPES)

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    idx = build_index(out / "index.json")
    print({k: len(v) for k, v in idx.items()})
    meta = [json.loads(l) for l in (a.snippets / "meta.jsonl").open(encoding="utf-8")]
    plans = [make_plan(k, a.seed, idx, meta) for k in range(a.n)]
    rng = np.random.default_rng(a.seed)
    names = list(a.recipes)
    perm = rng.permutation(len(names))
    codes = {f"{'S' if a.grain else 'R'}{i + 1}": names[j] for i, j in enumerate(perm)}
    (out / "key.json").write_text(json.dumps(codes, indent=1), encoding="utf-8")

    summary = []
    for p in plans:
        k = p["k"]
        src = Sources(p, a.snippets)
        write_wav(out / "speech" / f"{k:02d}.wav", src.speech * db_to_lin(p["speech_level_db"] - active_rms_db(src.speech)), fmt="pcm16")
        line = [f"{k:02d} {p['scenario']:<15} {p['effort']:<8} snr {p['snr_db']:5.1f}"]
        g = make_grain_plan(p, a.seed, idx) if a.grain else None
        if a.grain:
            p["grain"] = g
            r = RECIPES["scene_lufs"](src)
            write_wav(out / "mix_raw" / "scene_lufs" / f"{k:02d}.wav", r["mix"], fmt="pcm16")
            gg = min(db_to_lin(a.lufs - loudness_lufs(r["mix"])), 0.99 / (float(np.abs(r["mix"]).max()) + EPS))
            write_wav(out / "mix" / "scene_lufs" / f"{k:02d}.wav", r["mix"] * gg, fmt="pcm16")
        for name in names:
            r = GRAIN_RECIPES[name](src, g) if a.grain else RECIPES[name](src)
            y = r["mix"]
            write_wav(out / "mix_raw" / name / f"{k:02d}.wav", y, fmt="pcm16")
            # listening copy: loudness-matched (-23 LUFS, peak-safe) so the vote is about
            # realism, not about which column happens to be louder. SNR and clipping are untouched.
            gl = min(db_to_lin(a.lufs - loudness_lufs(y)), 0.99 / (float(np.abs(y).max()) + EPS))
            write_wav(out / "mix" / name / f"{k:02d}.wav", y * gl, fmt="pcm16")
            eff = active_rms_db(r["speech"]) - active_rms_db(r.get("cont", r["noise"]))   # continuous noise; transients are peak-ratio events
            clip = float(np.mean(np.abs(y) >= 0.999) * 100)
            summary.append(dict(k=k, recipe=name, scenario=p["scenario"], effort=p["effort"], snr_planned=p["snr_db"],
                                snr_effective_db=round(float(eff), 1), rms_dbfs=round(float(20 * np.log10(rms(y))), 1),
                                clip_pct=round(clip, 2), lufs=round(loudness_lufs(y), 1), room=r.get("room")))
            line.append(f"{name}: snr {eff:5.1f} rms {20 * np.log10(rms(y)):5.1f} clip {clip:.2f}%")
        print(" | ".join(line))
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "plan.json").write_text(json.dumps(plans, indent=1), encoding="utf-8")   # includes dist_used_m
    anchor = "scene_lufs" if a.grain else None
    title = "Noise layering trial 2: grain" if a.grain else "Noise layering trial"
    write_sheet(out, plans, codes, anchor=anchor, title=title)
    write_sheet(out, plans, codes, embed=True, anchor=anchor, title=title)
    print(f"-> {out / 'listen.html'}  (self-contained copy: listen_embedded.html)")


if __name__ == "__main__":
    main()
