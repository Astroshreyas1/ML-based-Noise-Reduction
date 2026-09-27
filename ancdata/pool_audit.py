"""Content audit of the noise pools with an AudioSet tagger (PANNs Cnn14, Kong et al. 2020).

Why: a pool is only as good as its worst file. YouTube (MAD) and Freesound (FSD50K)
clips carry music, speech and unrelated sounds under a "gunshot" or "helicopter"
folder name, and a clip that sounds like X but is labelled Y teaches the detector
the wrong thing and makes the scene sound wrong. Every pooled file is tagged; a
file stays usable only if one of its pool's expected AudioSet classes is present
and it is not dominated by music (or by speech, outside the interferer pools).

    ancdata pool-audit               -> data/pools_audit.parquet (+ outputs/pool_audit.md)

`Pools(..., require_audit=True)` then draws only audited files. DEMAND recordings
(long, curated ambiences) are passed without tagging.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from .paths import data_root, from_relative

TAG_SR = 32000

VEHICLE = ["Vehicle", "Engine", "Idling", "Truck", "Bus", "Car", "Motor vehicle (road)", "Heavy engine (low frequency)",
           "Medium engine (mid frequency)", "Light engine (high frequency)", "Engine starting", "Accelerating, revving, vroom",
           "Tractor", "Traffic noise, roadway noise", "Car passing by", "Motorcycle", "Rail transport", "Train",
           "Diesel", "Air brake", "Tire squeal", "Skidding"]
AIR = ["Aircraft", "Fixed-wing aircraft, airplane", "Jet engine", "Aircraft engine", "Propeller, airscrew", "Helicopter"]
GUN = ["Gunshot, gunfire", "Machine gun", "Fusillade", "Cap gun", "Artillery fire"]
BOOM = ["Explosion", "Boom", "Artillery fire", "Gunshot, gunfire", "Fireworks", "Burst, pop", "Eruption", "Thunder", "Firecracker"]
SPEECHY = ["Speech", "Shout", "Yell", "Bellow", "Crowd", "Chatter", "Conversation", "Male speech, man speaking",
           "Female speech, woman speaking", "Narration, monologue", "Children shouting", "Hubbub, speech noise, speech babble",
           "Whispering", "Radio"]
EXPECTED: dict[str, list[str]] = {
    "wind": ["Wind", "Wind noise (microphone)", "Rustling leaves", "Howl"],
    "rain": ["Rain", "Raindrop", "Rain on surface", "Thunderstorm"],
    "helicopter": ["Helicopter", "Aircraft", "Propeller, airscrew"],
    "fighter": AIR, "aircraft": AIR, "airplane": AIR,
    "drone": ["Propeller, airscrew", "Buzz", "Insect", "Mosquito", "Fly, housefly", "Mechanical fan", "Aircraft", "Hum",
              "Bee, wasp, etc."],
    "siren": ["Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)",
              "Emergency vehicle", "Alarm"],
    "thunder": ["Thunder", "Thunderstorm", "Rain"],
    "gun": GUN, "shelling": BOOM, "explosion": BOOM, "fireworks": BOOM,
    "footsteps": ["Walk, footsteps", "Run", "Shuffle", "Crunch", "Rustle"],
    "gear": ["Zipper (clothing)", "Keys jangling", "Rattle", "Walk, footsteps", "Clicking", "Jingle, tinkle", "Rustle",
             "Chink, clink", "Mechanisms", "Velcro, hook and loop fastener"],
    "fire": ["Fire", "Crackle", "Fire alarm"],
    "debris": ["Shatter", "Glass", "Breaking", "Smash, crash", "Crushing", "Chink, clink", "Crumpling, crinkling"],
    "glass": ["Glass", "Shatter", "Breaking", "Smash, crash", "Chink, clink"],
    "chainsaw": ["Chainsaw", "Power tool", "Engine"],
    "jackhammer": ["Jackhammer", "Power tool", "Drill", "Hammer"],
    "fan": ["Mechanical fan", "Air conditioning", "Hum", "Engine", "Idling", "Mechanisms"],
    "traffic": ["Traffic noise, roadway noise", "Vehicle", "Car", "Car passing by", "Outside, urban or manmade"],
    "water": ["Stream", "Waves, surf", "Ocean", "Water", "Gurgling", "Trickle, dribble", "Waterfall"],
    "nature": ["Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet", "Insect", "Cricket", "Crow",
               "Outside, rural or natural", "Wild animals"],
    "hardneg": ["Knock", "Door", "Slam", "Tap", "Thump, thud", "Wood", "Hammer", "Bang", "Clicking", "Walk, footsteps",
                "Drum", "Wood block", "Clapping", "Hands", "Crack", "Whack, thwack"],
    "interferer": SPEECHY,
    "vehicle": VEHICLE,
}


def expected_for(pool: str) -> list[str] | None:
    if pool.startswith("demand_"):
        return None
    if pool in ("fsd_interferer", "mad_communication"):
        return EXPECTED["interferer"]
    if pool == "esc50_train":
        return EXPECTED["vehicle"]
    for key in ("helicopter", "fighter", "aircraft", "airplane", "drone", "siren", "thunder", "shelling", "explosion",
                "fireworks", "footsteps", "gear", "fire", "debris", "glass", "chainsaw", "jackhammer", "fan", "traffic",
                "water", "nature", "hardneg", "wind", "rain"):
        if key in pool:
            return EXPECTED[key]
    if "gun" in pool:
        return EXPECTED["gun"]
    if pool in ("esc50_door_wood_knock", "esc50_can_opening", "esc50_clapping"):
        return EXPECTED["hardneg"]
    if any(k in pool for k in ("engine", "idling", "vehicle", "truck", "bus", "car", "motorcycle", "train")):
        return EXPECTED["vehicle"] + (["Car horn, honking", "Vehicle horn, car horn, honking"] if "horn" in pool else [])
    if pool == "esc50_sea_waves":
        return EXPECTED["water"]
    return None


def _load(rel: str, start_s: float, end_s: float, max_windows: int = 3, win_s: float = 10.0) -> np.ndarray:
    p = from_relative(rel)
    info = sf.info(str(p))
    a, b = int(start_s * info.samplerate), int(end_s * info.samplerate) if end_s > 0 else info.frames
    x, sr = sf.read(str(p), dtype="float32", always_2d=True, start=a, stop=min(b, a + int(90 * info.samplerate)))
    x = x.mean(1)
    if sr != TAG_SR:
        g = np.gcd(int(sr), TAG_SR)
        x = resample_poly(x, TAG_SR // g, sr // g).astype(np.float32)
    w = int(win_s * TAG_SR)
    if len(x) <= w:
        return np.pad(x, (0, max(0, TAG_SR - len(x))))[None]
    starts = np.linspace(0, len(x) - w, min(max_windows, int(np.ceil(len(x) / w)))).astype(int)
    return np.stack([x[s: s + w] for s in starts])


def run_audit(batch: int = 16, workers: int = 8, verbose: bool = True) -> pd.DataFrame:
    import torch
    from panns_inference import AudioTagging, labels

    df = pd.read_parquet(data_root() / "pools.parquet")
    df = df[~df.has_voice]
    df["expected"] = df.pool.map(lambda p: expected_for(p))
    todo = df[df.expected.notna()].drop_duplicates(["path", "start_s", "end_s"])
    names = list(labels)
    idx = {n: i for i, n in enumerate(names)}
    speech_i = [idx[n] for n in SPEECHY if n in idx]
    music_i = [i for i, n in enumerate(names) if n in ("Music", "Musical instrument", "Singing", "Song", "Pop music",
                                                       "Rock music", "Electronic music", "Soundtrack music",
                                                       "Background music", "Theme music", "Video game music")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    at = AudioTagging(checkpoint_path=str(Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"), device=device)
    rows = list(todo[["path", "start_s", "end_s"]].itertuples(index=False))
    probs: dict[tuple, np.ndarray] = {}
    from tqdm import tqdm
    with ThreadPoolExecutor(workers) as ex:
        for i in tqdm(range(0, len(rows), batch), disable=not verbose, desc="tagging"):
            chunk = rows[i: i + batch]
            wavs = list(ex.map(lambda r: _load(r.path, float(r.start_s), float(r.end_s)), chunk))
            for w, r in zip(wavs, chunk):          # one file per call: windows share a length, no zero padding
                with torch.no_grad():
                    clip, _ = at.inference(w)
                probs[(r.path, float(r.start_s), float(r.end_s))] = clip.max(0)
    out = []
    for r in df[df.expected.notna()].itertuples(index=False):
        pr = probs[(r.path, float(r.start_s), float(r.end_s))]
        exp_i = [idx[n] for n in r.expected if n in idx]
        exp_score = float(pr[exp_i].max()) if exp_i else 0.0
        top5 = np.argsort(pr)[::-1][:5]
        speech, music = float(pr[speech_i].max()), float(pr[music_i].max())
        interferer = r.pool in ("fsd_interferer", "mad_communication")
        ok = (exp_score >= 0.15 or (exp_score >= 0.05 and any(t in exp_i for t in top5))) and music < 0.3 \
            and (interferer or speech < 0.4)
        out.append({"path": r.path, "pool": r.pool, "start_s": r.start_s, "end_s": r.end_s, "expected_score": exp_score,
                    "speech": speech, "music": music, "top1": names[int(top5[0])], "top1_p": float(pr[top5[0]]),
                    "audit_ok": bool(ok)})
    res = pd.DataFrame(out)
    res.to_parquet(data_root() / "pools_audit.parquet", index=False)
    if verbose:
        summ = res.groupby("pool").agg(n=("audit_ok", "size"), kept=("audit_ok", "mean"),
                                       music=("music", lambda s: float((s >= 0.3).mean())),
                                       speech=("speech", lambda s: float((s >= 0.4).mean())))
        lines = ["# Pool content audit (PANNs Cnn14)", "", summ.round(3).to_markdown(), "",
                 "## Most common top-1 tags of rejected files per pool", ""]
        for pool, g in res[~res.audit_ok].groupby("pool"):
            lines.append(f"- **{pool}** ({len(g)}): " + ", ".join(f"{k} {v}" for k, v in g.top1.value_counts().head(6).items()))
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/pool_audit.md").write_text("\n".join(lines), encoding="utf-8")
        print(summ.round(3).to_string())
    return res
