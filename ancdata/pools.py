"""Noise pools for the battlefield chain: every raw corpus indexed once, with a
permanent split per file, a voice screen, and a memory-bounded decode cache.

    ancdata pools                 -> data/pools.parquet (+ data/pools_voice_flagged.txt)
    Pools("train").draw_audio("mad_helicopter", rng) -> (audio 16 kHz mono, row)

Split rules (leakage is by *recording*, never by clip):
    ESC-50          official folds 1-3 / 4 / 5
    MAD             hashed by YouTube video id 90/5/5 (MAD's own test folder shares videos with training)
    DEMAND          one 5-min recording per environment, split by TIME: 0-210 s train, 210-255 val, 255-300 test
    FSD50K          FSD50K dev split column (train/val) as is; eval set -> test
    UrbanSound8K    folds 1-8 / 9 / 10
    IDMT-Traffic    hashed by pass-by id (sample position in the long-term recording)
    Drone, Zenodo gunshots   hashed by recording id (all channels / versions of a shot share it)

Voice screen (``registry.has_voice``: WebRTC VAD *and* F0-band periodicity) on the
YouTube / Freesound-derived corpora (MAD, FSD50K, UrbanSound8K). FSD50K clips
tagged with any human-voice class are dropped before the screen. A voiced
noise clip would teach the model to delete speech; flagged files are listed,
never silently kept.

Pool names are what ``configs/battlefield.yaml`` refers to. FSD50K classes are
grouped into pools that match the battlefield categories (Section
docs/BATTLEFIELD_SOURCES.md).
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from .audio import load_mono, trim_silence
from .config import SR
from .paths import data_root, from_relative, sources_dir, to_relative
from .registry import assign_split, esc50_fold_split, has_voice

# --------------------------------------------------------------------------
# What goes into which pool
# --------------------------------------------------------------------------
ESC50_POOLS = {"wind", "rain", "helicopter", "engine", "airplane", "chainsaw", "siren", "thunderstorm",
               "crackling_fire", "train", "sea_waves", "car_horn", "footsteps", "door_wood_knock",
               "can_opening", "clapping", "glass_breaking", "fireworks"}
MAD_CLASSES = {0: "communication", 1: "gunshot", 2: "footsteps", 3: "shelling", 4: "vehicle",
               5: "helicopter", 6: "fighter"}
MAD_MIN_DUR = {"communication": 1.0, "gunshot": 1.0, "footsteps": 0.3, "shelling": 4.0, "vehicle": 5.0, "helicopter": 5.0, "fighter": 5.0}
DEMAND_ENVS = ("NFIELD", "NPARK", "NRIVER", "SPSQUARE", "STRAFFIC", "TBUS", "TCAR", "TMETRO")
DEMAND_SEGMENTS = {"train": (0.0, 210.0), "val": (210.0, 255.0), "test": (255.0, 300.0)}
FSD_POOLS: dict[str, set[str]] = {          # FSD50K's 200-class vocabulary: no Helicopter / Jet_engine / Machine_gun
    "fsd_wind": {"Wind"},
    "fsd_rain": {"Rain", "Raindrop"},
    "fsd_aircraft": {"Aircraft", "Fixed-wing_aircraft_and_airplane"},
    "fsd_engine": {"Engine", "Truck", "Bus", "Motor_vehicle_(road)", "Idling", "Accelerating_and_revving_and_vroom",
                   "Engine_starting"},
    "fsd_explosion": {"Explosion", "Boom"},
    "fsd_gunshot": {"Gunshot_and_gunfire"},
    "fsd_siren": {"Siren"},
    "fsd_thunder": {"Thunder", "Thunderstorm"},
    "fsd_hardneg": {"Slam", "Knock", "Hammer", "Walk_and_footsteps", "Thump_and_thud", "Crack"},
    # v4 additions (docs/NOISE_LAYERING.md section 8): texture, follow-on and bed variety
    "fsd_fire": {"Fire", "Crackle"},
    "fsd_debris": {"Shatter", "Crushing"},
    "fsd_gear": {"Zipper_(clothing)", "Keys_jangling", "Rattle", "Walk_and_footsteps"},
    "fsd_fan": {"Mechanical_fan"},
    "fsd_traffic": {"Traffic_noise_and_roadway_noise"},
    "fsd_water": {"Stream", "Waves_and_surf", "Ocean"},
    "fsd_nature": {"Insect", "Cricket", "Bird_vocalization_and_bird_call_and_bird_song", "Chirp_and_tweet", "Crow"},
    "fsd_fireworks": {"Fireworks"},
}
# Interfering talkers (shouts, crowd, chatter): voiced ON PURPOSE, labelled `speech_interferer`,
# never voice-screened and never a target. Music / singing / screaming tagged clips stay out.
FSD_INTERFERER = {"fsd_interferer": {"Shout", "Yell", "Crowd", "Chatter", "Conversation"}}
FSD_INTERFERER_EXCLUDE = {"Music", "Singing", "Male_singing", "Female_singing", "Musical_instrument", "Speech_synthesizer",
                          "Laughter", "Giggle", "Screaming", "Crying_and_sobbing"}
INTERFERER_POOLS = {"fsd_interferer", "mad_communication"}
FSD_VOICE = {"Speech", "Human_voice", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking",
             "Child_speech_and_kid_speaking", "Conversation", "Shout", "Yell", "Screaming", "Crowd", "Chatter",
             "Cheering", "Singing", "Music", "Laughter", "Human_group_actions", "Whispering",
             "Chewing_and_mastication", "Cough", "Crying_and_sobbing", "Giggle", "Chuckle_and_chortle",
             "Male_singing", "Female_singing", "Applause", "Speech_synthesizer"}
FSD_MIN_DUR = {"fsd_wind": 3.0, "fsd_rain": 3.0, "fsd_aircraft": 3.0, "fsd_engine": 3.0,
               "fsd_explosion": 0.3, "fsd_gunshot": 0.3, "fsd_siren": 2.0, "fsd_thunder": 2.0, "fsd_hardneg": 0.2,
               "fsd_fire": 3.0, "fsd_debris": 0.3, "fsd_gear": 0.3, "fsd_fan": 3.0, "fsd_traffic": 4.0, "fsd_water": 4.0,
               "fsd_nature": 4.0, "fsd_fireworks": 0.3, "fsd_interferer": 1.0}
US8K_POOLS = {"gun_shot", "engine_idling", "siren", "jackhammer"}
SCREENED_CORPORA = {"mad", "fsd50k", "urbansound8k"}
LICENCE = {"esc50": "CC BY-NC 3.0", "mad": "CC BY 4.0", "demand": "CC BY-SA 4.0", "fsd50k": "per-clip CC (FSD50K.metadata)",
           "urbansound8k": "CC BY-NC 4.0", "idmt": "CC BY 4.0", "drone": "research (GitHub)", "field_gunshot": "Zenodo record"}


@dataclass
class Row:
    path: str
    pool: str
    corpus: str
    category: str
    duration_s: float
    sr: int
    split: str
    group: str
    start_s: float
    end_s: float
    shot_time: float
    licence: str
    pp: bool = False          # FSD50K "present and predominant" by every rater (an isolated event)


def _h(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _info(p: Path) -> tuple[float, int]:
    i = sf.info(str(p))
    return float(i.frames) / float(i.samplerate), int(i.samplerate)


# --------------------------------------------------------------------------
# Indexers, one per corpus
# --------------------------------------------------------------------------
def _index_esc50() -> list[Row]:
    root = sources_dir() / "noise" / "esc50" / "ESC-50-master"
    if not root.exists():
        return []
    rows = []
    with (root / "meta" / "esc50.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["category"] not in ESC50_POOLS:
                continue
            p = root / "audio" / r["filename"]
            d, sr = _info(p)
            rows.append(Row(to_relative(p), f"esc50_{r['category']}", "esc50", r["category"], d, sr,
                            esc50_fold_split(int(r["fold"])), r["src_file"], 0.0, d, -1.0, LICENCE["esc50"]))
    return rows


def _index_mad() -> list[Row]:
    root = data_root() / "raw" / "mad" / "archive" / "MAD_dataset"
    if not root.exists():
        return []
    rows = []
    for csv_name, is_test in (("training.csv", False), ("test.csv", True)):
        with (root / csv_name).open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                cls = MAD_CLASSES[int(r["label"])]
                # communication (radio / shouted orders) only ever feeds the `speech_interferer` pool
                p = root / r["path"]
                if not p.exists():
                    continue
                d, sr = _info(p)
                if d < MAD_MIN_DUR[cls]:
                    continue
                vid = r["youtube url"].split("v=")[-1].split("&")[0] or p.parent.name
                # MAD's own test folder shares YouTube videos with its training folder, so the
                # split is re-drawn here by video id (90 / 5 / 5); a video never straddles splits
                split = assign_split(f"mad:{vid}")
                rows.append(Row(to_relative(p), f"mad_{cls}", "mad", cls, d, sr, split, vid, 0.0, d, -1.0, LICENCE["mad"]))
    return rows


def _index_demand() -> list[Row]:
    root = data_root() / "raw" / "demand"
    rows = []
    for env in DEMAND_ENVS:
        p = root / env / "ch01.wav"
        if not p.exists():
            continue
        d, sr = _info(p)
        for split, (a, b) in DEMAND_SEGMENTS.items():
            b = min(b, d)
            rows.append(Row(to_relative(p), f"demand_{env.lower()}", "demand", env.lower(), b - a, sr, split,
                            env, a, b, -1.0, LICENCE["demand"]))
    return rows


def _fsd_pp(root: Path) -> dict[str, dict[str, bool]]:
    """fname -> {class name: True if every rater marked it Present and Predominant}."""
    f = root / "FSD50K.metadata" / "pp_pnp_ratings_FSD50K.json"
    voc = root / "FSD50K.ground_truth" / "vocabulary.csv"
    if not f.exists() or not voc.exists():
        return {}
    with voc.open(encoding="utf-8") as fh:
        mid2name = {r[2]: r[1] for r in csv.reader(fh)}
    out: dict[str, dict[str, bool]] = {}
    for fname, ratings in json.loads(f.read_text(encoding="utf-8")).items():
        out[fname] = {mid2name.get(mid, mid): bool(v) and min(v) >= 1.0 for mid, v in ratings.items()}
    return out


def _index_fsd50k() -> list[Row]:
    root = data_root() / "raw" / "fsd50k"
    if not root.exists():
        return []
    rows = []
    pp = _fsd_pp(root)
    for csv_name, audio_dir, forced in (("dev.csv", "FSD50K.dev_audio", None), ("eval.csv", "FSD50K.eval_audio", "test")):
        with (root / "FSD50K.ground_truth" / csv_name).open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                labels = set(r["labels"].split(","))
                if labels & FSD_VOICE:
                    pools = [name for name, classes in FSD_INTERFERER.items()
                             if labels & classes and not labels & FSD_INTERFERER_EXCLUDE]
                else:
                    pools = [name for name, classes in FSD_POOLS.items() if labels & classes]
                if not pools:
                    continue
                p = root / audio_dir / f"{r['fname']}.wav"
                if not p.exists():
                    continue
                d, sr = _info(p)
                split = forced or ("val" if r.get("split") == "val" else "train")
                for pool in pools:
                    if d < FSD_MIN_DUR[pool]:
                        continue
                    classes = (FSD_POOLS.get(pool) or FSD_INTERFERER[pool]) & labels
                    rows.append(Row(to_relative(p), pool, "fsd50k", pool[4:], d, sr, split, r["fname"], 0.0, d, -1.0,
                                    LICENCE["fsd50k"], all(pp.get(r["fname"], {}).get(c, False) for c in classes)))
    return rows


def _index_us8k() -> list[Row]:
    root = data_root() / "raw" / "urbansound8k" / "UrbanSound8K"
    if not root.exists():
        return []
    rows = []
    with (root / "metadata" / "UrbanSound8K.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["class"] not in US8K_POOLS:
                continue
            p = root / "audio" / f"fold{r['fold']}" / r["slice_file_name"]
            if not p.exists():
                continue
            d, sr = _info(p)
            if d < 0.3:
                continue
            fold = int(r["fold"])
            split = "train" if fold <= 8 else ("val" if fold == 9 else "test")
            rows.append(Row(to_relative(p), f"us8k_{r['class']}", "urbansound8k", r["class"], d, sr, split,
                            r["fsID"], 0.0, d, -1.0, LICENCE["urbansound8k"]))
    return rows


def _index_idmt() -> list[Row]:
    root = data_root() / "raw" / "idmt_traffic" / "IDMT_Traffic" / "audio"
    if not root.exists():
        return []
    rows = []
    kinds = {"T": "truck", "C": "car", "B": "bus", "M": "motorcycle"}
    for p in sorted(root.glob("*_ME_CH12.wav")):
        parts = p.stem.split("_")
        if len(parts) < 8 or parts[-3][:1] not in kinds or parts[-3] == "BG":
            continue
        veh = parts[-3]                      # e.g. TL / CR
        if len(veh) != 2 or veh[0] not in kinds:
            continue
        d, sr = _info(p)
        group = parts[3]                     # sample position in the long-term recording = one pass-by
        rows.append(Row(to_relative(p), f"idmt_{kinds[veh[0]]}", "idmt", kinds[veh[0]], d, sr,
                        assign_split(f"idmt:{group}"), group, 0.0, d, -1.0, LICENCE["idmt"]))
    return rows


def _index_drone() -> list[Row]:
    root = data_root() / "raw" / "drone_audio" / "Binary_Drone_Audio" / "yes_drone"
    if not root.exists():
        return []
    rows = []
    for p in sorted(root.glob("*.wav")):
        d, sr = _info(p)
        group = p.stem.split("-")[0]
        rows.append(Row(to_relative(p), "drone", "drone", "drone", d, sr, assign_split(f"drone:{group}"), group,
                        0.0, d, -1.0, LICENCE["drone"]))
    return rows


def _index_field_gunshots() -> list[Row]:
    root = sources_dir() / "impulsive" / "field"
    if not root.exists():
        return []
    shots: dict[str, list[float]] = {}
    with (root / "gunshot-audio-all-metadata.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            shots[r["filename"]] = [float(v) for v in r["gunshot_location_in_seconds"].strip("[]").split()]
    rows = []
    for p in sorted((root / "edge-collected-gunshot-audio").glob("*/*.wav")):
        if "_chan" in p.stem and "_chan0" not in p.stem:
            continue
        key = p.stem.replace("_chan0", "")
        if key not in shots or not shots[key]:
            continue
        d, sr = _info(p)
        uuid = key.split("_v")[0]
        rows.append(Row(to_relative(p), "field_gunshot", "field_gunshot", p.parent.name, d, sr,
                        assign_split(f"field:{uuid}"), uuid, 0.0, d, float(shots[key][0]), LICENCE["field_gunshot"]))
    return rows


INDEXERS = [_index_esc50, _index_mad, _index_demand, _index_fsd50k, _index_us8k, _index_idmt, _index_drone,
            _index_field_gunshots]


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def _screen_one(rel: str) -> tuple[str, bool]:
    try:
        return rel, bool(has_voice(load_mono(from_relative(rel))))
    except Exception:
        return rel, True


def pools_path() -> Path:
    return data_root() / "pools.parquet"


def build_pools(screen: bool | str = True, workers: int = 4, verbose: bool = True) -> pd.DataFrame:
    """screen=True runs the voice screen; "cached" reuses data/pools_voice_flagged.txt from a previous run."""
    from tqdm import tqdm

    rows: list[Row] = []
    for fn in INDEXERS:
        r = fn()
        if verbose:
            print(f"{fn.__name__[7:]:>16}: {len(r)} rows")
        rows.extend(r)
    df = pd.DataFrame([r.__dict__ for r in rows])
    df["has_voice"] = False
    flag_file = data_root() / "pools_voice_flagged.txt"
    if screen == "incremental" and flag_file.exists() and pools_path().exists():
        # reuse the old verdicts; screen only the paths the previous index never screened
        old = pd.read_parquet(pools_path(), columns=["path", "corpus", "pool"])
        done = set(old.loc[old.corpus.isin(SCREENED_CORPORA) & ~old.pool.isin(INTERFERER_POOLS), "path"])
        flagged = {ln.strip() for ln in flag_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
        todo = sorted(set(df.loc[df.corpus.isin(SCREENED_CORPORA) & ~df.pool.isin(INTERFERER_POOLS), "path"]) - done)
        if workers > 1 and todo:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(workers) as ex:
                for rel, v in tqdm(ex.map(_screen_one, todo, chunksize=16), total=len(todo), desc="voice screen (new)",
                                   disable=not verbose):
                    if v:
                        flagged.add(rel)
        else:
            for rel in tqdm(todo, desc="voice screen (new)", disable=not verbose):
                if _screen_one(rel)[1]:
                    flagged.add(rel)
        df["has_voice"] = df.path.isin(flagged) & ~df.pool.isin(INTERFERER_POOLS)
        flag_file.write_text("\n".join(sorted(flagged)) + "\n", encoding="utf-8")
        if verbose:
            print(f"voice screen: {len(todo)} new files screened, {len(flagged)} flagged in total")
    elif screen == "cached" and flag_file.exists():
        flagged = {ln.strip() for ln in flag_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
        df["has_voice"] = df.path.isin(flagged) & ~df.pool.isin(INTERFERER_POOLS)
        if verbose:
            print(f"voice screen: reused {len(flagged)} flagged paths from {flag_file}")
    elif screen:
        todo = sorted(set(df.loc[df.corpus.isin(SCREENED_CORPORA) & ~df.pool.isin(INTERFERER_POOLS), "path"]))
        flagged: set[str] = set()
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(workers) as ex:
                for rel, v in tqdm(ex.map(_screen_one, todo, chunksize=32), total=len(todo), desc="voice screen",
                                   disable=not verbose):
                    if v:
                        flagged.add(rel)
        else:
            for rel in tqdm(todo, desc="voice screen", disable=not verbose):
                if _screen_one(rel)[1]:
                    flagged.add(rel)
        df["has_voice"] = df.path.isin(flagged) & ~df.pool.isin(INTERFERER_POOLS)
        (data_root() / "pools_voice_flagged.txt").write_text("\n".join(sorted(flagged)) + "\n", encoding="utf-8")
        if verbose:
            print(f"voice screen: {len(flagged)} / {len(todo)} files flagged -> data/pools_voice_flagged.txt")
    df.to_parquet(pools_path(), index=False)
    if verbose:
        summary(df)
    return df


def summary(df: pd.DataFrame) -> None:
    ok = df[~df.has_voice]
    tab = ok.groupby(["pool", "split"]).size().unstack(fill_value=0)
    tab["hours"] = ok.groupby("pool").duration_s.sum() / 3600
    print(tab.round(2).to_string())


def load_pools(path: Path | None = None) -> pd.DataFrame:
    p = path or pools_path()
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `ancdata pools` first")
    return pd.read_parquet(p)


def split_mask(col: pd.Series, split: str) -> pd.Series:
    """train | val | test, plus the virtual splits 'heldout' (val + test) and 'all' (demo use)."""
    if split == "all":
        return pd.Series(True, index=col.index)
    if split == "heldout":
        return col.isin(["val", "test"])
    return col == split


# --------------------------------------------------------------------------
# Accessor with a byte-bounded decode cache
# --------------------------------------------------------------------------
class Pools:
    """Per-process accessor for one split. `draw_audio` returns 16 kHz mono
    float32, edge-silence trimmed, from an LRU cache bounded in bytes so a
    worker never grows past `cache_mb` (DEMAND is read by segment, not whole)."""

    def __init__(self, split: str, path: Path | None = None, cache_mb: float = 512.0, sr: int = SR,
                 allow_voice: bool = False, require_audit: bool = False):
        df = load_pools(path)
        if not allow_voice:
            df = df[~df.has_voice]
        if require_audit:                        # ancdata/pool_audit.py: tagger-confirmed content; untagged pools pass
            aud = pd.read_parquet(data_root() / "pools_audit.parquet", columns=["path", "pool", "start_s", "end_s", "audit_ok"])
            df = df.merge(aud, on=["path", "pool", "start_s", "end_s"], how="left")
            df = df[df.audit_ok.isna() | (df.audit_ok == True)].drop(columns="audit_ok")  # noqa: E712
        self.df = df[split_mask(df.split, split)].reset_index(drop=True)
        self.split = split
        self.sr = sr
        self.by_pool: dict[str, np.ndarray] = {p: g.index.to_numpy() for p, g in self.df.groupby("pool")}
        pp = self.df["pp"].fillna(False).astype(bool) if "pp" in self.df else pd.Series(False, index=self.df.index)
        self.by_pool_pp: dict[str, np.ndarray] = {p: g.index[pp[g.index]].to_numpy() for p, g in self.df.groupby("pool")}
        self.cache_bytes = int(cache_mb * 1024 * 1024)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._used = 0

    def pools(self) -> list[str]:
        return sorted(self.by_pool)

    def n(self, pool: str) -> int:
        return len(self.by_pool.get(pool, ()))

    def draw(self, pool: str, rng: np.random.Generator, pp_only: bool = False) -> pd.Series:
        """pp_only: FSD50K clips every rater marked Present-and-Predominant (isolated events);
        falls back to the whole pool when fewer than 20 such clips exist in the split."""
        idx = self.by_pool.get(pool)
        if pp_only and len(self.by_pool_pp.get(pool, ())) >= 20:
            idx = self.by_pool_pp[pool]
        if idx is None or len(idx) == 0:
            raise KeyError(f"pool {pool!r} has no files in split {self.split!r}")
        return self.df.iloc[int(rng.choice(idx))]

    @staticmethod
    def _key(row: Any) -> str:
        return f"{row['path']}|{float(row['start_s']):.1f}|{float(row['end_s']):.1f}"

    def load(self, row: Any) -> np.ndarray:
        """`row` is a pools DataFrame row or a plain dict with path / start_s / end_s / duration_s."""
        key = self._key(row)
        x = self._cache.get(key)
        if x is not None:
            self._cache.move_to_end(key)
            return x
        p = from_relative(row["path"])
        start_s, end_s, dur = float(row["start_s"]), float(row["end_s"]), float(row["duration_s"])
        if start_s > 0 or end_s - start_s < dur - 1e-6:
            info = sf.info(str(p))
            a = int(start_s * info.samplerate)
            b = int(end_s * info.samplerate)
            audio, fsr = sf.read(str(p), dtype="float32", always_2d=True, start=a, stop=b)
            audio = audio.mean(axis=1)
            if fsr != self.sr:
                from scipy.signal import resample_poly
                g = np.gcd(int(fsr), int(self.sr))
                audio = resample_poly(audio, self.sr // g, fsr // g).astype(np.float32)
            x = np.ascontiguousarray(audio, dtype=np.float32)
        else:
            x = load_mono(p, self.sr)
        x = trim_silence(x)
        if len(x) < 16:
            x = np.zeros(self.sr, dtype=np.float32)
        nbytes = x.nbytes
        while self._used + nbytes > self.cache_bytes and self._cache:
            _, old = self._cache.popitem(last=False)
            self._used -= old.nbytes
        if nbytes <= self.cache_bytes:
            self._cache[key] = x
            self._used += nbytes
        return x

    def draw_audio(self, pool: str, rng: np.random.Generator) -> tuple[np.ndarray, pd.Series]:
        row = self.draw(pool, rng)
        return self.load(row), row

    def row_meta(self, row: Any) -> dict[str, Any]:
        return {"path": row["path"], "pool": row["pool"], "group": row["group"], "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"])}
