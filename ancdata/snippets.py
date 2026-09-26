"""Lombard speech snippets: fixed-length, labelled, speaker-disjoint.

The chain crops speech on the fly, but the listening trials and the noise-
layering work need a *fixed*, inspectable speech base: every snippet the same
length, every snippet carrying its labels (corpus, speaker, effort level,
transcript, which utterances it was built from and where they sit). This
module materialises that base once:

    ancdata snippets --seconds 6 --out data/snippets/lombard6s

Sources (docs/LOMBARD_SURVEY.md):

    Lombard GRID   s<spk>_<l|p>_<code>.wav   l = Lombard (80 dB SSN in headphones), p = plain
    AVID SENT      sp<spk>_s<sess>_sen<n>_<soft|normal|loud|veryloud>.wav   ~1.5 s sentences
    AVID PARA      sp<spk>_s<sess>_para<n>_<level>.wav                      ~30 s paragraphs

Snippet length 6.0 s (why not the chain's 4.0 s training crop): a GRID
utterance is 2.6 s and an AVID sentence 1.6 s, so 6 s holds 2-3 utterances of
the same talker at the same effort with natural 0.3-0.9 s pauses -- a short
radio message, not a looped phrase. The pauses expose noise-only frames the
model must learn to leave alone, and 6 s is long enough for a pass-by or an
approaching vehicle to evolve under the speech. The chain can still crop its
locked 4.0 s window out of a 6 s snippet; the reverse is not possible.

Rules: utterances are never re-used across snippets, never looped, and never
level-normalised (AVID levels are calibrated; GRID levels carry the effort);
pauses are digital silence with 10 ms fades at every cut; AVID paragraphs
are cut at detected pauses, never mid-word. Splits reuse the registry's
speaker hash, so a snippet's split agrees with the manifest's.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .audio import EPS, active_rms_db, load_mono, write_wav
from .config import SR
from .paths import data_root, sources_dir
from .registry import assign_split

# ---------------------------------------------------------------------------
# GRID sentence code -> words
# ---------------------------------------------------------------------------
_GRID_CMD = {"b": "bin", "l": "lay", "p": "place", "s": "set"}
_GRID_COL = {"b": "blue", "g": "green", "r": "red", "w": "white"}
_GRID_PREP = {"a": "at", "b": "by", "i": "in", "w": "with"}
_GRID_ADV = {"a": "again", "n": "now", "p": "please", "s": "soon"}
_GRID_DIG = {"z": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
             "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def grid_transcript(code: str) -> str:
    """'bbat9p' -> 'bin blue at t nine please'."""
    if len(code) != 6:
        raise ValueError(f"bad GRID code {code!r}")
    c, k, p, letter, d, a = code
    return " ".join([_GRID_CMD[c], _GRID_COL[k], _GRID_PREP[p], letter, _GRID_DIG[d], _GRID_ADV[a]])


# ---------------------------------------------------------------------------
# Utterance inventory
# ---------------------------------------------------------------------------
LOMBARD_CLASS = {"lombard", "loud", "veryloud"}   # effort levels that count as Lombard-class speech
ALL_LEVELS = {"plain", "lombard", "soft", "normal", "loud", "veryloud"}
LICENCE = {"lombardgrid": "CC BY 4.0", "avid": "CC BY 4.0"}


@dataclass
class Utt:
    path: Path
    corpus: str            # lombardgrid | avid
    kind: str              # grid | sent | para
    speaker_id: str        # s10 | sp20
    effort: str            # plain lombard | soft normal loud veryloud
    gender: str            # F | M | unknown
    sentence_id: str
    transcript: str | None
    spl_leq_a_db: float | None   # AVID calibrated level (A-weighted Leq, slow); None for GRID
    native_sr: int


def _avid_meta(avid_root: Path) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for csv_path in avid_root.glob("Metadata_with_labels_*_fullband.csv"):
        with csv_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                meta[row["Filename"]] = row
    return meta


def list_utterances(levels: Iterable[str] = LOMBARD_CLASS) -> list[Utt]:
    import soundfile as sf

    levels = set(levels)
    utts: list[Utt] = []

    grid = sources_dir() / "speech" / "lombardgrid" / "lombardgrid" / "audio"
    if grid.exists():
        for p in sorted(grid.glob("*.wav")):
            # 71 files are named s13_l_<prompted>_WRONG_<spoken>.wav: the talker
            # misread the prompt; the last code is what was actually said.
            parts = p.stem.split("_")
            spk, cond, code = parts[0], parts[1], parts[-1]
            if cond not in ("l", "p"):
                continue          # one stray file (s14_1_50_15_r_prwr3a.wav) with no condition tag
            effort = "lombard" if cond == "l" else "plain"
            if effort not in levels:
                continue
            try:
                transcript = grid_transcript(code)
            except (KeyError, ValueError):
                transcript = None
            utts.append(Utt(p, "lombardgrid", "grid", spk, effort, "unknown", code,
                            transcript, None, int(sf.info(str(p)).samplerate)))

    avid = next(iter((sources_dir() / "speech" / "avid").glob("Repositoty*")), None)
    if avid is not None:
        meta = _avid_meta(avid)
        pat = re.compile(r"^(sp\d+)_(s\d+)_(sen|para)(\d+)_(soft|normal|loud|veryloud)$")
        for sub in ("SENT", "PARA"):
            for p in sorted((avid / sub).glob("*.wav")):
                m = pat.match(p.stem)
                if not m:
                    raise ValueError(f"{p}: unexpected AVID filename")
                spk, _sess, kind, n, effort = m.groups()
                if effort not in levels:
                    continue
                row = meta.get(p.name, {})
                gender = {"Female": "F", "Male": "M"}.get(row.get("Gender", ""), "unknown")
                spl = float(row["L_eqAS"]) if row.get("L_eqAS") else None
                utts.append(Utt(p, "avid", kind, spk, effort, gender, f"{kind}{n}", None, spl,
                                int(sf.info(str(p)).samplerate)))
    return utts


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
FRAME_MS = 20.0


def _frame_db(x: np.ndarray, sr: int = SR, frame_ms: float = FRAME_MS) -> np.ndarray:
    n = int(sr * frame_ms / 1000.0)
    m = len(x) // n
    if m == 0:
        return np.array([20.0 * np.log10(np.sqrt(np.mean(x ** 2)) + EPS)])
    fr = x[: m * n].reshape(m, n)
    return 20.0 * np.log10(np.sqrt(np.mean(np.square(fr, dtype=np.float64), axis=1)) + EPS)


def trim_edges(x: np.ndarray, sr: int = SR, rel_db: float = -40.0, keep_ms: float = 60.0) -> np.ndarray:
    """Strip leading/trailing floor: frames below (active level + rel_db), keeping `keep_ms` on each side."""
    e = _frame_db(x, sr)
    thr = active_rms_db(x, sr) + rel_db
    idx = np.where(e > thr)[0]
    if len(idx) == 0:
        return x
    n = int(sr * FRAME_MS / 1000.0)
    keep = int(sr * keep_ms / 1000.0)
    a = max(0, idx[0] * n - keep)
    b = min(len(x), (idx[-1] + 1) * n + keep)
    return x[a:b]


def fade_edges(x: np.ndarray, sr: int = SR, ms: float = 10.0) -> np.ndarray:
    n = min(int(sr * ms / 1000.0), len(x) // 2)
    if n <= 0:
        return x
    y = x.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def speech_fraction(x: np.ndarray, sr: int = SR, rel_db: float = -25.0) -> float:
    e = _frame_db(x, sr)
    thr = active_rms_db(x, sr) + rel_db
    return float(np.mean(e > thr))


def best_cut(x: np.ndarray, lo: int, hi: int, sr: int = SR, rel_db: float = -20.0,
             min_pause_ms: float = 100.0, frame_ms: float = 10.0) -> int:
    """Sample index in [lo, hi] to cut a read paragraph at: the centre of the
    latest pause of >= min_pause_ms (frames below active + rel_db); if the
    talker never pauses that long, the quietest 50 ms in the window (a stop
    closure or word boundary). Read speech at high effort pauses rarely, so a
    strict pause detector would fall back to a hard cut far too often."""
    lo, hi = max(0, lo), min(len(x), hi)
    if hi - lo < int(0.1 * sr):
        return hi
    n = int(sr * frame_ms / 1000.0)
    e = _frame_db(x, sr, frame_ms)
    thr = active_rms_db(x, sr) + rel_db
    f_lo, f_hi = lo // n, max(lo // n + 1, hi // n)
    quiet = e[f_lo:f_hi] < thr
    min_frames = int(np.ceil(min_pause_ms / frame_ms))
    best = None
    i = 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if j - i >= min_frames:
                best = (f_lo + (i + j) // 2) * n
            i = j
        else:
            i += 1
    if best is not None:
        return int(best)
    k = 5                                   # 50 ms moving average of the frame energies
    seg = e[f_lo:f_hi]
    if len(seg) <= k:
        return int((f_lo + int(np.argmin(seg))) * n)
    sm = np.convolve(seg, np.ones(k) / k, mode="valid")
    return int((f_lo + int(np.argmin(sm)) + k // 2) * n)


# ---------------------------------------------------------------------------
# Snippet assembly
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    utterance: str      # source file stem
    src_start_s: float  # where in the (trimmed) source this piece begins
    src_end_s: float
    start_s: float      # where it sits inside the snippet
    end_s: float


@dataclass
class Snippet:
    id: str
    path: str
    corpus: str
    kind: str
    speaker_id: str
    gender: str
    effort: str
    lombard_class: bool
    split: str
    seconds: float
    n_utterances: int
    segments: list[Segment]
    sentence_ids: list[str]
    transcript: str | None
    speech_fraction: float
    active_rms_dbfs: float
    peak: float
    spl_leq_a_db: float | None
    native_sr: int
    licence: str
    seed: int
    extra: dict[str, Any] = field(default_factory=dict)


def pack_short_utterances(utts: list[Utt], rng: np.random.Generator, n_samples: int, sr: int = SR,
                          pause_s: tuple[float, float] = (0.3, 0.9), min_fill: float = 0.35
                          ) -> list[tuple[np.ndarray, list[Segment], list[Utt]]]:
    """Greedy packing of one (speaker, effort) group into fixed-length snippets.
    Each utterance is used exactly once; leftovers that would fill < min_fill of
    a snippet are dropped rather than padded into a mostly-silent example."""
    order = list(rng.permutation(len(utts)))
    audios = {i: fade_edges(trim_edges(load_mono(utts[i].path, sr), sr), sr) for i in order}
    out = []
    cur: list[int] = []
    cur_len = 0
    for i in order:
        a = audios[i]
        gap = int(rng.uniform(*pause_s) * sr) if cur else 0
        if cur and cur_len + gap + len(a) > n_samples:
            out.append(cur)
            cur, cur_len = [], 0
            gap = 0
        if len(a) > n_samples:
            a = a[:n_samples]
            audios[i] = fade_edges(a, sr)
        cur.append(i)
        cur_len += gap + len(a)
    if cur:
        out.append(cur)

    results = []
    for group in out:
        pieces = []
        total = 0
        for k, i in enumerate(group):
            gap = int(rng.uniform(*pause_s) * sr) if k else 0
            pieces.append((gap, i))
            total += gap + len(audios[i])
        if total > n_samples:                       # pause draw pushed it over: shrink gaps
            excess = total - n_samples
            pieces = [(max(0, g - int(np.ceil(excess / max(1, len(pieces) - 1)))) if k else 0, i)
                      for k, (g, i) in enumerate(pieces)]
            total = sum(g + len(audios[i]) for g, i in pieces)
        if total < min_fill * n_samples and len(group) < len(utts):
            continue                                # thin leftover; drop
        snippet = np.zeros(n_samples, dtype=np.float32)
        pos = int(rng.integers(0, n_samples - total + 1))
        segs = []
        for g, i in pieces:
            pos += g
            a = audios[i]
            snippet[pos: pos + len(a)] = a
            segs.append(Segment(utts[i].path.stem, 0.0, len(a) / sr, pos / sr, (pos + len(a)) / sr))
            pos += len(a)
        results.append((snippet, segs, [utts[i] for _, i in pieces]))
    return results


def window_long_utterance(utt: Utt, rng: np.random.Generator, n_samples: int, sr: int = SR,
                          min_window: float = 0.66, min_last: float = 0.35
                          ) -> list[tuple[np.ndarray, list[Segment], list[Utt]]]:
    """Cut a paragraph into <= n_samples windows at detected pauses (never mid-word)."""
    x = trim_edges(load_mono(utt.path, sr), sr)
    results = []
    start = 0
    lo_len = int(min_window * n_samples)
    while start < len(x):
        hi = min(len(x), start + n_samples)
        end = hi if hi == len(x) else best_cut(x, start + lo_len, hi, sr)
        piece = fade_edges(x[start:end], sr)
        if len(piece) < min_last * n_samples and start > 0:
            break                                   # thin tail; drop
        snippet = np.zeros(n_samples, dtype=np.float32)
        pos = int(rng.integers(0, n_samples - len(piece) + 1)) if len(piece) < n_samples else 0
        snippet[pos: pos + len(piece)] = piece
        results.append((snippet, [Segment(utt.path.stem, start / sr, end / sr, pos / sr, (pos + len(piece)) / sr)], [utt]))
        start = end
    return results


def _group_seed(seed: int, key: tuple[str, ...]) -> int:
    """Stable per-group seed (Python's hash() is salted per process)."""
    import hashlib
    return int(hashlib.sha256(f"{seed}:{':'.join(key)}".encode()).hexdigest()[:15], 16)


def build_snippets(out: Path, seconds: float = 6.0, levels: Iterable[str] = LOMBARD_CLASS, seed: int = 0,
                   limit_per_group: int | None = None, fmt: str = "pcm16", verbose: bool = True) -> list[Snippet]:
    from tqdm import tqdm

    sr = SR
    n = int(round(seconds * sr))
    utts = list_utterances(levels)
    if not utts:
        raise FileNotFoundError("no Lombard GRID / AVID utterances under data/sources/speech")
    groups: dict[tuple[str, str, str, str], list[Utt]] = {}
    for u in utts:
        groups.setdefault((u.corpus, u.kind, u.speaker_id, u.effort), []).append(u)

    out.mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)
    rows: list[Snippet] = []
    counters: dict[str, int] = {}
    for key in tqdm(sorted(groups), desc="snippets", disable=not verbose):
        corpus, kind, spk, effort = key
        g = groups[key]
        rng = np.random.default_rng(_group_seed(seed, key))
        if kind == "para":
            made = [r for u in g for r in window_long_utterance(u, rng, n, sr)]
        else:
            made = pack_short_utterances(g, rng, n, sr)
        if limit_per_group:
            made = made[:limit_per_group]
        split = assign_split(f"{corpus}:{spk}")
        for audio, segs, members in made:
            extra: dict[str, Any] = {}
            pk = float(np.abs(audio).max())
            if pk > 0.999:                          # resampling overshoot on a hot AVID take
                audio = audio * (0.999 / pk)
                extra["gain_db"] = float(20.0 * np.log10(0.999 / pk))
            prefix = f"{corpus}_{spk}_{effort}"
            counters[prefix] = counters.get(prefix, 0) + 1
            sid = f"{prefix}_{counters[prefix]:04d}"
            rel = f"audio/{sid}.wav"
            write_wav(out / rel, audio, sr, fmt=fmt)
            spls = [m.spl_leq_a_db for m in members if m.spl_leq_a_db is not None]
            tr = [m.transcript for m in members if m.transcript]
            rows.append(Snippet(
                id=sid, path=rel, corpus=corpus, kind=kind, speaker_id=spk, gender=members[0].gender,
                effort=effort, lombard_class=effort in LOMBARD_CLASS, split=split, seconds=seconds,
                n_utterances=len(members), segments=segs, sentence_ids=[m.sentence_id for m in members],
                transcript=" | ".join(tr) if tr else None, speech_fraction=speech_fraction(audio, sr),
                active_rms_dbfs=active_rms_db(audio, sr), peak=float(np.abs(audio).max()),
                spl_leq_a_db=float(np.mean(spls)) if spls else None, native_sr=members[0].native_sr,
                licence=LICENCE[corpus], seed=seed, extra=extra,
            ))
    write_meta(rows, out, seconds, seed, levels)
    if verbose:
        report(rows)
    return rows


def write_meta(rows: list[Snippet], out: Path, seconds: float, seed: int, levels: Iterable[str]) -> None:
    import pandas as pd

    with (out / "meta.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")
    df = pd.DataFrame([{**asdict(r), "segments": json.dumps([asdict(s) for s in r.segments]),
                        "sentence_ids": json.dumps(r.sentence_ids), "extra": json.dumps(r.extra)} for r in rows])
    df.to_parquet(out / "meta.parquet", index=False)
    (out / "README.md").write_text(
        f"# Lombard speech snippets ({seconds:.1f} s, 16 kHz mono)\n\n"
        f"Built by `ancdata snippets --seconds {seconds} --seed {seed}` from Lombard GRID + AVID "
        f"(levels: {', '.join(sorted(levels))}). One row per snippet in `meta.jsonl` / `meta.parquet`; "
        "audio under `audio/`. Levels are native (not normalised). See `ancdata/snippets.py` for the rules.\n",
        encoding="utf-8")


def report(rows: list[Snippet]) -> None:
    import pandas as pd

    df = pd.DataFrame([{"corpus": r.corpus, "kind": r.kind, "effort": r.effort, "split": r.split,
                        "speaker": r.speaker_id, "sf": r.speech_fraction, "lvl": r.active_rms_dbfs} for r in rows])
    print(f"{len(df)} snippets, {len(df) * rows[0].seconds / 3600:.2f} h, {df.speaker.nunique()} speakers")
    print(df.groupby(["corpus", "kind", "effort", "split"]).size().unstack(fill_value=0).to_string())
    print("speech fraction: median %.2f  p10 %.2f  p90 %.2f" % (df.sf.median(), df.sf.quantile(.1), df.sf.quantile(.9)))
    print("active level dBFS: median %.1f  p10 %.1f  p90 %.1f" % (df.lvl.median(), df.lvl.quantile(.1), df.lvl.quantile(.9)))


def load_snippets(root: Path | None = None):
    """meta.parquet as a DataFrame; `segments` / `sentence_ids` decoded."""
    import pandas as pd

    root = root or data_root() / "snippets" / "lombard6s"
    df = pd.read_parquet(root / "meta.parquet")
    df["segments"] = df["segments"].map(json.loads)
    df["sentence_ids"] = df["sentence_ids"].map(json.loads)
    df["abs_path"] = df["path"].map(lambda p: str(root / p))
    return df
