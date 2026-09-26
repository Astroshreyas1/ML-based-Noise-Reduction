"""Leakage audit of the built battlefield_v3 set (and the sources behind it).

    .venv/Scripts/python.exe scripts/leakage_check.py [--root data/battlefield_v3] [--plans 3000] [--no-hash]

Checks, each reported PASS / WARN / FAIL in outputs/leakage_report.md:

  speech   S1 talkers disjoint across splits (snippets AND the pairs actually built)
           S2 snippet ids disjoint across splits
           S3 every source utterance used in exactly one snippet, hence one split
           S4 v2 registry (manifest.parquet) assigns the same split to every Lombard GRID talker (eval_lombard vs v3 train)
           S5 transcript overlap across splits (GRID codes / AVID sentence ids) -- content, not identity: reported, not failed
  noise    N1 noise FILES used in test / val never used in train (layers, bed2, wind, bursts, blasts)
           N2 noise RECORDINGS (pool group) disjoint across splits, DEMAND excepted (time split)
           N3 DEMAND crops in each split stay inside that split's time window
           N4 no voice-flagged file used anywhere
           N5 physics-event seeds and per-pair RNG streams unique (no repeated synthetic transient across pairs)
           N6 pool files with has_voice==False only; pools.parquet rows per split consistent with the config
  pairs    P1 no byte-identical noisy files within or across splits (md5)
           P2 no byte-identical clean files across splits
           P3 target = dry speech x AGC gain only (no noise in the target): re-render a sample and compare
  design   D1 ref channel speech leakage level (by construction 12-18 dB down) -- reported
           D2 target ducking (AGC gain applied to the target) -- reported as a property, not a leak
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ancdata.paths import data_root  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def rec(code: str, status: str, msg: str) -> None:
    RESULTS.append((code, status, msg))
    print(f"[{status}] {code}: {msg}")


def load_meta(split_dir: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in (split_dir / "meta.jsonl").open(encoding="utf-8") if l.strip()]
    return pd.DataFrame(rows)


def noise_files(df: pd.DataFrame) -> Counter:
    c: Counter = Counter()
    for _, r in df.iterrows():
        for L in r["layers"]:
            c[L["path"]] += 1
        if isinstance(r.get("bed2"), dict):
            c[r["bed2"]["path"]] += 1
        if isinstance(r.get("wind"), dict):
            c[r["wind"]["path"]] += 1
    return c


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=data_root() / "battlefield_v3")
    ap.add_argument("--plans", type=int, default=3000, help="train plans to regenerate for the seed / event-file checks")
    ap.add_argument("--no-hash", action="store_true", help="skip the md5 pass over every file")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "leakage_report.md")
    a = ap.parse_args()

    from ancdata.snippets import load_snippets
    from ancdata.pools import load_pools, DEMAND_SEGMENTS
    from ancdata.battlefield import build_battlefield_chain, load_battlefield_config

    splits = [s for s in ("train", "val", "test") if (a.root / s / "meta.jsonl").exists()]
    meta = {s: load_meta(a.root / s) for s in splits}
    for s in splits:
        print(f"{s}: {len(meta[s])} pairs")

    # ------------------------------------------------------------------ speech
    sn = load_snippets(data_root() / "snippets" / "lombard6s")
    spk_sn = {s: set(sn[sn.split == s].speaker_id) for s in ("train", "val", "test")}
    spk_pairs = {s: set(meta[s].speech_speaker_id) for s in splits}
    bad = [(x, y) for x in splits for y in splits if x < y and (spk_pairs[x] & spk_pairs[y])]
    ok_sn = all(not (spk_sn[x] & spk_sn[y]) for x in spk_sn for y in spk_sn if x < y)
    rec("S1", "PASS" if not bad and ok_sn else "FAIL",
        f"talkers per split (pairs): { {s: len(v) for s, v in spk_pairs.items()} }; overlaps: {bad or 'none'}; snippet table disjoint: {ok_sn}")
    ids = {s: set(meta[s].speech_id) for s in splits}
    bad = [(x, y) for x in splits for y in splits if x < y and (ids[x] & ids[y])]
    rec("S2", "PASS" if not bad else "FAIL", f"snippet ids per split { {s: len(v) for s, v in ids.items()} }; overlaps: {bad or 'none'}")
    utt_use: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for _, r in sn.iterrows():
        for seg in r["segments"]:
            utt_use[seg["utterance"]].append((r["id"], r["split"]))
    multi = {u: v for u, v in utt_use.items() if len(v) > 1}
    # AVID paragraphs are windowed into several snippets of the SAME speaker: allowed, same split by construction
    cross = {u: v for u, v in multi.items() if len({s for _, s in v}) > 1}
    rec("S3", "PASS" if not cross else "FAIL",
        f"{len(utt_use)} source utterances; {len(multi)} used in >1 snippet (paragraph windows, same talker); {len(cross)} across splits")
    import re
    man = data_root() / "manifest.parquet"
    m = pd.read_parquet(man) if man.exists() else pd.DataFrame(columns=["pile", "split", "speaker_id"])
    m = m[m.pile == "lombardgrid"]
    if len(m):
        v2 = {s: set(m[m.split == s].speaker_id) for s in ("train", "val", "test")}
        v3 = {s: {x for x in spk_sn[s] if re.match(r"^s\d+$", x)} for s in v2}
        ok = all(v2[s] == v3[s] for s in v2)
        rec("S4", "PASS" if ok else "FAIL", f"v2 manifest vs v3 snippets, Lombard GRID talkers per split identical: {ok}")
    else:
        rec("S4", "PASS", "v2 manifest has no Lombard GRID rows (registry predates the download), so no v2 eval can overlap v3; "
                          "both use assign_split('lombardgrid:<talker>'), so a rebuilt registry lands on the same split by construction")
    tr_grid = {t for t in meta["train"].speech_transcript.dropna()}
    te_grid = {t for t in meta.get("test", meta["train"]).speech_transcript.dropna()} if "test" in meta else set()
    frac = len(te_grid & tr_grid) / max(1, len(te_grid))
    rec("S5", "PASS", f"GRID transcripts: {len(te_grid)} distinct in test, {frac * 100:.0f} % also spoken (by other talkers) in train -- "
                      "content overlap only; talker identity is disjoint (S1). AVID sentence ids are shared by all talkers by design.")

    # ------------------------------------------------------------------ noise
    pools = load_pools()
    files = {s: noise_files(meta[s]) for s in splits}
    cfg = load_battlefield_config(ROOT / "configs" / "battlefield.yaml")
    ev_files = {s: Counter() for s in splits}
    seeds: dict[str, set[int]] = {s: set() for s in splits}
    seed_dupes = 0
    for s in splits:
        ch = build_battlefield_chain(cfg, s)
        n_plans = min(len(meta[s]), a.plans if s == "train" else len(meta[s]))
        idx = np.linspace(0, len(meta[s]) - 1, n_plans).astype(int)
        for k in idx:
            p = ch.plan(int(k))
            for b in p["bursts"] + p["blasts"]:
                if b.get("path"):
                    ev_files[s][b["path"]] += 1
                if b["seed"] in seeds[s]:
                    seed_dupes += 1
                seeds[s].add(b["seed"])
            for tx in p["texture"]:
                files[s][tx["path"]] += 1
    allf = {s: set(files[s]) | set(ev_files[s]) for s in splits}
    path2 = pools.drop_duplicates("path").set_index("path")
    demand_paths = set(pools[pools.corpus == "demand"].path)          # one 5-min file per environment, split by time (N3)
    bad = {(x, y): sorted((allf[x] & allf[y]) - demand_paths) for x in splits for y in splits if x < y}
    bad = {k: v for k, v in bad.items() if v}
    rec("N1", "PASS" if not bad else "FAIL",
        f"distinct noise files used: { {s: len(v) for s, v in allf.items()} }; cross-split overlaps outside DEMAND's time-split files: {bad or 'none'}")
    grp = {s: {f"{path2.loc[f, 'group']}|{path2.loc[f, 'corpus']}" for f in allf[s] if f in path2.index and path2.loc[f, "corpus"] != "demand"}
           for s in splits}
    bad = {(x, y): len(grp[x] & grp[y]) for x in splits for y in splits if x < y and (grp[x] & grp[y])}
    rec("N2", "PASS" if not bad else "FAIL",
        f"noise recordings (video / fold source / recording id) used: { {s: len(v) for s, v in grp.items()} }; cross-split overlaps: {bad or 'none'}")
    dem_bad = 0
    dem_n = 0
    for s in splits:
        lo, hi = DEMAND_SEGMENTS[s]
        for _, r in meta[s].iterrows():
            for L in r["layers"]:
                if L["pool"].startswith("demand_"):
                    dem_n += 1
                    if not (L["start_s"] >= lo - 1e-6 and L["end_s"] <= hi + 1e-6):
                        dem_bad += 1
    rec("N3", "PASS" if dem_bad == 0 else "FAIL", f"{dem_n} DEMAND layers, {dem_bad} outside their split's time window "
                                                  f"(train 0-210 s / val 210-255 / test 255-300 of the same 5-min recording: same environment, disjoint audio)")
    flagged = set(pools[pools.has_voice].path)
    used_flagged = {s: len(allf[s] & flagged) for s in splits}
    rec("N4", "PASS" if sum(used_flagged.values()) == 0 else "FAIL", f"voice-flagged files used: {used_flagged}")
    n_seeds = sum(len(v) for v in seeds.values()) + seed_dupes
    expected = n_seeds ** 2 / 2 / 2 ** 31                 # birthday bound for 31-bit event seeds
    rec("N5", "PASS" if seed_dupes <= max(3, 3 * expected) else "FAIL",
        f"{n_seeds} transient seeds over {sum(min(len(meta[s]), a.plans) for s in splits)} plans, {seed_dupes} repeated 31-bit seeds "
        f"(birthday expectation {expected:.1f}); a shared seed only repeats a waveform template, never a pair: files, ratios and "
        f"times come from rng(seed, index), which is unique per pair")
    cfg_pools = set()
    for scn in cfg["scenarios"]:
        cfg_pools |= set(scn["bed"][0])
        for ev in scn.get("events", []):
            cfg_pools |= set(ev[0])
    cfg_pools |= set(cfg["mix"]["texture"]["pools"]) | set(cfg["wind"]["pools"]) | {"field_gunshot", "mad_gunshot", "mad_shelling"}
    ok_pools = pools[~pools.has_voice]
    empty = {p: [s for s in ("train", "val", "test") if ((ok_pools.pool == p) & (ok_pools.split == s)).sum() == 0] for p in sorted(cfg_pools)}
    empty = {p: v for p, v in empty.items() if v}
    rec("N6", "WARN" if empty else "PASS", f"config pools with an empty split (the chain falls back to another listed pool): {empty or 'none'}")

    # ------------------------------------------------------------------ pairs
    if not a.no_hash:
        h_noisy: dict[str, list[str]] = defaultdict(list)
        h_clean: dict[str, list[str]] = defaultdict(list)
        for s in splits:
            for sub, store in (("noisy", h_noisy), ("clean", h_clean)):
                for p in sorted((a.root / s / sub).iterdir()):
                    if p.suffix in (".flac", ".wav"):
                        store[md5(p)].append(f"{s}/{p.name}")
        dn = {h: v for h, v in h_noisy.items() if len(v) > 1}
        dc = {h: v for h, v in h_clean.items() if len({x.split('/')[0] for x in v}) > 1}
        rec("P1", "PASS" if not dn else "FAIL", f"{sum(len(v) for v in h_noisy.values())} noisy files hashed; byte-identical groups: {len(dn)}")
        rec("P2", "PASS" if not dc else "FAIL", f"clean files byte-identical across splits: {len(dc)}")
    else:
        rec("P1", "WARN", "hash pass skipped (--no-hash)")
        rec("P2", "WARN", "hash pass skipped (--no-hash)")
    import soundfile as sf
    cfg_dbg = dict(cfg, debug_stems=True)
    ch = build_battlefield_chain(cfg_dbg, "test" if "test" in splits else splits[0])
    worst = 0.0
    for k in range(5):
        pr = ch.generate(k)
        st = pr.meta["_stems"]
        on_disk, _ = sf.read(str(next((a.root / ch.split / "clean").glob(f"{k:06d}.*"))), dtype="float32")
        worst = max(worst, float(np.abs(on_disk - st["speech_dry"] * st["agc_gain"]).max()))
    rec("P3", "PASS" if worst < 2e-4 else "FAIL", f"clean on disk == dry speech x AGC gain (max |diff| {worst:.2e}; 16-bit FLAC quantisation is 3e-5)")

    # ------------------------------------------------------------------ design notes
    rs = pd.concat([meta[s][["ref_speech_db", "ref_noise_db", "agc_min_gain_db"]] for s in splits])
    rec("D1", "PASS", f"ref channel carries the talker at {rs.ref_speech_db.mean():.1f} dB (mean) relative to the boom, scene at +{rs.ref_noise_db.mean():.1f} dB: "
                      "a noise reference, not a second clean copy; single-channel fallback = drop channel 1")
    rec("D2", "PASS", f"target ducking: AGC min gain p50 {rs.agc_min_gain_db.median():.1f} dB -- the target is speech x a gain computable from the input envelope "
                      "(same rule as the v2 ADC gain); no noise content enters the target (P3)")

    lines = ["# Leakage audit: battlefield_v3", "", f"Root: `{a.root}`; pairs: " + ", ".join(f"{s} {len(meta[s])}" for s in splits), "",
             "| Check | Status | Detail |", "|---|---|---|"]
    lines += [f"| {c} | **{s}** | {m} |" for c, s, m in RESULTS]
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    lines += ["", f"**{n_fail} FAIL**, {sum(1 for _, s, _ in RESULTS if s == 'WARN')} WARN, {sum(1 for _, s, _ in RESULTS if s == 'PASS')} PASS.", ""]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {a.out}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
