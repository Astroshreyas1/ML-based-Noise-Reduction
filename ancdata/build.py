"""Memory-aware parallel builder for the battlefield dataset.

    ancdata battlefield --split train            # every snippet x max_scenes_per_snippet, auto workers
    ancdata battlefield --split test --n 1072    # frozen float32 set

Workers are sized from the machine, not guessed: ``min(physical cores - 1,
available RAM x 0.6 / build.worker_mb)``; each worker holds one chain with a
byte-bounded decode cache (``pools.cache_mb``), so memory stays flat however
long the build runs. Generation is index-deterministic, so the output is
identical for any worker count, and ``--resume`` skips ids already on disk.
Each worker appends its meta lines to a shard; shards are merged, sorted by
id, and the report (composition tables + unprocessed-input baseline on the
test split) is written next to the data.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .audio import write_wav
from .battlefield import BattlefieldChain, build_battlefield_chain, load_battlefield_config
from .materialize import pair_meta
from .paths import data_root


def dataset_root(cfg: dict[str, Any]) -> Path:
    return data_root() / cfg["name"]


def auto_workers(cfg: dict[str, Any], verbose: bool = True) -> int:
    import psutil
    phys = psutil.cpu_count(logical=False) or os.cpu_count() or 2
    avail = psutil.virtual_memory().available
    worker_mb = float(cfg["build"].get("worker_mb", 900))
    worker_mb = max(worker_mb, float(cfg["pools"].get("cache_mb", 512)) + 150.0)
    by_ram = int(avail * 0.6 / (worker_mb * 1024 * 1024))
    w = max(1, min(phys - 1, by_ram))
    if verbose:
        print(f"workers: {w}  (physical cores {phys}, available RAM {avail / 2**30:.1f} GiB, "
              f"{worker_mb:.0f} MB per worker -> RAM allows {by_ram})")
    return w


def _write(out: Path, pair, fmt: str) -> str:
    k = pair.meta["index"]
    write_wav(out / "noisy" / f"{k:06d}.wav", pair.noisy, fmt=fmt)
    write_wav(out / "clean" / f"{k:06d}.wav", pair.clean, fmt=fmt)
    m = pair_meta(pair)
    m["ambience"] = m.get("scenario")          # metrics.results_table groups by this column
    return json.dumps({"id": f"{k:06d}", **m}, default=float)


def _task(args) -> list[str]:
    cfg, split, out, indices, fmt = args
    chain = build_battlefield_chain(cfg, split)
    out = Path(out)
    lines = []
    shard = out / "_shards" / f"{os.getpid()}.jsonl"
    shard.parent.mkdir(exist_ok=True)
    with shard.open("a", encoding="utf-8") as fh:
        for k in indices:
            line = _write(out, chain.generate(k), fmt)
            fh.write(line + "\n")
            fh.flush()
            lines.append(line)
    return lines


def _existing_ids(out: Path) -> set[int]:
    ids = set()
    d = out / "noisy"
    if d.exists():
        for p in d.iterdir():
            if p.suffix in (".wav", ".flac"):
                try:
                    ids.add(int(p.stem))
                except ValueError:
                    pass
    return ids


def build_split(cfg: dict[str, Any] | str | Path, split: str, n: int | None = None, out: Path | None = None,
                workers: int | str = "auto", fmt: str | None = None, resume: bool = True, start: int = 0,
                task_size: int = 100, verbose: bool = True) -> Path:
    from tqdm import tqdm

    cfg_path = None if isinstance(cfg, dict) else Path(cfg)
    if not isinstance(cfg, dict):
        cfg = load_battlefield_config(cfg)
    chain = BattlefieldChain(cfg, split)          # cheap: reads two parquet files
    n = int(n if n is not None else chain.max_examples)
    out = Path(out) if out else dataset_root(cfg) / split
    fmt = fmt or cfg["build"]["fmt"].get(split, "flac")
    (out / "noisy").mkdir(parents=True, exist_ok=True)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    wanted = list(range(start, start + n))
    have = _existing_ids(out) if resume else set()
    todo = [k for k in wanted if k not in have]
    if verbose:
        print(f"{split}: {n} pairs of {float(cfg['segment_seconds']):.1f} s = {n * float(cfg['segment_seconds']) / 3600:.1f} h "
              f"({len(have & set(wanted))} already on disk, {len(todo)} to build, fmt={fmt}) -> {out}")
    if workers == "auto":
        workers = auto_workers(cfg, verbose)
    workers = int(workers)
    lines: list[str] = []
    if todo:
        tasks = [(cfg, split, str(out), todo[i:i + task_size], fmt) for i in range(0, len(todo), task_size)]
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(workers) as ex:
                for res in tqdm(ex.map(_task, tasks), total=len(tasks), desc=f"build {split}", unit="task",
                                disable=not verbose):
                    lines.extend(res)
        else:
            for t in tqdm(tasks, desc=f"build {split}", unit="task", disable=not verbose):
                lines.extend(_task(t))
    # merge: lines from this run + previously written meta for ids kept from disk
    meta_path = out / "meta.jsonl"
    keep: dict[str, str] = {}
    if resume and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    keep[json.loads(ln)["id"]] = ln
    for ln in lines:
        keep[json.loads(ln)["id"]] = ln
    wanted_ids = {f"{k:06d}" for k in wanted}
    merged = [keep[i] for i in sorted(keep) if i in wanted_ids]
    with meta_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(merged) + "\n")
    shutil.rmtree(out / "_shards", ignore_errors=True)
    if cfg_path:
        shutil.copy2(cfg_path, out / "config.yaml")
    from .demo_check import registered_ranges
    files = sorted((out / "noisy").glob("*.wav")) + sorted((out / "noisy").glob("*.flac"))
    if len(files) > 2000:                         # a sample is enough for the demo-day range check
        files = [files[i] for i in np.linspace(0, len(files) - 1, 2000).astype(int)]
    registered_ranges(files, out / "ranges.json")
    if verbose:
        print(f"{split}: {len(merged)} pairs, meta -> {meta_path}")
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _hist(vals: np.ndarray, edges: list[float]) -> str:
    h, _ = np.histogram(vals, bins=edges)
    return " | ".join(f"{edges[i]:g}..{edges[i + 1]:g}: {100 * h[i] / max(1, len(vals)):.0f} %" for i in range(len(h)))


def split_summary(root: Path) -> dict[str, Any]:
    import pandas as pd
    rows = [json.loads(l) for l in (root / "meta.jsonl").open(encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    seg = 6.0
    cfg_path = root / "config.yaml"
    if cfg_path.exists():
        import yaml
        seg = float(yaml.safe_load(cfg_path.read_text(encoding="utf-8")).get("segment_seconds", 6.0))
    ev = [e for r in rows for e in r.get("events", [])]
    evc = pd.Series([e["category"] for e in ev]).value_counts() if ev else pd.Series(dtype=int)
    return {
        "n": len(df), "hours": len(df) * seg / 3600, "speakers": int(df.speech_speaker_id.nunique()),
        "snippets": int(df.speech_id.nunique()),
        "scenario": df.scenario.value_counts().to_dict(),
        "effort": df.speech_effort.value_counts().to_dict(),
        "corpus": df.speech_corpus.value_counts().to_dict(),
        "gunfire_pct": 100 * float(df.gunfire.mean()), "blast_pct": 100 * float((df.n_blasts > 0).mean()),
        "wind_pct": 100 * float(df.wind.notna().mean()),
        "snr_lufs_hist": _hist(df.snr_lufs_db.to_numpy(), [-8, -4, 0, 4, 8, 12, 15]),
        "snr_effective_q": [float(v) for v in df.snr_effective_db.quantile([.1, .5, .9])],
        "clip_pct_mean": float(df.clip_pct.mean()), "clip_gt_0.1_pct": 100 * float((df.clip_pct > 0.1).mean()),
        "agc_min_gain_q": [float(v) for v in df.agc_min_gain_db.quantile([.1, .5, .9])],
        "events": evc.to_dict(), "events_per_clip": len(ev) / max(1, len(df)),
        "speech_level_q": [float(v) for v in df.speech_level_db.quantile([.1, .5, .9])],
    }


def write_report(root: Path, baseline: bool = True, verbose: bool = True) -> Path:
    root = Path(root)
    parts = [f"# Build report: {root.name}\n"]
    for split in ("train", "val", "test"):
        d = root / split
        if not (d / "meta.jsonl").exists():
            continue
        s = split_summary(d)
        parts.append(f"## {split}\n")
        parts.append(f"- pairs: **{s['n']}** = {s['hours']:.1f} h; {s['speakers']} speakers, {s['snippets']} distinct snippets "
                     f"(each used {s['n'] / max(1, s['snippets']):.1f}x)")
        parts.append(f"- speech corpus: {s['corpus']}; effort: {s['effort']}")
        parts.append(f"- scenario mix: {dict(sorted(s['scenario'].items()))}")
        parts.append(f"- gunfire in {s['gunfire_pct']:.0f} % of clips, blasts in {s['blast_pct']:.0f} %, wind layer in {s['wind_pct']:.0f} %; "
                     f"{s['events_per_clip']:.1f} labelled events per clip: {s['events']}")
        parts.append(f"- SNR (K-weighted, drawn): {s['snr_lufs_hist']}; effective active-RMS SNR p10/p50/p90 = "
                     f"{' / '.join(f'{v:.1f}' for v in s['snr_effective_q'])} dB")
        parts.append(f"- speech level p10/p50/p90 = {' / '.join(f'{v:.1f}' for v in s['speech_level_q'])} dBFS; "
                     f"AGC min gain p10/p50/p90 = {' / '.join(f'{v:.1f}' for v in s['agc_min_gain_q'])} dB; "
                     f"clipped samples {s['clip_pct_mean']:.3f} % on average, {s['clip_gt_0.1_pct']:.0f} % of clips above 0.1 %\n")
        if split == "test" and baseline:
            from .metrics import evaluate_set, results_table
            csv = d / "baseline_metrics.csv"
            if verbose:
                print("baseline metrics on test (unprocessed boom channel) ...")
            df = evaluate_set(d, channel=0)
            df.to_csv(csv, index=False)
            parts.append("### Baseline: unprocessed boom channel vs target (SI-SNR dB / STOI / PESQ-WB)\n")
            parts.append("```\n" + results_table(df) + "\n```\n")
    rep = root / "BUILD_REPORT.md"
    rep.write_text("\n".join(parts), encoding="utf-8")
    if verbose:
        print(f"report -> {rep}")
    return rep
