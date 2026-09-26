"""Freeze N pairs to disk for evaluation.

Layout::

    <out>/
      noisy/000000.wav     (2-channel float32: boom, ref)
      clean/000000.wav     (mono float32 dry target)
      meta.jsonl           one JSON object per pair: snr, room, events, ...
      config.yaml          the exact config used

Metrics are comparable across weeks only if the eval set never changes;
that is why it is written once with a fixed seed and then left alone.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .audio import write_wav
from .chain import Chain, Pair
from .config import load_config
from .paths import eval_dir
from .stream import build_chain


def pair_meta(pair: Pair) -> dict[str, Any]:
    m = {k: v for k, v in pair.meta.items() if not k.startswith("_")}
    m["events"] = [
        {"start": e.start, "end": e.end, "category": e.category, "ratio_db": e.ratio_db,
         "local_snr_db": e.local_snr_db, **{f"ev_{k}": v for k, v in e.params.items()}}
        for e in pair.events
    ]
    return m


def _write_pair(out: Path, k: int, pair: Pair, fmt: str, stems: bool) -> str:
    write_wav(out / "noisy" / f"{k:06d}.wav", pair.noisy, fmt=fmt)
    write_wav(out / "clean" / f"{k:06d}.wav", pair.clean, fmt=fmt)
    if stems and "_stems" in pair.meta:
        for name, x in pair.meta["_stems"].items():
            write_wav(out / "stems" / f"{k:06d}_{name}.wav", x.astype("float32"), fmt=fmt)
    return json.dumps({"id": f"{k:06d}", **pair_meta(pair)}, default=float)


def _worker(args):
    cfg, split, out, indices, fmt, stems = args
    chain = build_chain(cfg, split)
    return [_write_pair(Path(out), k, chain.generate(k), fmt, stems) for k in indices]


def materialize(cfg: dict[str, Any] | str | Path, split: str, n: int, out: Path | None = None,
                start_index: int = 0, chain: Chain | None = None, verbose: bool = True,
                fmt: str = "float", stems: bool = False, workers: int = 1) -> Path:
    """fmt='float' for frozen eval sets (exact); 'flac' or 'pcm16' for large training dumps.
    stems=True also writes speech / per-layer noise / events stems for inspection.
    workers>1 splits the index range across processes (generation is index-deterministic,
    so the result is identical to a single process)."""
    cfg_path = None if isinstance(cfg, dict) else Path(cfg)
    if not isinstance(cfg, dict):
        cfg = load_config(cfg)
    if stems:
        cfg = dict(cfg, debug_stems=True)
    out = Path(out) if out else eval_dir(Path(cfg["_path"]).stem.replace("eval_", ""))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already exists; eval sets are frozen. Delete it explicitly to regenerate.")
    (out / "noisy").mkdir(parents=True, exist_ok=True)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    if stems:
        (out / "stems").mkdir(exist_ok=True)
    indices = list(range(start_index, start_index + n))
    lines: list[str] = []
    if workers > 1 and chain is None:
        from concurrent.futures import ProcessPoolExecutor
        chunks = [indices[i::workers] for i in range(workers)]
        with ProcessPoolExecutor(workers) as ex:
            for res in tqdm(ex.map(_worker, [(cfg, split, str(out), c, fmt, stems) for c in chunks]),
                            total=workers, desc=f"materialize {out.name} ({workers} workers)", disable=not verbose):
                lines.extend(res)
        lines.sort()
    else:
        chain = chain or build_chain(cfg, split)
        for k in tqdm(indices, desc=f"materialize {out.name}", disable=not verbose):
            lines.append(_write_pair(out, k, chain.generate(k), fmt, stems))
    with (out / "meta.jsonl").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    if cfg_path:
        shutil.copy2(cfg_path, out / "config.yaml")
    # registered input ranges for the demo-day capture check
    from .demo_check import registered_ranges
    registered_ranges(sorted((out / "noisy").glob("*.wav")) + sorted((out / "noisy").glob("*.flac")), out / "ranges.json")
    if verbose:
        print(f"wrote {n} pairs -> {out}")
    return out
