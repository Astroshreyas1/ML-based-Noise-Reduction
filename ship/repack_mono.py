"""Repack data/battlefield_v31 for a small disk: noisy -> boom channel only (the hub model is single-channel),
optionally everything at 8 kHz (the model only processes 0-4 kHz; the loader upsamples back to 16 kHz).

    python ship/repack_mono.py --src data/battlefield_v31 --dst data/battlefield_v31_mono [--sr 16000] [--workers 6]
"""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def one(args):
    src, dst, sr = args
    x, fs = sf.read(str(src), dtype="float32", always_2d=True)
    y = x[:, 0]
    if sr != fs:
        y = resample_poly(y, sr, fs).astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst.with_suffix(".flac")), np.clip(y, -1, 1), sr, subtype="PCM_16")
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs = []
    for split in ("train", "val", "test"):
        (a.dst / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.src / split / "meta.jsonl", a.dst / split / "meta.jsonl")
        for sub in ("noisy", "clean"):
            for f in (a.src / split / sub).iterdir():
                d = a.dst / split / sub / f.name
                if not d.with_suffix(".flac").exists():
                    jobs.append((f, d, a.sr))
    for extra in ("words", "DATASET.md", "battlefield_v31.yaml"):
        s = a.src / extra
        if s.is_dir():
            shutil.copytree(s, a.dst / extra, dirs_exist_ok=True)
        elif s.exists():
            shutil.copy2(s, a.dst / extra)
    print(f"{len(jobs)} files to write", flush=True)
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for _ in ex.map(one, jobs, chunksize=64):
            done += 1
            if done % 10000 == 0:
                print(done, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
