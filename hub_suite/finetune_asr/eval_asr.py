"""Map-facing ASR evaluation: the deployed decoder (hubsuite.stt.FinalASR: faster-whisper, beam 5, radio
prompt + hotwords, loop collapse) on the held-out test set (unseen TTS voices + real radio clips), HubNet
output as input.

Reports per model: WER (sim / real), grid exact (the true grid among extracted grids), grid digit error rate,
wrong grids that would be CONFIRMED at the map threshold, and a threshold sweep so CONFIRM_P can be re-set.
Also the hallucination rate on noise-only radio audio (any word output on a transmission with no speech).

    python finetune_asr/eval_asr.py --data data_asr --models large-v3-turbo runs_asr/turbo/ct2 --out runs_asr/eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (loads torch's pip cuBLAS/cuDNN first, so CTranslate2 finds them on Linux)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hubsuite.radiotext import extract  # noqa: E402
from hubsuite.situation import _grid_confidence  # noqa: E402
from hubsuite.stt import FinalASR  # noqa: E402
from train_whisper import audio_path, load_manifest, norm_words, read16, wer  # noqa: E402


def digit_err(truth: str, got: list[str]) -> float:
    if not got:
        return 1.0
    best = min(got, key=lambda g: wer(list(truth), list(g))[0])
    return wer(list(truth), list(best))[0] / len(truth)


def noise_only(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """3 s of battlefield noise (half with gunfire) through the radio link with no talker: any word whisper
    prints here is invented."""
    from hubsuite.radio_link import radio_link
    from hubsuite.sim import battlefield_noise
    out = []
    for i in range(n):
        noise = battlefield_noise(3 * 16000, rng, gunfire=bool(i % 2)) * 0.02
        y, _, _, _ = radio_link(noise.astype(np.float32), np.zeros_like(noise, dtype=np.float32), rng,
                                {"link_p": {"cvsd16": 0.5, "fm": 0.5}, "vox_p": 0.0})
        out.append(y)
    return out


def run(model: str, data: Path, recs: list[dict], noise: list[np.ndarray], enh: bool) -> dict:
    asr = FinalASR(model, "cuda")
    rows = []
    for r in recs:
        text, dt, words = asr.transcribe(read16(audio_path(data, "test", r, enh)))
        row = {"id": r["id"], "src": r["src"], "ref": r["text"], "hyp": text, "ms": dt * 1000}
        e, n = wer(norm_words(r["text"]), norm_words(text))
        row["err"], row["n"] = e, n
        if r.get("grids"):
            ex = extract(text)
            t = r["grids"][-1]
            conf = _grid_confidence(ex["grids"], words) if words else {}
            row.update(truth_grid=t, grids=ex["grids"], hit=t in ex["grids"], digit_err=digit_err(t, ex["grids"]),
                       wrong=[(g, conf.get(g)) for g in ex["grids"] if g != t and len(g) == len(t)])
        rows.append(row)
    halluc = sum(1 for y in noise if norm_words(asr.transcribe(y)[0])) / max(1, len(noise))
    g = [r for r in rows if "truth_grid" in r]
    res = {"model": model, "input": "hubnet" if enh else "radio",
           "wer_sim": sum(r["err"] for r in rows if r["src"] == "sim") / max(1, sum(r["n"] for r in rows if r["src"] == "sim")),
           "wer_real": sum(r["err"] for r in rows if r["src"] == "real") / max(1, sum(r["n"] for r in rows if r["src"] == "real")),
           "grid_exact": float(np.mean([r["hit"] for r in g])) if g else None,
           "grid_digit_err": float(np.mean([r["digit_err"] for r in g])) if g else None,
           "n_grid": len(g), "halluc_noise_only": halluc,
           "median_ms": float(np.median([r["ms"] for r in rows]))}
    wrong_conf = [c for r in g for _, c in r["wrong"] if c is not None]
    res["wrong_confirmed@"] = {f"{p:.2f}": sum(c >= p for c in wrong_conf) for p in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9)}
    res["n_wrong_grids"] = len(wrong_conf)
    return {"summary": res, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-noise", type=int, default=60)
    ap.add_argument("--radio-too", action="store_true", help="also evaluate on the raw radio input (no HubNet)")
    a = ap.parse_args()
    recs = load_manifest(a.data, "test")
    noise = noise_only(a.n_noise, np.random.default_rng(7))
    out = []
    for m in a.models:
        for enh in ([True, False] if a.radio_too else [True]):
            r = run(m, a.data, recs, noise, enh)
            print(json.dumps(r["summary"]), flush=True)
            out.append(r)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
