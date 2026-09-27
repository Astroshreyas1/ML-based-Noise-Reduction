"""Score a HubNet checkpoint on a built v3.1 split (default: the whole test split).

    .venv/Scripts/python.exe model/evaluate_hub.py --ckpt model/runs/hub1/ckpt_best.pt [--split test] [--kw]

Deliverable metrics (user decisions 2026-09-27), vs the radio-band clean target, VOX-muted head excluded:
    SNR   = SI-SNR (dB)                   target > 15
    STOI  = STOI at 16 kHz                target > 0.85
    PESQ  = PESQ-NB, ITU-T P.862 at 8 kHz target > 2.5      (PESQ-WB also reported)
--kw adds keyword recall: faster-whisper transcribes input / output / target; recall of the
military / radio keywords present in the target transcript.
Writes <run>/eval_hub_<split>.csv and .md (overall, by link, by SNR bin, by scenario, by corpus).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))

from data_hub import HubPairs  # noqa: E402
from hub_net import HubNet  # noqa: E402
from train_hub import pesq_nb, si_snr  # noqa: E402


def pesq_wb(est: np.ndarray, ref: np.ndarray) -> float:
    from pesq import pesq
    try:
        return float(pesq(16000, ref, est, "wb"))
    except Exception:
        return float("nan")


def main() -> None:
    from pystoi import stoi
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int)
    ap.add_argument("--kw", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device)
    hp = ck["hp"]
    net = HubNet(hp["c"], hp["n_blocks"], hp["h_intra"], hp["h_inter"], hp["df_order"]).to(device).eval()
    net.load_state_dict(ck["model"])
    ds = HubPairs(Path(hp["data_root"]) / a.split, None)
    n = min(a.n or len(ds), len(ds))
    wm = None
    if a.kw:
        from ancdata.speech_extra import cuda12_dlls
        cuda12_dlls()
        from faster_whisper import WhisperModel

        from ancdata.radio_vocab import KEYWORDS
        wm = WhisperModel("small.en", device="cuda" if device.type == "cuda" else "cpu",
                          compute_type="float16" if device.type == "cuda" else "int8")

        def words(x: np.ndarray) -> list[str]:
            segs, _ = wm.transcribe(x.astype(np.float32), language="en", beam_size=1, condition_on_previous_text=False)
            return [re.sub(r"[^a-z\-]", "", w.lower()) for s in segs for w in s.text.split()]
    rows = []
    with torch.no_grad():
        for i in range(n):
            d = ds.item(i)
            m = ds.meta[i]
            x = torch.from_numpy(d["noisy"])[None].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                y, _ = net(x)
            y = y.float()[0].cpu().numpy()
            c, v, nz = d["clean"], d["valid"], d["noisy"]
            r = {"id": d["id"], "link": d["link"], "snr_lufs_db": d["snr"], "scenario": d["scenario"],
                 "corpus": m.get("speech_corpus"),
                 "si_snr_in": float(si_snr(torch.from_numpy(nz), torch.from_numpy(c), torch.from_numpy(v))),
                 "si_snr": float(si_snr(torch.from_numpy(y), torch.from_numpy(c), torch.from_numpy(v))),
                 "stoi_in": stoi(c * v, nz * v, 16000), "stoi": stoi(c * v, y * v, 16000),
                 "pesq_nb_in": pesq_nb(nz * v, c * v), "pesq_nb": pesq_nb(y * v, c * v), "pesq_wb": pesq_wb(y * v, c * v)}
            if wm is not None:
                ref_w = [w for w in words(c) if w in KEYWORDS]
                if ref_w:
                    for tag, sig in (("in", nz), ("out", y)):
                        got = set(words(sig))
                        r[f"kw_recall_{tag}"] = float(np.mean([w in got for w in ref_w]))
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{n}", flush=True)
    df = pd.DataFrame(rows)
    out = a.ckpt.parent
    df.to_csv(out / f"eval_hub_{a.split}.csv", index=False)
    cols = [c for c in ("si_snr_in", "si_snr", "stoi_in", "stoi", "pesq_nb_in", "pesq_nb", "pesq_wb",
                        "kw_recall_in", "kw_recall_out") if c in df.columns]
    L = [f"# {a.ckpt} on {a.split} ({n} pairs)", "",
         "Targets: SI-SNR > 15 dB, STOI > 0.85, PESQ-NB > 2.5 (radio-band target, VOX-muted head excluded).", "",
         "## Overall\n\n" + df[cols].mean().round(3).to_frame("mean").to_markdown() + "\n",
         "## By link\n\n" + df.groupby("link")[cols].mean().round(3).to_markdown() + "\n"]
    df["snr_bin"] = pd.cut(df.snr_lufs_db, [-9, -4, 0, 4, 8, 12, 16])
    L.append("## By drawn SNR\n\n" + df.groupby("snr_bin", observed=True)[cols].mean().round(3).to_markdown() + "\n")
    L.append("## By scenario\n\n" + df.groupby("scenario")[cols].mean().round(3).to_markdown() + "\n")
    L.append("## By speech corpus\n\n" + df.groupby("corpus")[cols].mean().round(3).to_markdown() + "\n")
    (out / f"eval_hub_{a.split}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:5]))


if __name__ == "__main__":
    main()
