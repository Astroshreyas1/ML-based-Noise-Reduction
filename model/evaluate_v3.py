"""Score a checkpoint on a built split (default: data/battlefield_v3/test).

    .venv/Scripts/python.exe model/evaluate_v3.py --ckpt model/runs/v3a/ckpt_best.pt [--split test] [--n 300]

Per pair: SI-SNR (input and output), STOI, PESQ-WB, event-head recall / precision at
the frame level; tables by scenario, by drawn SNR bin and by transient local SNR,
plus the "blackout" rate (speech frames inside event spans whose output energy is
> 20 dB below the target's). Writes <run>/eval_<split>.csv and eval_<split>.md.
The whole 6 s clip goes through the network in one call; the model is causal with
an explicit state, so this equals frame-by-frame streaming (checked in the selftest
of the export, not here).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))

from anc_net_v3 import ANCNetV3  # noqa: E402
from data import PairSet  # noqa: E402
from losses import si_snr_db  # noqa: E402
from ancdata.metrics import pesq_wb, stoi as stoi_fn  # noqa: E402

HOP = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, help="first N pairs only")
    ap.add_argument("--no-pesq", action="store_true")
    ap.add_argument("--single-channel", action="store_true", help="zero the ref at test time")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device)
    hp = ck["hp"]
    net = ANCNetV3(tuple(hp["widths"]), hp["gru_hidden"], hp["gru_layers"], frontend_pr=hp.get("frontend_pr", False)).to(device).eval()
    net.load_state_dict(ck["model"])
    ds = PairSet(ROOT / "data" / "battlefield_v3" / a.split, crop_seconds=None)
    n = min(a.n or len(ds), len(ds))
    rows = []
    with torch.no_grad():
        for i in range(n):
            it = ds[i]
            noisy = it["noisy"][None].to(device)
            if a.single_channel:
                noisy[:, 1] = 0
            clean = it["clean"][None].to(device)
            est, ev_logits, gain, _ = net(noisy)
            est_np, clean_np, in_np = est[0].cpu().numpy(), clean[0].cpu().numpy(), noisy[0, 0].cpu().numpy()
            fc = it["frame_class"].numpy()
            pred = ev_logits[0].argmax(-1).cpu().numpy()[: len(fc)]
            pos, ppos = fc > 0, pred > 0
            m = ds.meta[i]
            # blackout: frames inside event windows where target has speech energy but output lost > 20 dB
            evw = it["ev_window"].numpy()
            t = len(fc)
            tgt_e = (clean_np[: t * HOP].reshape(t, HOP) ** 2).mean(-1)
            out_e = (est_np[: t * HOP].reshape(t, HOP) ** 2).mean(-1)
            win_f = evw[: t * HOP].reshape(t, HOP).max(-1) > 0
            speech_f = tgt_e > 1e-6
            sel = win_f & speech_f
            blackout = float(np.mean((out_e[sel] < tgt_e[sel] * 1e-2))) if sel.any() else np.nan
            r = {"id": it["id"], "scenario": it["scenario"], "snr_lufs_db": it["snr"], "gunfire": bool(m["gunfire"]),
                 "n_blasts": m["n_blasts"],
                 "min_local_snr_db": min([e["local_snr_db"] for e in m.get("events", []) if e["category"] != "hard_negative"], default=np.nan),
                 "si_snr_in": float(si_snr_db(noisy[:, 0], clean)[0]), "si_snr": float(si_snr_db(est, clean)[0]),
                 "stoi_in": stoi_fn(in_np, clean_np), "stoi": stoi_fn(est_np, clean_np),
                 "event_recall": float(((ppos & pos).sum() / max(1, pos.sum()))) if pos.any() else np.nan,
                 "event_precision": float(((ppos & pos).sum() / max(1, ppos.sum()))) if ppos.any() else np.nan,
                 "false_alarm_frames": float(np.mean(ppos & ~pos)), "blackout": blackout,
                 "gain_min": float(gain.min())}
            if not a.no_pesq:
                try:
                    r["pesq_in"] = pesq_wb(in_np, clean_np)
                    r["pesq"] = pesq_wb(est_np, clean_np)
                except Exception:
                    r["pesq_in"] = r["pesq"] = np.nan
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{n}", flush=True)
    df = pd.DataFrame(rows)
    df["si_snri"] = df.si_snr - df.si_snr_in
    df["stoi_i"] = df.stoi - df.stoi_in
    out = a.ckpt.parent
    df.to_csv(out / f"eval_{a.split}.csv", index=False)
    cols = [c for c in ("si_snr_in", "si_snr", "si_snri", "stoi_in", "stoi", "pesq_in", "pesq", "event_recall", "event_precision",
                        "false_alarm_frames", "blackout") if c in df.columns]
    lines = [f"# {a.ckpt} on {a.split} ({n} pairs){' single-channel' if a.single_channel else ''}", ""]
    lines.append("## Overall\n\n" + df[cols].mean().round(3).to_frame("mean").to_markdown() + "\n")
    lines.append("## By scenario\n\n" + df.groupby("scenario")[cols].mean().round(3).to_markdown() + "\n")
    df["snr_bin"] = pd.cut(df.snr_lufs_db, [-9, -4, 0, 4, 8, 12, 16])
    lines.append("## By drawn SNR (K-weighted)\n\n" + df.groupby("snr_bin", observed=True)[cols].mean().round(3).to_markdown() + "\n")
    tr = df[df.min_local_snr_db.notna()]
    if len(tr):
        tr = tr.assign(tsnr=pd.cut(tr.min_local_snr_db, [-60, -10, 0, 6, 60]))
        lines.append("## By transient local SNR (clips with gunfire / blasts)\n\n" + tr.groupby("tsnr", observed=True)[cols].mean().round(3).to_markdown() + "\n")
    lines.append(f"## Gunfire vs none\n\n" + df.groupby("gunfire")[cols].mean().round(3).to_markdown() + "\n")
    (out / f"eval_{a.split}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4]))
    print(f"-> {out / f'eval_{a.split}.md'}")


if __name__ == "__main__":
    main()
