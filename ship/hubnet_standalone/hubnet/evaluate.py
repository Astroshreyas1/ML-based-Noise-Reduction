"""Score a checkpoint on the whole test split against the deliverable targets.

    python -m hubnet.evaluate --data /path/battlefield_v31 --ckpt runs/hub1/ckpt_best.pt [--split test]

SI-SNR (dB), STOI (16 kHz), PESQ-NB (8 kHz, ITU P.862) + PESQ-WB, vs the radio-band target, VOX-muted
head excluded. Targets: SI-SNR > 15, STOI > 0.85, PESQ-NB > 2.5. Writes <ckpt dir>/eval_<split>.{csv,md}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import HubPairs
from .hub_net import HubNet
from .train import pesq_nb, si_snr


def pesq_wb(est: np.ndarray, ref: np.ndarray) -> float:
    from pesq import pesq
    try:
        return float(pesq(16000, ref, est, "wb"))
    except Exception:
        return float("nan")


def main() -> None:
    from pystoi import stoi
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device)
    hp = ck["hp"]
    net = HubNet(hp["c"], hp["n_blocks"], hp["h_intra"], hp["h_inter"], hp["df_order"]).to(device).eval()
    net.load_state_dict(ck["model"])
    ds = HubPairs(a.data / a.split, None)
    n = min(a.n or len(ds), len(ds))
    rows = []
    with torch.no_grad():
        for i in range(n):
            d = ds.item(i)
            x = torch.from_numpy(d["noisy"])[None].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                y, _ = net(x)
            y = y.float()[0].cpu().numpy()
            c, v, nz = d["clean"], d["valid"], d["noisy"]
            rows.append({"id": d["id"], "link": d["link"], "snr_lufs_db": d["snr"], "scenario": d["scenario"],
                         "corpus": d["corpus"],
                         "si_snr_in": float(si_snr(torch.from_numpy(nz), torch.from_numpy(c), torch.from_numpy(v))),
                         "si_snr": float(si_snr(torch.from_numpy(y), torch.from_numpy(c), torch.from_numpy(v))),
                         "stoi_in": stoi(c * v, nz * v, 16000), "stoi": stoi(c * v, y * v, 16000),
                         "pesq_nb_in": pesq_nb(nz * v, c * v), "pesq_nb": pesq_nb(y * v, c * v),
                         "pesq_wb": pesq_wb(y * v, c * v)})
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{n}", flush=True)
    df = pd.DataFrame(rows)
    out = a.ckpt.parent
    df.to_csv(out / f"eval_{a.split}.csv", index=False)
    cols = ["si_snr_in", "si_snr", "stoi_in", "stoi", "pesq_nb_in", "pesq_nb", "pesq_wb"]
    ok = {"si_snr": df.si_snr.mean() > 15, "stoi": df.stoi.mean() > 0.85, "pesq_nb": df.pesq_nb.mean() > 2.5}
    L = [f"# {a.ckpt} on {a.split} ({n} pairs)", "",
         "Targets (whole split mean): " + ", ".join(f"{k} {'MET' if v else 'NOT met'}" for k, v in ok.items()), "",
         "## Overall\n\n" + df[cols].mean().round(3).to_frame("mean").to_markdown() + "\n",
         "## By link\n\n" + df.groupby("link")[cols].mean().round(3).to_markdown() + "\n"]
    df["snr_bin"] = pd.cut(df.snr_lufs_db, [-9, -4, 0, 4, 8, 12, 16])
    L.append("## By drawn SNR\n\n" + df.groupby("snr_bin", observed=True)[cols].mean().round(3).to_markdown() + "\n")
    L.append("## By scenario\n\n" + df.groupby("scenario")[cols].mean().round(3).to_markdown() + "\n")
    L.append("## By speech corpus\n\n" + df.groupby("corpus")[cols].mean().round(3).to_markdown() + "\n")
    (out / f"eval_{a.split}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:5]))


if __name__ == "__main__":
    main()
