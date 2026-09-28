"""Where does HubNet lose? Per-regime breakdown + the radio link's own ceiling.

A  ceiling   clean speech (the test targets) sent through the link with no acoustic noise: the best SI-SNR /
             PESQ-NB any enhancer can expect if it cannot undo the codec. Per link, with and without bit
             errors / RF noise.
B  model     every test pair: input vs output SI-SNR / STOI / PESQ-NB, broken down by link, SNR bin,
             scenario, speech corpus, gunfire, VOX clipping.
C  data      the same test pairs read from the 8 kHz repack (what the workstation trains on) vs the 16 kHz
             original, to see whether the repack costs anything.

    python scripts/diagnose_hubnet.py --ckpt runs/a40_hub1/ckpt_best.pt --out runs/a40_hub1/diagnose
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ship" / "hubnet_standalone"))
from ancdata.radio import radio_link  # noqa: E402
from hubnet.data import HubPairs  # noqa: E402
from hubnet.hub_net import HubNet  # noqa: E402
from hubnet.train import pesq_nb, si_snr  # noqa: E402


def sisnr(y: np.ndarray, c: np.ndarray, v: np.ndarray) -> float:
    return float(si_snr(torch.from_numpy(y), torch.from_numpy(c), torch.from_numpy(v)))


def ceiling(ds: HubPairs, n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
    for i in idx:
        c = ds.item(int(i))["clean"]
        for link in ("cvsd16", "cvsd32", "fm"):
            for ideal in (True, False):
                p = {"link_p": {link: 1.0}, "vox_p": 0.0, "squelch_p": 0.0}
                if ideal:
                    p.update(ber_good=(0.0, 0.0), ber_bad=(0.0, 0.0), bad_occupancy=(0.0, 0.0), fm_cnr_db=(60.0, 60.0))
                y, t, _, _ = radio_link(c, c, np.random.default_rng(int(rng.integers(2 ** 31))), p)
                one = np.ones_like(t)
                rows.append({"link": link, "channel": "ideal" if ideal else "realistic", "si_snr": sisnr(y, t, one),
                             "pesq_nb": pesq_nb(y, t)})
    return pd.DataFrame(rows)


@torch.no_grad()
def run_model(net, ds: HubPairs, dev, idx=None) -> pd.DataFrame:
    from pystoi import stoi
    rows = []
    for i in (range(len(ds)) if idx is None else idx):
        d = ds.item(i)
        nz, c, v = d["noisy"], d["clean"], d["valid"]
        y = net(torch.from_numpy(nz)[None].to(dev))[0][0].float().cpu().numpy()
        m = ds.meta[i]
        r = m.get("radio") or {}
        rows.append({"id": d["id"], "link": d["link"], "snr": d["snr"], "scenario": d["scenario"], "corpus": d["corpus"],
                     "gunfire": bool(m.get("gunfire")), "vox": bool(r.get("mute_span")),
                     "si_snr_in": sisnr(nz, c, v), "si_snr": sisnr(y, c, v),
                     "stoi_in": stoi(c * v, nz * v, 16000), "stoi": stoi(c * v, y * v, 16000),
                     "pesq_nb_in": pesq_nb(nz * v, c * v), "pesq_nb": pesq_nb(y * v, c * v)})
    return pd.DataFrame(rows)


def table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by)
    t = g[["si_snr_in", "si_snr", "stoi_in", "stoi", "pesq_nb_in", "pesq_nb"]].mean()
    t.insert(0, "n", g.size())
    t["gain_db"] = t.si_snr - t.si_snr_in
    return t.round(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "battlefield_v31")
    ap.add_argument("--data8k", type=Path, default=ROOT / "data" / "battlefield_v31_8k")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-ceiling", type=int, default=120)
    ap.add_argument("--n-8k", type=int, default=300)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    hp = ck["hp"]
    net = HubNet(hp["c"], hp["n_blocks"], hp["h_intra"], hp["h_inter"], hp["df_order"])
    net.load_state_dict(ck["model"])
    net = net.to(dev).eval()
    ds = HubPairs(a.data / "test", None)
    rep = [f"# HubNet diagnosis: {a.ckpt} (step {ck.get('step')})\n"]

    ce = ceiling(ds, a.n_ceiling, np.random.default_rng(0))
    ce.to_csv(a.out / "ceiling.csv", index=False)
    t = ce.groupby(["link", "channel"])[["si_snr", "pesq_nb"]].agg(["mean", "median"]).round(2)
    rep += ["## A. Link ceiling (clean speech through the link, no acoustic noise)\n", t.to_markdown(), ""]
    print(t, flush=True)

    df = run_model(net, ds, dev)
    df.to_csv(a.out / "test_rows.csv", index=False)
    df["snr_bin"] = pd.cut(df.snr, [-99, -4, 0, 4, 8, 99], labels=["<-4", "-4..0", "0..4", "4..8", ">8"])
    rep += ["## B. Model on the whole test split\n", table(df.assign(all="all"), "all").to_markdown(), ""]
    for by in ("link", "snr_bin", "scenario", "corpus", "gunfire", "vox"):
        t = table(df, by)
        rep += [f"### by {by}\n", t.to_markdown(), ""]
        print(t, flush=True)

    idx = list(np.random.default_rng(1).choice(len(ds), min(a.n_8k, len(ds)), replace=False))
    d8 = run_model(net, HubPairs(a.data8k / "test", None), dev, idx)
    d16 = df.iloc[idx]
    c = pd.DataFrame({"16k original": d16[["si_snr_in", "si_snr", "pesq_nb_in", "pesq_nb"]].mean().values,
                      "8k repack": d8[["si_snr_in", "si_snr", "pesq_nb_in", "pesq_nb"]].mean().values},
                     index=["si_snr_in", "si_snr", "pesq_nb_in", "pesq_nb"]).round(3)
    rep += [f"## C. 16 kHz original vs 8 kHz repack ({len(idx)} test pairs)\n", c.to_markdown(), ""]
    print(c, flush=True)
    (a.out / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    print(json.dumps({"written": str(a.out / "REPORT.md")}))


if __name__ == "__main__":
    main()
