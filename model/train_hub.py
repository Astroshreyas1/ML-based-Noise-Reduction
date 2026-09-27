"""Train HubNet (model/hub_net.py) on data/battlefield_v31 (radio-link input, radio-band target).

    .venv/Scripts/python.exe model/train_hub.py --name hub1 --steps 200000
    .venv/Scripts/python.exe model/train_hub.py --name hub1 --resume

Loss (band-limited target, VOX-muted head excluded everywhere):
    -SI-SNR / 10                                   the "SNR" deliverable, directly
    + multi-resolution STFT on |X|^0.3 (L1 + spectral convergence), 3 resolutions
    + 0.3 * compressed complex (RI) L1 at the model's own resolution   (phase: PESQ / STOI)
    + w_kw * (-SI-SNR / 10) on keyword windows      military / radio vocabulary (+-50 ms spans)
Metrics every `val_every` steps on fixed val pairs: SI-SNR (valid-masked), STOI (16 kHz, vs the
radio-band target), PESQ-NB (8 kHz, ITU P.862; the deliverable's PESQ) on a subset.
Checkpoint selection: val SI-SNR + 10 * STOI + 3 * PESQ-NB (all three targets matter).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))

from data_hub import HubPairs, load_words  # noqa: E402
from hub_net import HubNet  # noqa: E402


@dataclass
class HP:
    data_root: str = str(ROOT / "data" / "battlefield_v31")
    batch: int = 8
    crop_s: float = 2.0
    steps: int = 200000
    lr: float = 6e-4
    lr_min: float = 1e-5
    warmup: int = 2000
    weight_decay: float = 1e-2
    grad_clip: float = 5.0
    workers: int = 4
    val_every: int = 4000
    val_pairs: int = 192
    val_pesq_pairs: int = 64
    seed: int = 0
    # model
    c: int = 128
    n_blocks: int = 6
    h_intra: int = 64
    h_inter: int = 256
    df_order: int = 3
    # loss
    w_sisnr: float = 1.0
    w_stft: float = 1.0
    w_ri: float = 0.3
    w_kw: float = 0.5


def si_snr(est: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    if mask is not None:
        est, ref = est * mask, ref * mask
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    s = a * ref
    return 10 * torch.log10(s.pow(2).sum(-1) / ((est - s).pow(2).sum(-1) + eps) + eps)


class HubLoss(torch.nn.Module):
    RES = ((256, 64), (512, 128), (1024, 256))

    def __init__(self, hp: HP):
        super().__init__()
        self.hp = hp
        for i, (n, _) in enumerate(self.RES):
            self.register_buffer(f"win{i}", torch.hann_window(n))

    def _spec(self, x: torch.Tensor, i: int) -> torch.Tensor:
        n, h = self.RES[i]
        return torch.stft(x, n, h, window=getattr(self, f"win{i}"), return_complex=True)

    def forward(self, est: torch.Tensor, ref: torch.Tensor, valid: torch.Tensor, kw: torch.Tensor) -> dict[str, torch.Tensor]:
        hp = self.hp
        est, ref = est.float() * valid, ref.float() * valid
        out = {"sisnr": -si_snr(est, ref).mean() / 10}
        stft = 0.0
        ri = 0.0
        for i in range(len(self.RES)):
            E, R = self._spec(est, i), self._spec(ref, i)
            me, mr = E.abs().clamp_min(1e-7) ** 0.3, R.abs().clamp_min(1e-7) ** 0.3
            stft = stft + F.l1_loss(me, mr) + torch.norm(mr - me) / (torch.norm(mr) + 1e-7)
            if i == 1:
                ce = me * E / E.abs().clamp_min(1e-7)
                cr = mr * R / R.abs().clamp_min(1e-7)
                ri = F.l1_loss(torch.view_as_real(ce), torch.view_as_real(cr))
        out["stft"] = stft / len(self.RES)
        out["ri"] = ri
        has_kw = kw.sum(-1) > 0.05 * kw.shape[-1]
        if has_kw.any():
            out["kw"] = -si_snr(est[has_kw], ref[has_kw], kw[has_kw]).mean() / 10
        else:
            out["kw"] = est.new_zeros(())
        out["total"] = hp.w_sisnr * out["sisnr"] + hp.w_stft * out["stft"] + hp.w_ri * out["ri"] + hp.w_kw * out["kw"]
        return out


def collate(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        out[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else v
    return out


def pesq_nb(est: np.ndarray, ref: np.ndarray) -> float:
    from pesq import pesq
    from scipy.signal import resample_poly
    try:
        return float(pesq(8000, resample_poly(ref, 1, 2), resample_poly(est, 1, 2), "nb"))
    except Exception:
        return float("nan")


def validate(net: HubNet, vs: HubPairs, hp: HP, device: torch.device) -> dict[str, float]:
    from pystoi import stoi
    net.eval()
    res = {"sisnr": [], "sisnr_in": [], "stoi": [], "pesq_nb": [], "pesq_nb_in": []}
    with torch.no_grad():
        for i in range(len(vs)):
            d = vs.item(i)
            x = torch.from_numpy(d["noisy"])[None].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                y, _ = net(x)
            y = y.float()[0].cpu().numpy()
            v, c, nz = d["valid"], d["clean"], d["noisy"]
            res["sisnr"].append(float(si_snr(torch.from_numpy(y), torch.from_numpy(c), torch.from_numpy(v))))
            res["sisnr_in"].append(float(si_snr(torch.from_numpy(nz), torch.from_numpy(c), torch.from_numpy(v))))
            if i < hp.val_pesq_pairs:
                res["stoi"].append(stoi(c * v, y * v, 16000, extended=False))
                res["pesq_nb"].append(pesq_nb(y * v, c * v))
                res["pesq_nb_in"].append(pesq_nb(nz * v, c * v))
    net.train()
    return {k: float(np.nanmean(v)) for k, v in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--val-every", type=int)
    a = ap.parse_args()
    hp = HP()
    for k in ("steps", "batch", "lr"):
        if getattr(a, k) is not None:
            setattr(hp, k, getattr(a, k))
    if a.val_every:
        hp.val_every = a.val_every
    run = ROOT / "model" / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    words = load_words(ROOT / "data")
    ts = HubPairs(Path(hp.data_root) / "train", hp.crop_s, words=words)
    vs = HubPairs(Path(hp.data_root) / "val", None, limit=hp.val_pairs, seed=1, words=words)
    dl = DataLoader(ts, batch_size=hp.batch, shuffle=True, num_workers=hp.workers, drop_last=True,
                    collate_fn=collate, persistent_workers=False, pin_memory=True)   # fresh workers each epoch see reloaded words
    net = HubNet(hp.c, hp.n_blocks, hp.h_intra, hp.h_inter, hp.df_order).to(device)
    crit = HubLoss(hp).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    step, best = 0, -1e9
    if a.resume and (run / "ckpt_last.pt").exists():
        ck = torch.load(run / "ckpt_last.pt", map_location=device)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck.get("best", -1e9)
        print(f"resumed at step {step}")
    (run / "hp.json").write_text(json.dumps(asdict(hp), indent=1), encoding="utf-8")
    n_par = sum(p.numel() for p in net.parameters())
    print(f"HubNet {n_par / 1e6:.2f} M params, train {len(ts)} pairs, val {len(vs)}", flush=True)
    log = (run / "log.csv").open("a", newline="", encoding="utf-8")
    wr = csv.writer(log)
    if step == 0:
        wr.writerow(["step", "lr", "loss", "sisnr_loss", "stft", "ri", "kw", "val_sisnr", "val_sisnr_in", "val_stoi",
                     "val_pesq_nb", "val_pesq_nb_in", "sec_per_step"])

    def lr_at(s: int) -> float:
        if s < hp.warmup:
            return hp.lr * (s + 1) / hp.warmup
        p = (s - hp.warmup) / max(1, hp.steps - hp.warmup)
        return hp.lr_min + 0.5 * (hp.lr - hp.lr_min) * (1 + math.cos(math.pi * min(1.0, p)))

    t0 = time.time()
    agg: dict[str, float] = {}
    nagg = 0
    while step < hp.steps:
        for b in dl:
            if step >= hp.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x, y, v, kw = (b[k].to(device, non_blocking=True) for k in ("noisy", "clean", "valid", "kw"))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                est, _ = net(x)
            L = crit(est, y, v, kw)
            opt.zero_grad(set_to_none=True)
            L["total"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), hp.grad_clip)
            opt.step()
            step += 1
            for k, val in L.items():
                agg[k] = agg.get(k, 0.0) + float(val)
            nagg += 1
            if step % 200 == 0:
                dt = (time.time() - t0) / 200
                t0 = time.time()
                print(f"step {step} loss {agg['total'] / nagg:.4f} sisnr {-10 * agg['sisnr'] / nagg:.2f} dB "
                      f"kw {-10 * agg['kw'] / nagg:.2f} dB  {dt:.3f} s/step", flush=True)
                if step % hp.val_every != 0:
                    wr.writerow([step, lr_at(step), agg["total"] / nagg, agg["sisnr"] / nagg, agg["stft"] / nagg,
                                 agg["ri"] / nagg, agg["kw"] / nagg, "", "", "", "", "", dt])
                    log.flush()
                    agg, nagg = {}, 0
            if step % hp.val_every == 0:
                ts.words = vs.words = load_words(ROOT / "data")     # word timings may be refreshed during a run
                m = validate(net, vs, hp, device)
                score = m["sisnr"] + 10 * m["stoi"] + 3 * m["pesq_nb"]
                print(f"  VAL step {step}: SI-SNR {m['sisnr']:.2f} (in {m['sisnr_in']:.2f}) STOI {m['stoi']:.3f} "
                      f"PESQ-NB {m['pesq_nb']:.2f} (in {m['pesq_nb_in']:.2f})", flush=True)
                wr.writerow([step, lr_at(step), agg.get("total", 0) / max(nagg, 1), agg.get("sisnr", 0) / max(nagg, 1),
                             agg.get("stft", 0) / max(nagg, 1), agg.get("ri", 0) / max(nagg, 1), agg.get("kw", 0) / max(nagg, 1),
                             m["sisnr"], m["sisnr_in"], m["stoi"], m["pesq_nb"], m["pesq_nb_in"], ""])
                log.flush()
                agg, nagg = {}, 0
                ck = {"model": net.state_dict(), "opt": opt.state_dict(), "step": step, "best": best, "hp": asdict(hp),
                      "val": m}
                if score > best:
                    best = score
                    ck["best"] = best
                    torch.save(ck, run / "ckpt_best.pt")
                torch.save(ck, run / "ckpt_last.pt")
    log.close()


if __name__ == "__main__":
    main()
