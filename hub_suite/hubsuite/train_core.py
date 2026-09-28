"""Train HubNet (strict real-time radio-hub speech enhancer, 18 ms) -- standalone workstation trainer.

    python -m hubnet.train --data /path/battlefield_v31 --out runs/hub1
    python -m hubnet.train --data /path/battlefield_v31 --out runs/hub1 --resume

Auto-sizing (override with flags): the schedule is fixed in AUDIO seen (default 1.6 M two-second crops =
888 h, the laptop plan of 200k steps x batch 8). Batch = 8 per 6 GB of free VRAM (capped 64); steps =
samples / batch; LR scales with sqrt(batch / 8). Gradient checkpointing only when VRAM < 16 GB (it
recomputes the forward pass: ~30 % slower, 3x less memory). Multi-GPU: torchrun --nproc_per_node=K
(DDP; batch is per GPU, steps shrink by K).

Loss (radio-band target, VOX-muted head excluded):
    -SI-SNR/10 + MR-STFT on |X|^0.3 (L1 + spectral convergence, 3 resolutions)
    + 0.3 * compressed complex L1 + 0.5 * keyword-window SI-SNR (military / radio vocabulary)
Validation every --val-every steps: SI-SNR, STOI (16 kHz), PESQ-NB (8 kHz, ITU P.862) vs the radio-band
target. Best checkpoint by SI-SNR + 10*STOI + 3*PESQ-NB. Deliverable targets: >15 dB, >0.85, >2.5.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from .pairs import HubPairs, collate, load_words
from .hub_net import HubNet


@dataclass
class HP:
    batch: int = 8
    crop_s: float = 2.0
    samples: int = 1_600_000             # total training crops (all GPUs); steps = samples / (batch * world)
    steps: int = 0                       # derived
    lr: float = 6e-4                     # at batch 8; scaled by sqrt(batch_total / 8)
    lr_min: float = 1e-5
    warmup_frac: float = 0.01
    weight_decay: float = 1e-2
    grad_clip: float = 5.0
    workers: int = 8
    val_every: int = 4000
    val_pairs: int = 192
    val_pesq_pairs: int = 64
    checkpointing: bool = True
    seed: int = 0
    c: int = 128
    n_blocks: int = 6
    h_intra: int = 64
    h_inter: int = 256
    df_order: int = 3
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

    def forward(self, est, ref, valid, kw):
        hp = self.hp
        est, ref = est.float() * valid, ref.float() * valid
        out = {"sisnr": -si_snr(est, ref).mean() / 10}
        stft, ri = 0.0, 0.0
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
        has = kw.sum(-1) > 0.05 * kw.shape[-1]
        out["kw"] = -si_snr(est[has], ref[has], kw[has]).mean() / 10 if has.any() else est.new_zeros(())
        out["total"] = hp.w_sisnr * out["sisnr"] + hp.w_stft * out["stft"] + hp.w_ri * out["ri"] + hp.w_kw * out["kw"]
        return out


def pesq_nb(est: np.ndarray, ref: np.ndarray) -> float:
    from pesq import pesq
    from scipy.signal import resample_poly
    try:
        return float(pesq(8000, resample_poly(ref, 1, 2), resample_poly(est, 1, 2), "nb"))
    except Exception:
        return float("nan")


def validate(net, vs: HubPairs, hp: HP, device) -> dict[str, float]:
    from pystoi import stoi
    net.eval()
    r = {"sisnr": [], "sisnr_in": [], "stoi": [], "pesq_nb": [], "pesq_nb_in": []}
    with torch.no_grad():
        for i in range(len(vs)):
            d = vs.item(i)
            x = torch.from_numpy(d["noisy"])[None].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                y, _ = net(x)
            y = y.float()[0].cpu().numpy()
            v, c, nz = d["valid"], d["clean"], d["noisy"]
            r["sisnr"].append(float(si_snr(torch.from_numpy(y), torch.from_numpy(c), torch.from_numpy(v))))
            r["sisnr_in"].append(float(si_snr(torch.from_numpy(nz), torch.from_numpy(c), torch.from_numpy(v))))
            if i < hp.val_pesq_pairs:
                r["stoi"].append(stoi(c * v, y * v, 16000, extended=False))
                r["pesq_nb"].append(pesq_nb(y * v, c * v))
                r["pesq_nb_in"].append(pesq_nb(nz * v, c * v))
    net.train()
    return {k: float(np.nanmean(v)) for k, v in r.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="folder with train/ val/ test/ words/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch", type=int, help="per GPU (default: auto from VRAM)")
    ap.add_argument("--samples", type=int, help="total training crops (default 1.6 M)")
    ap.add_argument("--steps", type=int, help="override the derived step count")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--val-every", type=int)
    ap.add_argument("--checkpointing", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (Linux; try it, ~10-30 % faster)")
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = HP()
    vram = torch.cuda.get_device_properties(device).total_memory / 2 ** 30 if device.type == "cuda" else 0
    hp.batch = a.batch or int(min(64, 8 * max(1, round(vram / 6))))
    hp.checkpointing = {"on": True, "off": False}.get(a.checkpointing, vram < 16)
    if a.samples:
        hp.samples = a.samples
    if a.workers is not None:
        hp.workers = a.workers
    if a.val_every:
        hp.val_every = a.val_every
    hp.steps = a.steps or int(math.ceil(hp.samples / (hp.batch * world)))
    lr = hp.lr * math.sqrt(hp.batch * world / 8)
    warmup = max(500, int(hp.warmup_frac * hp.steps))
    main_proc = rank == 0

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(hp.seed + rank)
    np.random.seed(hp.seed + rank)
    words = load_words(a.data / "words")
    ts = HubPairs(a.data / "train", hp.crop_s, words=words)
    vs = HubPairs(a.data / "val", None, limit=hp.val_pairs, seed=1, words=words)
    sampler = DistributedSampler(ts, shuffle=True, seed=hp.seed) if ddp else None
    dl = DataLoader(ts, batch_size=hp.batch, shuffle=sampler is None, sampler=sampler, num_workers=hp.workers,
                    drop_last=True, collate_fn=collate, persistent_workers=hp.workers > 0, pin_memory=True)
    net = HubNet(hp.c, hp.n_blocks, hp.h_intra, hp.h_inter, hp.df_order).to(device)
    net.checkpoint = hp.checkpointing
    raw = net
    if a.compile:
        net = torch.compile(net)
    if ddp:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device.index])
    crit = HubLoss(hp).to(device)
    opt = torch.optim.AdamW(raw.parameters(), lr=lr, weight_decay=hp.weight_decay)
    step, best = 0, -1e9
    if a.resume and (out / "ckpt_last.pt").exists():
        ck = torch.load(out / "ckpt_last.pt", map_location=device)
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck.get("best", -1e9)
    if main_proc:
        (out / "hp.json").write_text(json.dumps({**asdict(hp), "lr_scaled": lr, "warmup": warmup, "world": world,
                                                 "vram_gib": vram}, indent=1), encoding="utf-8")
        print(f"HubNet {sum(p.numel() for p in raw.parameters()) / 1e6:.2f} M params | GPU {vram:.0f} GiB x {world} | "
              f"batch {hp.batch}/GPU | steps {hp.steps} | lr {lr:.2e} | checkpointing {hp.checkpointing} | "
              f"train {len(ts)} val {len(vs)}", flush=True)
        log = (out / "log.csv").open("a", newline="", encoding="utf-8")
        wr = csv.writer(log)
        if step == 0:
            wr.writerow(["step", "lr", "loss", "sisnr_loss", "stft", "ri", "kw", "val_sisnr", "val_sisnr_in", "val_stoi",
                         "val_pesq_nb", "val_pesq_nb_in", "sec_per_step"])

    def lr_at(s: int) -> float:
        if s < warmup:
            return lr * (s + 1) / warmup
        p = (s - warmup) / max(1, hp.steps - warmup)
        return hp.lr_min + 0.5 * (lr - hp.lr_min) * (1 + math.cos(math.pi * min(1.0, p)))

    t0 = time.time()
    agg: dict[str, float] = {}
    nagg, epoch = 0, 0
    while step < hp.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
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
            torch.nn.utils.clip_grad_norm_(raw.parameters(), hp.grad_clip)
            opt.step()
            step += 1
            for k, val in L.items():
                agg[k] = agg.get(k, 0.0) + float(val.detach())
            nagg += 1
            if main_proc and step % 200 == 0:
                dt = (time.time() - t0) / 200
                t0 = time.time()
                eta_h = dt * (hp.steps - step) / 3600
                print(f"step {step}/{hp.steps} loss {agg['total'] / nagg:.4f} sisnr {-10 * agg['sisnr'] / nagg:.2f} dB "
                      f"kw {-10 * agg['kw'] / nagg:.2f} dB  {dt:.3f} s/step  ETA {eta_h:.1f} h", flush=True)
                if step % hp.val_every != 0:
                    wr.writerow([step, lr_at(step), agg["total"] / nagg, agg["sisnr"] / nagg, agg["stft"] / nagg,
                                 agg["ri"] / nagg, agg["kw"] / nagg, "", "", "", "", "", dt])
                    log.flush()
                    agg, nagg = {}, 0
            if main_proc and (step % hp.val_every == 0 or step == hp.steps):
                m = validate(raw, vs, hp, device)
                score = m["sisnr"] + 10 * m["stoi"] + 3 * m["pesq_nb"]
                print(f"  VAL step {step}: SI-SNR {m['sisnr']:.2f} (in {m['sisnr_in']:.2f}) STOI {m['stoi']:.3f} "
                      f"PESQ-NB {m['pesq_nb']:.2f} (in {m['pesq_nb_in']:.2f})", flush=True)
                wr.writerow([step, lr_at(step), agg.get("total", 0) / max(nagg, 1), agg.get("sisnr", 0) / max(nagg, 1),
                             agg.get("stft", 0) / max(nagg, 1), agg.get("ri", 0) / max(nagg, 1), agg.get("kw", 0) / max(nagg, 1),
                             m["sisnr"], m["sisnr_in"], m["stoi"], m["pesq_nb"], m["pesq_nb_in"], ""])
                log.flush()
                agg, nagg = {}, 0
                ck = {"model": raw.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                      "hp": asdict(hp), "val": m}
                if score > best:
                    best = score
                    ck["best"] = best
                    torch.save(ck, out / "ckpt_best.pt")
                torch.save(ck, out / "ckpt_last.pt")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
