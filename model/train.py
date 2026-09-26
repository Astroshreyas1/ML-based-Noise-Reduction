"""Train ANC-Net v3 on data/battlefield_v3.

    .venv/Scripts/python.exe model/train.py --name v3a --steps 40000
    .venv/Scripts/python.exe model/train.py --name v3a --resume            # continues from the last checkpoint

Hyper-parameters (HP below) are a first guess for a 6 GB laptop GPU (RTX 4050):
batch 8 x 4 s crops, AdamW 5e-4 with 1 k warm-up and cosine to 1e-5, bf16 autocast,
grad-clip 5, ref-channel augmentation (docs/MODEL_V3_IDEATION.md §4) on the GPU,
validation every 2 k steps on 192 fixed val pairs (SI-SNR improvement, STOI on 48).
Losses: model/losses.py. Logs: model/runs/<name>/log.csv, checkpoints ckpt_last.pt / ckpt_best.pt.
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

from anc_net_v3 import ANCNetV3, count_params  # noqa: E402
from data import PairSet  # noqa: E402
from losses import ANCLoss, LossWeights, si_snr_db  # noqa: E402


@dataclass
class HP:
    data_root: str = str(ROOT / "data" / "battlefield_v3")
    batch: int = 8
    crop_s: float = 4.0
    steps: int = 40000
    lr: float = 5e-4
    lr_min: float = 1e-5
    warmup: int = 1000
    weight_decay: float = 1e-2
    grad_clip: float = 5.0
    amp: str = "bf16"                      # bf16 | fp16 | none
    workers: int = 4
    val_every: int = 2000
    val_pairs: int = 192
    val_stoi_pairs: int = 48
    seed: int = 0
    # model
    widths: tuple = (16, 32, 48, 64)
    gru_hidden: int = 128
    gru_layers: int = 2
    frontend_pr: bool = True               # perfect-reconstruction synthesis window (v3a was trained with False)
    # ref-channel augmentation
    ref_dropout_p: float = 0.3
    ref_gain_db: float = 6.0
    ref_delay_max: int = 5
    ref_noise_p: float = 0.5
    ref_noise_db: tuple = (-6.0, 0.0)
    ref_noise_hz: float = 1500.0
    # loss weights
    w_sisnr: float = 1.0
    w_mrstft: float = 1.0
    w_ri: float = 0.3
    w_mel: float = 0.2
    w_event: float = 0.3
    w_evwin: float = 0.3
    w_gain: float = 0.05


def augment_ref(noisy: torch.Tensor, hp: HP, gen: torch.Generator) -> torch.Tensor:
    """noisy (B, 2, N) on device. Randomises the reference channel so a model cannot
    learn v3's synthetic ref verbatim: dropout, gain, integer delay, coherence-breaking
    noise above ref_noise_hz."""
    b, _, n = noisy.shape
    dev = noisy.device
    ref = noisy[:, 1].clone()
    drop = torch.rand(b, generator=gen, device=dev) < hp.ref_dropout_p
    gain = 10 ** ((torch.rand(b, generator=gen, device=dev) * 2 - 1) * hp.ref_gain_db / 20)
    for i in range(b):
        d = int(torch.randint(-hp.ref_delay_max, hp.ref_delay_max + 1, (1,), generator=gen, device=dev))
        if d:
            ref[i] = torch.roll(ref[i], d)
    ref = ref * gain[:, None]
    add = torch.rand(b, generator=gen, device=dev) < hp.ref_noise_p
    if add.any():
        white = torch.randn(b, n, generator=gen, device=dev)
        spec = torch.fft.rfft(white)
        freqs = torch.fft.rfftfreq(n, 1 / 16000).to(dev)
        spec = spec * (freqs >= hp.ref_noise_hz)
        hp_noise = torch.fft.irfft(spec, n)
        lvl = 10 ** ((hp.ref_noise_db[0] + torch.rand(b, generator=gen, device=dev) * (hp.ref_noise_db[1] - hp.ref_noise_db[0])) / 20)
        ref_rms = ref.pow(2).mean(-1).sqrt() + 1e-6
        hp_noise = hp_noise / (hp_noise.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6) * (ref_rms * lvl)[:, None]
        ref = ref + hp_noise * add[:, None].float()
    ref = ref * (~drop)[:, None].float()
    return torch.stack([noisy[:, 0], ref.clamp(-1, 1)], 1)


def lr_at(step: int, hp: HP) -> float:
    if step < hp.warmup:
        return hp.lr * (step + 1) / hp.warmup
    p = min(1.0, (step - hp.warmup) / max(1, hp.steps - hp.warmup))
    return hp.lr_min + 0.5 * (hp.lr - hp.lr_min) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def validate(net, loader, device, n_stoi: int) -> dict[str, float]:
    from pystoi import stoi
    net.eval()
    si, si_in, stois, per_scn = [], [], [], {}
    done_stoi = 0
    for batch in loader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        est, _, _, _ = net(noisy)
        s = si_snr_db(est.float(), clean).cpu().numpy()
        s0 = si_snr_db(noisy[:, 0].float(), clean).cpu().numpy()
        si.extend(s.tolist())
        si_in.extend(s0.tolist())
        for scn, a, b in zip(batch["scenario"], s, s0):
            per_scn.setdefault(scn, []).append(float(a - b))
        if done_stoi < n_stoi:
            e_np, c_np = est.float().cpu().numpy(), clean.cpu().numpy()
            for k in range(len(e_np)):
                if done_stoi >= n_stoi:
                    break
                stois.append(stoi(c_np[k], e_np[k], 16000, extended=False))
                done_stoi += 1
    net.train()
    out = {"val_sisnr": float(np.mean(si)), "val_sisnr_in": float(np.mean(si_in)),
           "val_sisnri": float(np.mean(si) - np.mean(si_in)), "val_stoi": float(np.mean(stois)) if stois else float("nan")}
    out["val_sisnri_by_scenario"] = {k: round(float(np.mean(v)), 2) for k, v in sorted(per_scn.items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v3a")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--single-channel", action="store_true", help="ref always zero (ablation)")
    ap.add_argument("--val-every", type=int)
    ap.add_argument("--val-pairs", type=int)
    ap.add_argument("--lr", type=float)
    a = ap.parse_args()
    hp = HP()
    if a.steps:
        hp.steps = a.steps
    if a.batch:
        hp.batch = a.batch
    if a.workers is not None:
        hp.workers = a.workers
    if a.single_channel:
        hp.ref_dropout_p = 1.0
    if a.val_every:
        hp.val_every = a.val_every
    if a.val_pairs:
        hp.val_pairs = a.val_pairs
        hp.val_stoi_pairs = min(hp.val_stoi_pairs, a.val_pairs)
    if a.lr:
        hp.lr = a.lr
    run = ROOT / "model" / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "hp.json").write_text(json.dumps(asdict(hp), indent=1), encoding="utf-8")

    torch.manual_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}  {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    net = ANCNetV3(hp.widths, hp.gru_hidden, hp.gru_layers, frontend_pr=hp.frontend_pr).to(device)
    print(f"params {count_params(net):,}")
    loss_fn = ANCLoss(LossWeights(hp.w_sisnr, hp.w_mrstft, hp.w_ri, hp.w_mel, hp.w_event, hp.w_evwin, hp.w_gain)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=hp.lr, weight_decay=hp.weight_decay, betas=(0.9, 0.99))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(hp.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=(hp.amp == "fp16" and device.type == "cuda"))

    train_set = PairSet(Path(hp.data_root) / "train", crop_seconds=hp.crop_s)
    val_set = PairSet(Path(hp.data_root) / "val", crop_seconds=None, limit=hp.val_pairs, seed=1)
    train_loader = DataLoader(train_set, batch_size=hp.batch, shuffle=True, num_workers=hp.workers, drop_last=True,
                              pin_memory=(device.type == "cuda"), persistent_workers=hp.workers > 0)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0)
    print(f"train {len(train_set)} pairs, val {len(val_set)} pairs, {hp.steps} steps of batch {hp.batch} x {hp.crop_s} s")

    step, best = 0, -1e9
    log_path = run / "log.csv"
    if a.resume and (run / "ckpt_last.pt").exists():
        ck = torch.load(run / "ckpt_last.pt", map_location=device)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]; best = ck.get("best", best)
        print(f"resumed at step {step}")
    else:
        with log_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["step", "lr", "loss", "sisnr_db", "mrstft_mag", "mrstft_sc", "ri", "mel", "event_ce",
                                     "evwin_sisnr_db", "gain_hinge", "event_acc", "event_recall", "sec_per_step",
                                     "val_sisnr", "val_sisnri", "val_stoi"])
    gen = torch.Generator(device=device)
    gen.manual_seed(hp.seed + 17)
    net.train()
    acc: dict[str, list[float]] = {}
    t0 = time.time()
    while step < hp.steps:
        for batch in train_loader:
            if step >= hp.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, hp)
            noisy = augment_ref(batch["noisy"].to(device, non_blocking=True), hp, gen)
            clean = batch["clean"].to(device, non_blocking=True)
            frame_class = batch["frame_class"].to(device)
            ev_window = batch["ev_window"].to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
                est, ev_logits, gain, _ = net(noisy)
            loss, terms = loss_fn(est.float(), clean, ev_logits.float(), gain.float(), frame_class, ev_window)
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.unscale_(opt)
            else:
                loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), hp.grad_clip)
            if scaler.is_enabled():
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            step += 1
            terms["loss"] = float(loss.detach())
            for k, v in terms.items():
                acc.setdefault(k, []).append(v)
            if step % 50 == 0:
                sps = (time.time() - t0) / 50
                t0 = time.time()
                m = {k: float(np.mean(v)) for k, v in acc.items()}
                acc = {}
                print(f"step {step:6d} lr {lr_at(step, hp):.2e} loss {m['loss']:.3f} sisnr {m['sisnr_db']:5.2f} "
                      f"mag {m['mrstft_mag']:.3f} sc {m['mrstft_sc']:.3f} ri {m['ri']:.3f} mel {m['mel']:.3f} "
                      f"ce {m['event_ce']:.3f} acc {m['event_acc']:.2f} rec {m['event_recall']:.2f} evwin {m['evwin_sisnr_db']:5.2f} "
                      f"g {m['gain_hinge']:.3f} gn {float(gn):.1f} {sps * 1000:.0f} ms/step", flush=True)
                row = [step, lr_at(step, hp), m["loss"], m["sisnr_db"], m["mrstft_mag"], m["mrstft_sc"], m["ri"], m["mel"],
                       m["event_ce"], m["evwin_sisnr_db"], m["gain_hinge"], m["event_acc"], m["event_recall"], sps, "", "", ""]
                if step % hp.val_every == 0 or step == hp.steps:
                    v = validate(net, val_loader, device, hp.val_stoi_pairs)
                    print(f"  VAL step {step}: SI-SNR {v['val_sisnr']:.2f} dB (input {v['val_sisnr_in']:.2f}, "
                          f"improvement {v['val_sisnri']:.2f}), STOI {v['val_stoi']:.3f}; by scenario {v['val_sisnri_by_scenario']}", flush=True)
                    row[-3:] = [v["val_sisnr"], v["val_sisnri"], v["val_stoi"]]
                    ck = {"model": net.state_dict(), "opt": opt.state_dict(), "step": step, "hp": asdict(hp), "val": v, "best": best}
                    torch.save(ck, run / "ckpt_last.pt")
                    if v["val_sisnri"] > best:
                        best = v["val_sisnri"]; ck["best"] = best
                        torch.save(ck, run / "ckpt_best.pt")
                        print(f"  new best (SI-SNRi {best:.2f}) -> ckpt_best.pt", flush=True)
                with log_path.open("a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(row)
    print("done")


if __name__ == "__main__":
    main()
