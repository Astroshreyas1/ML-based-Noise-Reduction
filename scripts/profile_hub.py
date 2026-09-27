"""Where does a HubNet training step go? Times each stage with CUDA sync (forward pieces, loss, backward),
checkpointing on/off, and the data loader alone.  .venv/Scripts/python.exe scripts/profile_hub.py"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))

from hub_net import F_USE, HubNet  # noqa: E402
from train_hub import HP, HubLoss  # noqa: E402


def timed(fn, n=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def main():
    dev = torch.device("cuda")
    hp = HP()
    B, N = hp.batch, int(hp.crop_s * 16000)
    x = torch.randn(B, N, device=dev)
    y = torch.randn(B, N, device=dev)
    v = torch.ones(B, N, device=dev)
    kw = torch.zeros(B, N, device=dev)
    crit = HubLoss(hp).to(dev)
    for ckpt in (True, False):
        net = HubNet(hp.c, hp.n_blocks, hp.h_intra, hp.h_inter, hp.df_order).to(dev).train()
        net.checkpoint = ckpt
        opt = torch.optim.AdamW(net.parameters(), 1e-4)

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                est, _ = net(x)
            L = crit(est, y, v, kw)
            opt.zero_grad()
            L["total"].backward()
            opt.step()

        def fwd():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                net(x)
        try:
            torch.cuda.reset_peak_memory_stats()
            print(f"checkpoint={ckpt}: full step {timed(step):.0f} ms, forward only {timed(fwd):.0f} ms, "
                  f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
        except torch.OutOfMemoryError:
            print(f"checkpoint={ckpt}: OOM")
        del net, opt
        torch.cuda.empty_cache()

    # per-module forward+backward, checkpoint off
    net = HubNet(hp.c, hp.n_blocks, hp.h_intra, hp.h_inter, hp.df_order).to(dev).train()
    net.checkpoint = False
    X = net.stft_.stft(x)[:, :, :F_USE]
    feat = net.features(X)
    e1 = net.enc1(feat).detach().requires_grad_()
    e2 = net.enc2(e1).detach().requires_grad_()
    blk = net.blocks[0]
    b, c, t, f = e2.shape
    print(f"tensor (B, C, T, F) = {tuple(e2.shape)}: intra-GRU sequences {b * t} x len {f}; inter-GRU sequences {b * f} x len {t}")

    def run(fn):
        def g():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = fn()
            (o[0] if isinstance(o, tuple) else o).float().sum().backward()
        return timed(g)
    z = e2.permute(0, 2, 3, 1).reshape(b * t, f, c).detach().requires_grad_()
    zi = e2.permute(0, 3, 2, 1).reshape(b * f, t, c).detach().requires_grad_()
    print(f"  intra bi-GRU (1 block)  {run(lambda: blk.intra(z)):.0f} ms")
    print(f"  inter GRU (1 block)     {run(lambda: blk.inter(zi)):.0f} ms")
    print(f"  whole DP block          {run(lambda: blk(e2)):.0f} ms  x {len(net.blocks)} blocks")
    print(f"  encoder                 {run(lambda: net.enc2(net.enc1(feat))):.0f} ms")
    est = torch.randn(B, N, device=dev, requires_grad=True)
    print(f"  loss (MR-STFT etc.)     {run(lambda: crit(est, y, v, kw)['total']):.0f} ms")

    # data loader alone
    from torch.utils.data import DataLoader

    from data_hub import HubPairs, load_words
    from train_hub import collate
    ts = HubPairs(Path(hp.data_root) / "train", hp.crop_s, words=load_words(ROOT / "data"))
    for w in (4, 6):
        dl = DataLoader(ts, batch_size=B, shuffle=True, num_workers=w, collate_fn=collate, drop_last=True)
        it = iter(dl)
        next(it)
        t0 = time.time()
        for _ in range(30):
            next(it)
        print(f"data loader, {w} workers: {(time.time() - t0) / 30 * 1000:.0f} ms / batch")


if __name__ == "__main__":
    main()
