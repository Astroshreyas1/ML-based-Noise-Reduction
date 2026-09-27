"""Quick speed / memory check on this machine before a long run: times training steps at a few batch
sizes with and without gradient checkpointing and prints the projected wall-clock for the default schedule.

    python -m hubnet.bench
"""
from __future__ import annotations

import time

import torch

from .hub_net import HubNet
from .train import HP, HubLoss


def main() -> None:
    dev = torch.device("cuda")
    hp = HP()
    vram = torch.cuda.get_device_properties(dev).total_memory / 2 ** 30
    print(f"{torch.cuda.get_device_name(dev)}, {vram:.0f} GiB, torch {torch.__version__}")
    crit = HubLoss(hp).to(dev)
    for ckpt in (False, True):
        for b in (8, 16, 32, 64):
            net = HubNet(hp.c, hp.n_blocks, hp.h_intra, hp.h_inter, hp.df_order).to(dev).train()
            net.checkpoint = ckpt
            opt = torch.optim.AdamW(net.parameters(), 1e-4)
            x = torch.randn(b, 32000, device=dev)
            one = torch.ones_like(x)
            try:
                torch.cuda.reset_peak_memory_stats()
                for i in range(6):
                    if i == 2:
                        torch.cuda.synchronize()
                        t = time.time()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        y, _ = net(x)
                    L = crit(y, x, one, torch.zeros_like(x))["total"]
                    opt.zero_grad()
                    L.backward()
                    opt.step()
                torch.cuda.synchronize()
                dt = (time.time() - t) / 4
                steps = hp.samples / b
                print(f"checkpointing={ckpt!s:5} batch {b:3d}: {dt * 1000:6.0f} ms/step, "
                      f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:5.1f} GiB -> {steps * dt / 3600:5.1f} h for the schedule")
            except torch.OutOfMemoryError:
                print(f"checkpointing={ckpt!s:5} batch {b:3d}: OOM")
            del net, opt
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
