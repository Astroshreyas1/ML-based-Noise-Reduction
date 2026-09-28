"""Fine-tune Laya on radio-hub typed decisions (single GPU or torchrun DDP).

Adapted from Laya's own notebook (notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb): RLCD =
GRPO-style policy gradient over Gaussian-perturbed logits with a proper-scoring-rule reward, plus soft
cross-entropy; encoder lr 2.5e-5, head 1e-4; temperature calibration on a held-out slice at the end.

    python finetune/train_laya.py --data finetune/data --out artefacts/laya_radio [--epochs 3]
    python finetune/eval_laya.py  --data finetune/data --model artefacts/laya_radio      # base vs fine-tuned
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch


def build_items(rows, tok, cfg):
    from laya.common import QTYPES, build_sequence, render_options
    items = []
    for row in rows:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, q in questions.items():
            if qid not in gold:
                continue
            t, crit, g = q["type"], q.get("criteria", {}), gold[qid]
            if t == "choice":
                target = [g["probabilities"].get(k, 0.0) for k in crit]
            elif t == "noul":
                target = [g["probabilities"].get("false", 0.5), g["probabilities"].get("true", 0.5)]
            else:
                n = len(crit) if isinstance(crit, list) else 4
                target = [g["probabilities"].get(str(i), 0.0) for i in range(n)]
            s = sum(target)
            target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
            seq, markers = build_sequence(tok, state, {"t": t, "ins": q["instructions"], "crit": crit}, cfg["max_len"],
                                          cfg["head_max_len"])
            if len(markers) != len(render_options({"t": t, "crit": crit})):
                continue
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target,
                          "label": target.index(max(target))})
    return items


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax))
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"])
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "target": target,
            "qtype": torch.tensor([it["qtype"] for it in items])}


def fit_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0))


def main() -> None:
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model, proper_reward
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="convaiinnovations/laya")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--no-ckpt", action="store_true", help="disable gradient checkpointing (big GPUs: faster)")
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    device = torch.device("cuda")
    model_dir = a.base if os.path.isdir(a.base) else snapshot_download(a.base, allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"])
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    cfg.update(gradient_checkpointing=True, max_tokens_per_batch=4096, max_len=min(cfg.get("max_len", 512), 512),
               head_max_len=256)
    rows = [json.loads(l) for l in (a.data / "train.jsonl").open(encoding="utf-8")]
    items = build_items(rows, tok, cfg)
    order = list(range(len(items)))
    random.Random(20260927).shuffle(order)
    n_cal = min(400, len(items) // 10)
    cal = [items[i] for i in order[:n_cal]]
    train = [items[i] for i in order[n_cal:]]
    mine = train[rank::world]
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    if not a.no_ckpt and torch.cuda.get_device_properties(0).total_memory < 30 * 2 ** 30:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device).train()
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True) if ddp else model
    enc = [p for n, p in net.named_parameters() if "encoder." in n]
    head = [p for n, p in net.named_parameters() if "encoder." not in n]
    opt = torch.optim.AdamW([{"params": enc, "lr": 2.5e-5}, {"params": head, "lr": 1e-4}], weight_decay=0.01)
    total = max(1, (len(mine) // (a.micro * a.accum)) * a.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    G, S0, S1 = 4, 0.4, 0.1
    if rank == 0:
        print(f"{len(train)} train items ({len(cal)} calibration) | {a.epochs} epochs | world {world}", flush=True)
    t0 = time.time()
    for ep in range(a.epochs):
        random.Random(42 + ep + rank).shuffle(mine)
        sigma = S0 + (S1 - S0) * ep / max(1, a.epochs - 1)
        opt.zero_grad(set_to_none=True)
        for bi, s in enumerate(range(0, len(mine), a.micro)):
            b = collate(mine[s: s + a.micro], tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = net(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                                  b["marker_mask"].to(device), b["qtype"].to(device))
            logits = logits.float()
            mask = b["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = b["target"].to(device)
            eps = torch.randn((G,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), b["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = (r - r.mean(0, keepdim=True)) / (r.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss = (-(adv * logp).mean() - (target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()) / a.accum \
                + 0.0 * act.sum()
            scaler.scale(loss).backward()
            if (bi + 1) % a.accum == 0 or s + a.micro >= len(mine):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if rank == 0 and (bi + 1) % 100 == 0:
                print(f"epoch {ep + 1} batch {bi + 1}/{math.ceil(len(mine) / a.micro)} loss {loss.item() * a.accum:.4f} "
                      f"reward {r.mean().item():.3f} {time.time() - t0:.0f}s", flush=True)
    if rank == 0:
        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(cal), 16):
                chunk = cal[s: s + 16]
                b = collate(chunk, tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    lz, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                                  b["marker_mask"].to(device), b["qtype"].to(device))
                lz = lz.float().cpu().numpy()
                for i, it in enumerate(chunk):
                    preds.append((it["qtype"], lz[i, : len(it["markers"])], it["target"]))
        temps = [fit_temp([(z, t) for q, z, t in preds if q == qt]) for qt in range(3)]
        a.out.mkdir(parents=True, exist_ok=True)
        save_file({k2: v.half().contiguous().cpu() for k2, v in model.state_dict().items()}, str(a.out / "model.safetensors"))
        model.encoder.config.save_pretrained(str(a.out / "encoder"))
        tok.save_pretrained(str(a.out / "tokenizer"))
        cfg.update(fine_tuned=True, model_name="laya-radio-hub", temperature=temps)
        cfg.pop("temperature_by_options", None)
        (a.out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"saved {a.out} | temperatures {temps} | {time.time() - t0:.0f}s", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
