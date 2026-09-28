"""Fine-tune whisper on hub audio (radio link -> HubNet) so grids reach the map intact.

Data: finetune_asr/make_asr_data.py manifests ({split}_sim.jsonl, {split}_real.jsonl). Each draw picks the
HubNet output (p = --p-enh) or the raw radio audio, so the ASR does not lock onto one HubNet checkpoint.
Validation every --val-every steps on held-out TTS voices + real radio clips: WER, and the map metric
(grid exact = the true grid is among the grids hubsuite.radiotext.extract pulls from the hypothesis).
Best checkpoint by grid exact, then WER; saved as a HuggingFace model, then converted for faster-whisper:

    python finetune_asr/train_whisper.py --data data_asr --out runs_asr/turbo --base openai/whisper-large-v3-turbo
    ct2-transformers-converter --model runs_asr/turbo/best --output_dir runs_asr/turbo/ct2 --quantization float16 \
        --copy_files tokenizer.json preprocessor_config.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hubsuite.radiotext import extract  # noqa: E402

SR = 16000


def load_manifest(data: Path, split: str, srcs=("sim", "real")) -> list[dict]:
    out = []
    for s in srcs:
        p = data / f"{split}_{s}.jsonl"
        if p.exists():
            out += [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return out


def read16(path: Path) -> np.ndarray:
    x, fs = sf.read(str(path), dtype="float32")
    return x if fs == SR else resample_poly(x, SR // fs, 1).astype(np.float32)


def audio_path(data: Path, split: str, rec: dict, enh: bool) -> Path:
    p = data / split / ("enh" if enh else "radio") / f"{rec['id']}.flac"
    return p if p.exists() else data / split / "radio" / f"{rec['id']}.flac"


class ASRSet(Dataset):
    def __init__(self, data: Path, split: str, recs: list[dict], proc, p_enh: float, train: bool, p_neg: float = 0.0):
        self.data, self.split, self.recs, self.proc, self.p_enh, self.train = data, split, recs, proc, p_enh, train
        self.p_neg = p_neg

    def __len__(self) -> int:
        return len(self.recs)

    def _noise_only(self) -> np.ndarray:
        """Battlefield noise through the radio link, no talker; level spans raw radio down to HubNet's residual."""
        from hubsuite.radio_link import radio_link
        from hubsuite.sim import battlefield_noise
        rng = np.random.default_rng(random.getrandbits(63))
        n = int(rng.uniform(1.0, 8.0) * SR)
        x = (battlefield_noise(n, rng, gunfire=bool(rng.random() < 0.5)) * 0.05).astype(np.float32)
        y, _, _, _ = radio_link(x, np.zeros_like(x), rng, {"link_p": {"cvsd16": 0.55, "fm": 0.35, "cvsd32": 0.10}})
        return (y * 10 ** (rng.uniform(-30, 0) / 20)).astype(np.float32)

    def __getitem__(self, i: int) -> dict:
        r = self.recs[i]
        if self.train and random.random() < self.p_neg:          # nothing said -> nothing written
            r, x = {"text": ""}, self._noise_only()
        else:
            enh = random.random() < self.p_enh if self.train else True
            x = read16(audio_path(self.data, self.split, r, enh))
        if self.train:
            x = x * 10 ** (random.uniform(-6, 6) / 20)                          # input level
        feats = self.proc.feature_extractor(x, sampling_rate=SR, return_tensors="np").input_features[0]
        ids = self.proc.tokenizer(r["text"]).input_ids
        if ids and ids[0] == self.proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"):
            ids = ids[1:]                                                        # the model prepends it
        return {"feats": torch.from_numpy(feats), "labels": torch.tensor(ids[:440]), "i": i}


def collate(b: list[dict]) -> dict:
    L = max(len(x["labels"]) for x in b)
    lab = torch.full((len(b), L), -100, dtype=torch.long)
    for k, x in enumerate(b):
        lab[k, : len(x["labels"])] = x["labels"]
    return {"feats": torch.stack([x["feats"] for x in b]), "labels": lab, "i": [x["i"] for x in b]}


def norm_words(t: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", t.lower().replace("-", " ")).split()


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)], len(ref)


def grid_hit(rec: dict, hyp: str) -> tuple[int, int]:
    if not rec.get("grids"):
        return 0, 0
    got = extract(hyp)["grids"]
    return int(rec["grids"][-1] in got), 1


@torch.no_grad()
def validate(model, proc, ds: ASRSet, dev, bs: int = 16) -> dict:
    model.eval()
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, collate_fn=collate)
    e = n = gh = gn = 0
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(b["feats"].to(dev),
                                 language="en", task="transcribe", num_beams=1, max_new_tokens=160)
        for i, hyp in zip(b["i"], proc.batch_decode(out, skip_special_tokens=True)):
            r = ds.recs[i]
            de, dn = wer(norm_words(r["text"]), norm_words(hyp))
            e, n = e + de, n + dn
            h, k = grid_hit(r, hyp)
            gh, gn = gh + h, gn + k
    model.train()
    return {"wer": e / max(n, 1), "grid_exact": gh / max(gn, 1), "n_grid": gn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--p-enh", type=float, default=0.75)
    ap.add_argument("--p-neg", type=float, default=0.05, help="share of noise-only items with an empty transcript")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--freeze-encoder", action="store_true", help="decoder only (small GPUs)")
    a = ap.parse_args()
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.manual_seed(0)
    random.seed(0)
    dev = torch.device("cuda")
    proc = WhisperProcessor.from_pretrained(a.base)
    proc.tokenizer.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(a.base, torch_dtype=torch.float32).to(dev)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if a.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    tr = load_manifest(a.data, "train")
    va = load_manifest(a.data, "val")
    va = random.Random(1).sample(va, min(a.val_n, len(va)))
    ts = ASRSet(a.data, "train", tr, proc, a.p_enh, True, a.p_neg)
    vs = ASRSet(a.data, "val", va, proc, 1.0, False)
    dl = DataLoader(ts, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True,
                    persistent_workers=True, pin_memory=True)
    steps = int(math.ceil(a.epochs * len(dl) / a.accum))
    warm = max(50, steps // 20)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, steps - warm)))))
    a.out.mkdir(parents=True, exist_ok=True)
    log = (a.out / "log.jsonl").open("a", encoding="utf-8")
    base = validate(model, proc, vs, dev)
    print(f"train {len(tr)} val {len(va)} steps {steps} | base: {base}", flush=True)
    log.write(json.dumps({"step": 0, **base}) + "\n")
    best = (base["grid_exact"], -base["wer"])
    step, k, t0, run = 0, 0, time.time(), 0.0
    model.train()
    while step < steps:
        for b in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_features=b["feats"].to(dev), labels=b["labels"].to(dev)).loss / a.accum
            loss.backward()
            run += float(loss.detach()) * a.accum
            k += 1
            if k % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                dt = (time.time() - t0) / 50
                t0 = time.time()
                print(f"step {step}/{steps} loss {run / 50:.4f} lr {sched.get_last_lr()[0]:.2e} {dt:.2f} s/step "
                      f"ETA {dt * (steps - step) / 3600:.2f} h", flush=True)
                run = 0.0
            if step % a.val_every == 0 or step == steps:
                m = validate(model, proc, vs, dev)
                print(f"  VAL step {step}: {m}", flush=True)
                log.write(json.dumps({"step": step, **m}) + "\n")
                log.flush()
                if (m["grid_exact"], -m["wer"]) > best:
                    best = (m["grid_exact"], -m["wer"])
                    model.save_pretrained(a.out / "best",            # fp16 on disk: the shared disk is small
                                          state_dict={k: v.half() for k, v in model.state_dict().items()})
                    proc.save_pretrained(a.out / "best")
                    print("  saved best", flush=True)
            if step >= steps:
                break
    (a.out / "done.json").write_text(json.dumps({"best_grid_exact": best[0], "best_wer": -best[1], "base": base}))


if __name__ == "__main__":
    main()
