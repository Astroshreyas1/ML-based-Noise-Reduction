"""Continual field adaptation (feature 4: "make it RL: crude synthetic pretraining, real training on incoming data").

Research (docs/RESEARCH_comfort_and_field_learning.md, section B) decides the shape:
* No clean reference exists in the field, and non-intrusive quality scores are gameable: in CHiME-7 UDASE,
  systems trained to MAXIMISE DNSMOS were WORST with listeners; only DNSMOS-BAK tracked listeners (r 0.73).
* HubNet is a deterministic regressor; policy-gradient RL needs a stochastic policy and a trustworthy
  reward, and optimising a reward invites "clean-sounding" hallucination -- unacceptable on a command net.
So the "reinforcement" is in the GATE, not the gradient: rewards decide WHICH field audio the model learns
from and WHETHER a candidate is promoted; the gradient is plain regression (RemixIT-style self-training,
adapted to the radio link) with >= 50 % synthetic replay so the synthetic pretraining is never forgotten.

Loop (nightly, or every ~5 h of speech-active traffic; never on the live GPU path):
    1. buffer      transmissions from the hub (squelch-open segments); noise-only spans harvested separately
    2. teacher     T = promoted model, E = EMA shadow; s_T = T(x), s_E = E(x)
    3. gate        keep a segment only if DNSMOS-BAK(s_T) - BAK(x) >= 0.5, SI-SNR(s_T, s_E) >= 15 dB,
                   and (optional) whisper keyword agreement(s_T, s_E) >= 0.9
    4. pairs       additive remix  s_T,i + (x_j - s_T,j)                    (receiver-side noise)
                   link re-degrade radio_link(s_T,i + harvested noise) -> s_T,i   (codec damage; non-additive)
                   synthetic replay v3.1 pairs with TRUE targets, >= 50 % of every batch
    5. student     S = copy(T), lr 1e-5, ~3k steps, HubLoss + 1e-3 * ||S - T||^2 (anchor)
    6. promote     only if the synthetic v3.1 probe does not regress (per metric thresholds), the output
                   invents no energy on noise-only probes, and (if available) the field probe improves;
                   then shadow-run before going live; every step logged; rollback = previous promoted model

    python -m hubsuite.adapt --buffer field_buffer/ --replay ../data/battlefield_v31 --registry artefacts/registry
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .hub_net import HubNet
from .pairs import HubPairs, collate, load_words
from .radio_link import radio_link
from .train_core import HP, HubLoss, pesq_nb, si_snr

SR = 16000
ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------------------
# Non-intrusive quality: DNSMOS P.835 (ONNX, Reddy et al. 2022). BAK is the only score that tracked
# listeners in UDASE; SIG / OVRL penalise the 3.4 kHz radio band itself, so only differences are used.
# --------------------------------------------------------------------------------------------------
class DNSMOS:
    P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    LEN = 144160

    def __init__(self, path: Path = ROOT / "artefacts" / "dnsmos" / "sig_bak_ovr.onnx"):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def __call__(self, x: np.ndarray) -> dict[str, float]:
        x = np.asarray(x, np.float32)
        while len(x) < self.LEN:
            x = np.concatenate([x, x])
        starts = range(0, max(1, len(x) - self.LEN + 1), SR)
        s = []
        for a in starts:
            seg = x[a: a + self.LEN][None]
            sig, bak, ovr = self.sess.run(None, {"input_1": seg})[0][0]
            s.append((self.P_SIG(sig), self.P_BAK(bak), self.P_OVR(ovr)))
        m = np.mean(s, axis=0)
        return {"sig": float(m[0]), "bak": float(m[1]), "ovrl": float(m[2])}


# --------------------------------------------------------------------------------------------------
# Registry of promoted models (rollback = previous entry)
# --------------------------------------------------------------------------------------------------
class Registry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "registry.json"
        self.entries = json.loads(self.index.read_text()) if self.index.exists() else []

    def promoted(self) -> Path | None:
        live = [e for e in self.entries if e["state"] == "promoted"]
        return Path(live[-1]["path"]) if live else None

    def add(self, path: Path, state: str, report: dict) -> None:
        self.entries.append({"path": str(path), "state": state, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "report": report})
        self.index.write_text(json.dumps(self.entries, indent=1), encoding="utf-8")

    def rollback(self) -> Path | None:
        live = [e for e in self.entries if e["state"] == "promoted"]
        if len(live) >= 2:
            live[-1]["state"] = "rolled_back"
            self.index.write_text(json.dumps(self.entries, indent=1), encoding="utf-8")
        return self.promoted()


def load_model(path: Path, device) -> HubNet:
    ck = torch.load(path, map_location=device)
    hp = ck["hp"]
    net = HubNet(hp["c"], hp["n_blocks"], hp["h_intra"], hp["h_inter"], hp["df_order"]).to(device).eval()
    net.load_state_dict(ck["model"])
    return net


@torch.no_grad()
def enhance(net: HubNet, x: np.ndarray, device) -> np.ndarray:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        y, _ = net(torch.from_numpy(np.asarray(x, np.float32))[None].to(device))
    return y.float()[0].cpu().numpy()


# --------------------------------------------------------------------------------------------------
# Field buffer: transmissions written by the hub (squelch-open segments), as 16 kHz wavs
# --------------------------------------------------------------------------------------------------
def energy_vad(x: np.ndarray, thr_db: float = -45.0, frame: int = 320) -> np.ndarray:
    k = len(x) // frame
    e = 10 * np.log10(np.mean(x[: k * frame].reshape(k, frame) ** 2, axis=1) + 1e-12)
    return np.repeat(e > thr_db, frame)


def harvest(buffer: Path, seg_s: float = 6.0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(speech segments, noise-only segments) from the buffer's transmissions."""
    speech, noise = [], []
    n = int(seg_s * SR)
    for f in sorted(Path(buffer).glob("*.wav")):
        x, sr = sf.read(str(f), dtype="float32", always_2d=True)
        x = x[:, 0]
        if sr != SR:
            from scipy.signal import resample_poly
            x = resample_poly(x, SR, sr).astype(np.float32)
        for a in range(0, max(1, len(x) - n + 1), n):
            s = x[a: a + n]
            if len(s) < n:
                continue
            (speech if energy_vad(s).mean() > 0.3 else noise).append(s)
    return speech, noise


def gate(x: np.ndarray, s_t: np.ndarray, s_e: np.ndarray, dns: DNSMOS | None) -> tuple[bool, dict]:
    info = {"agree_db": float(si_snr(torch.from_numpy(s_t), torch.from_numpy(s_e)))}
    ok = info["agree_db"] >= 15.0
    if dns is not None:
        bx, bt = dns(x)["bak"], dns(s_t)["bak"]
        info.update(bak_in=bx, bak_out=bt)
        ok = ok and (bt - bx) >= 0.5
    return ok, info


# --------------------------------------------------------------------------------------------------
# One adaptation round
# --------------------------------------------------------------------------------------------------
def adapt_round(buffer: Path, replay_root: Path, registry: Path, teacher: Path | None = None, steps: int = 3000,
                batch: int = 8, device_str: str = "cuda", probe_pairs: int = 96, seed: int = 0) -> dict:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)
    reg = Registry(registry)
    tpath = teacher or reg.promoted()
    if tpath is None:
        raise SystemExit("no teacher: pass --teacher or promote a model first")
    T = load_model(tpath, device)
    E = load_model(Path(str(tpath).replace(".pt", "_ema.pt")), device) if Path(str(tpath).replace(".pt", "_ema.pt")).exists() else T
    try:
        dns = DNSMOS()
    except Exception:
        dns = None
    speech, noise = harvest(buffer)
    kept, log = [], []
    for x in speech:
        s_t = enhance(T, x, device)
        s_e = enhance(E, x, device) if E is not T else s_t
        ok, info = gate(x, s_t, s_e, dns)
        log.append({**info, "kept": ok})
        if ok:
            kept.append((x, s_t))
    report: dict = {"teacher": str(tpath), "segments": len(speech), "noise_segments": len(noise), "kept": len(kept)}
    if not kept and not noise:
        report["result"] = "nothing to learn from"
        return report
    words = load_words(replay_root / "words")
    replay = HubPairs(replay_root / "train", 2.0, words=words)
    hp = HP()
    crit = HubLoss(hp).to(device)
    S = copy.deepcopy(T).train()
    S.checkpoint = True
    anchor = [p.detach().clone() for p in T.parameters()]
    opt = torch.optim.AdamW(S.parameters(), lr=1e-5, weight_decay=0.0)
    crop = 2 * SR
    radio_params = {"link_p": {"cvsd16": 0.55, "fm": 0.35, "cvsd32": 0.10}}

    def field_pair() -> tuple[np.ndarray, np.ndarray]:
        x, s = kept[rng.randrange(len(kept))] if kept else (None, None)
        a = rng.randrange(0, 6 * SR - crop)
        if kept and rng.random() < 0.5 and len(kept) > 1:        # additive remix: another segment's residual
            x2, s2 = kept[rng.randrange(len(kept))]
            y = s[a: a + crop]
            n = (x2 - s2)[a: a + crop]
            return (y + n * rng.uniform(0.5, 1.5)).astype(np.float32), y
        base = s if kept else np.zeros(6 * SR, np.float32)
        if noise:                                                 # link re-degradation with harvested field noise
            base = base + noise[rng.randrange(len(noise))] * rng.uniform(0.3, 1.0)
        yin, tgt, _, _ = radio_link(base.astype(np.float32), (s if kept else base).astype(np.float32),
                                    np.random.default_rng(rng.randrange(2 ** 31)), radio_params)
        return yin[a: a + crop], tgt[a: a + crop]

    for step in range(steps):
        items = [replay[rng.randrange(len(replay))] for _ in range(batch // 2)]      # >= 50 % synthetic replay
        xs = [it["noisy"] for it in items]
        ys = [it["clean"] for it in items]
        vs = [it["valid"] for it in items]
        for _ in range(batch - batch // 2):
            fx, fy = field_pair()
            xs.append(torch.from_numpy(fx))
            ys.append(torch.from_numpy(fy))
            vs.append(torch.ones(crop))
        x, y, v = (torch.stack(t).to(device) for t in (xs, ys, vs))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            est, _ = S(x)
        L = crit(est, y, v, torch.zeros_like(v))["total"]
        L = L + 1e-3 * sum(((p - q) ** 2).sum() for p, q in zip(S.parameters(), anchor))
        opt.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(S.parameters(), 1.0)
        opt.step()
    S.eval()
    rep_t = probe(T, replay_root, device, probe_pairs)
    rep_s = probe(S, replay_root, device, probe_pairs)
    report.update(probe_teacher=rep_t, probe_student=rep_s)
    ok = (rep_s["si_snr"] - rep_t["si_snr"] >= -0.2 and rep_s["stoi"] - rep_t["stoi"] >= -0.005
          and rep_s["pesq_nb"] - rep_t["pesq_nb"] >= -0.05
          and rep_s["silence_out_db"] <= max(rep_t["silence_out_db"] + 1.0, -60.0))   # invents no audible sound
    out = Path(registry) / f"cand_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    ck = torch.load(tpath, map_location="cpu")
    ck["model"] = {k: v.detach().cpu() for k, v in S.state_dict().items()}
    torch.save(ck, out)
    report["candidate"] = str(out)
    report["result"] = "shadow" if ok else "rejected"
    reg.add(out, "shadow" if ok else "rejected", report)
    return report


@torch.no_grad()
def probe(net: HubNet, replay_root: Path, device, n: int) -> dict:
    """Synthetic v3.1 TEST probe (true targets) + the invented-energy check on noise-only input."""
    from pystoi import stoi
    ds = HubPairs(replay_root / "test", None, limit=n, seed=7)
    r = {"si_snr": [], "stoi": [], "pesq_nb": []}
    for i in range(len(ds)):
        d = ds.item(i)
        y = enhance(net, d["noisy"], device)
        c, v = d["clean"], d["valid"]
        r["si_snr"].append(float(si_snr(torch.from_numpy(y), torch.from_numpy(c), torch.from_numpy(v))))
        r["stoi"].append(stoi(c * v, y * v, SR))
        r["pesq_nb"].append(pesq_nb(y * v, c * v))
    rng = np.random.default_rng(3)
    quiet = (rng.standard_normal(6 * SR) * 0.003).astype(np.float32)          # no speech: output must stay quiet
    sil = enhance(net, quiet, device)
    out = {k: float(np.nanmean(v)) for k, v in r.items()}
    out["silence_out_db"] = float(10 * np.log10(np.mean(sil ** 2) + 1e-12))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", type=Path, required=True, help="folder of recorded transmissions (*.wav)")
    ap.add_argument("--replay", type=Path, required=True, help="battlefield_v31 folder (synthetic replay + probe)")
    ap.add_argument("--registry", type=Path, default=ROOT / "artefacts" / "registry")
    ap.add_argument("--teacher", type=Path)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--promote", type=Path, help="promote a shadow candidate after its shadow period")
    ap.add_argument("--rollback", action="store_true")
    a = ap.parse_args()
    reg = Registry(a.registry)
    if a.rollback:
        print("live model now:", reg.rollback())
        return
    if a.promote:
        reg.add(a.promote, "promoted", {"by": "operator"})
        print("promoted", a.promote)
        return
    print(json.dumps(adapt_round(a.buffer, a.replay, a.registry, a.teacher, a.steps), indent=1))


if __name__ == "__main__":
    main()
