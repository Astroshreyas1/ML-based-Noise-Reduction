"""Probes: where and why ANC-Net v3 falls short of the targets, and whether the
data (Lombard speech, noise) or the model is the limit.

    .venv/Scripts/python.exe model/probe.py --ckpt model/runs/v3a/ckpt_best.pt --n 200

P1  Oracle masks at the model's own front end (asym 256/128/hop 64): ideal complex
    ratio mask clipped to |M| <= 1, and the ideal magnitude mask with noisy phase.
    -> the ceiling any mask-based model can reach with this STFT and this target.
P2  Oracle with a 512/128 symmetric STFT (32 ms latency) -> what the 8 ms window costs.
P3  Self-distortion: model on [target, 0] (no noise) -> how much the model damages
    clean speech on its own.
P4  Target definition: score outputs against the DRY speech (before the AGC gain)
    as well as the target; PESQ(dry, target) = what the ducking alone costs against a
    judge who compares to dry speech. Also PESQ(target, mic-EQ(target)).
P5  Where the error lives: residual energy in target-silent frames (noise leak) vs
    error in speech frames (speech distortion), per pair.
P6  Breakdown of the eval CSV by speech corpus / effort / talker / bed pool / gunfire.
P7  Speech-source quality: per-corpus in-utterance floor and bandwidth of the dry
    snippets (is the Lombard material itself the problem?).
Writes <run>/probes.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))

from anc_net_v3 import ANCNetV3  # noqa: E402
from anc_net import AsymSTFT, HOP  # noqa: E402
from data import PairSet  # noqa: E402
from losses import si_snr_db  # noqa: E402
from ancdata.metrics import pesq_wb, stoi as stoi_fn  # noqa: E402


def score(est: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    e, r = torch.from_numpy(est)[None], torch.from_numpy(ref)[None]
    out = {"si_snr": float(si_snr_db(e, r)[0]), "stoi": stoi_fn(est, ref)}
    try:
        out["pesq"] = pesq_wb(est, ref)
    except Exception:
        out["pesq"] = np.nan
    return out


def oracle_asym(fe: AsymSTFT, boom: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    X = fe.stft(boom[None, None])[0, 0]                 # (2, F, T)
    S = fe.stft(target[None, None])[0, 0]
    xc = torch.complex(X[0], X[1])
    sc = torch.complex(S[0], S[1])
    if kind == "crm":
        m = sc / (xc + 1e-8)
        mag = m.abs().clamp(max=1.0)
        m = m / (m.abs() + 1e-8) * mag
        y = m * xc
    else:                                              # ideal magnitude mask, noisy phase
        g = (sc.abs() / (xc.abs() + 1e-8)).clamp(max=1.0)
        y = g * xc
    return fe.istft(torch.stack([y.real, y.imag])[None], boom.shape[-1])[0]


def oracle_sym(boom: torch.Tensor, target: torch.Tensor, n_fft: int = 512, hop: int = 128) -> torch.Tensor:
    win = torch.hann_window(n_fft)
    X = torch.stft(boom, n_fft, hop, window=win, return_complex=True)
    S = torch.stft(target, n_fft, hop, window=win, return_complex=True)
    m = S / (X + 1e-8)
    m = m / (m.abs() + 1e-8) * m.abs().clamp(max=1.0)
    return torch.istft(m * X, n_fft, hop, window=win, length=boom.shape[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev)
    hp = ck["hp"]
    net = ANCNetV3(tuple(hp["widths"]), hp["gru_hidden"], hp["gru_layers"], frontend_pr=hp.get("frontend_pr", False)).to(dev).eval()
    net.load_state_dict(ck["model"])
    fe = AsymSTFT(pr=True)                      # oracle ceilings with the perfect-reconstruction front end
    fe_ckpt = AsymSTFT(pr=hp.get('frontend_pr', False))
    ds = PairSet(ROOT / "data" / "battlefield_v3" / "test", crop_seconds=None)
    idx = np.linspace(0, len(ds) - 1, a.n).astype(int)

    from ancdata.battlefield import build_battlefield_chain, load_battlefield_config
    from ancdata.channel import mic_eq, DEFAULT
    cfg = dict(load_battlefield_config(ROOT / "configs" / "battlefield.yaml"), debug_stems=True)
    chain = build_battlefield_chain(cfg, "test")

    rows = []
    with torch.no_grad():
        for k in idx:
            it = ds[int(k)]
            noisy = it["noisy"]
            target = it["clean"]
            boom = noisy[0]
            pair = chain.generate(int(k))
            dry = torch.from_numpy(pair.meta["_stems"]["speech_dry"])
            gain = torch.from_numpy(pair.meta["_stems"]["agc_gain"])
            est = net(noisy[None].to(dev))[0][0].cpu()
            est_clean_in = net(torch.stack([target, torch.zeros_like(target)])[None].to(dev))[0][0].cpu()
            r = {"id": it["id"], "scenario": it["scenario"], "snr": it["snr"]}
            r.update({f"in_{k2}": v for k2, v in score(boom.numpy(), target.numpy()).items()})
            r.update({f"model_{k2}": v for k2, v in score(est.numpy(), target.numpy()).items()})
            r.update({f"oracle_crm_{k2}": v for k2, v in score(oracle_asym(fe, boom, target, "crm").numpy(), target.numpy()).items()})
            r.update({f"identity_ckptfe_{k2}": v for k2, v in score(fe_ckpt.istft(fe_ckpt.stft(target[None, None])[:, 0], len(target))[0].numpy(), target.numpy()).items()})
            r.update({f"oracle_irm_{k2}": v for k2, v in score(oracle_asym(fe, boom, target, "irm").numpy(), target.numpy()).items()})
            r.update({f"oracle512_{k2}": v for k2, v in score(oracle_sym(boom, target).numpy(), target.numpy()).items()})
            r.update({f"selfdist_{k2}": v for k2, v in score(est_clean_in.numpy(), target.numpy()).items()})
            r.update({f"model_vs_dry_{k2}": v for k2, v in score(est.numpy(), dry.numpy()).items()})
            r.update({f"target_vs_dry_{k2}": v for k2, v in score(target.numpy(), dry.numpy()).items()})
            eq = torch.from_numpy(mic_eq(target.numpy(), DEFAULT))
            r.update({f"eq_vs_target_{k2}": v for k2, v in score(eq.numpy(), target.numpy()).items()})
            # P5 error split by frame type
            t = len(target) // HOP
            te = (target[: t * HOP].reshape(t, HOP) ** 2).mean(-1)
            ee = ((est - target)[: t * HOP].reshape(t, HOP) ** 2).mean(-1)
            oe = (est[: t * HOP].reshape(t, HOP) ** 2).mean(-1)
            silent = te < 1e-7
            r["leak_db"] = float(10 * np.log10(oe[silent].mean() / (te[~silent].mean() + 1e-12) + 1e-12)) if silent.any() else np.nan
            r["speech_err_db"] = float(10 * np.log10(ee[~silent].mean() / (te[~silent].mean() + 1e-12) + 1e-12))
            r["silent_frac"] = float(silent.float().mean())
            r["agc_min_db"] = float(20 * np.log10(gain.min() / gain.max() + 1e-9))
            rows.append(r)
    df = pd.DataFrame(rows)
    out = a.ckpt.parent
    df.to_csv(out / "probes.csv", index=False)

    def m(prefix):
        return {k: round(float(df[f"{prefix}_{k}"].mean()), 3) for k in ("si_snr", "stoi", "pesq")}
    lines = [f"# Probes: {a.ckpt} on {a.n} test pairs\n"]
    tab = pd.DataFrame({name: m(p) for name, p in [("input", "in"), ("ANC-Net v3", "model"), ("oracle CRM |M|<=1 @256/128/64", "oracle_crm"),
                                                   ("oracle IRM (noisy phase) @256/128/64", "oracle_irm"), ("oracle CRM @512/128 (32 ms)", "oracle512")]}).T
    lines.append("## P1-P2 Ceilings vs the model (scored against the target)\n\n" + tab.to_markdown() + "\n")
    lines.append(f"## P3 Self-distortion: model fed [target, 0]\n\n{m('selfdist')} (perfect = 4.64 PESQ / STOI 1.0)\n")
    lines.append("## P4 Target definition\n\n" + pd.DataFrame({"model vs DRY speech": m("model_vs_dry"), "model vs target": m("model"),
                                                              "target vs dry (AGC ducking alone)": m("target_vs_dry"),
                                                              "mic-EQ(target) vs target": m("eq_vs_target")}).T.to_markdown() + "\n")
    lines.append(f"## P5 Where the error lives\n\nresidual in target-silent frames: {df.leak_db.mean():.1f} dB re speech energy (silent fraction {df.silent_frac.mean():.2f}); "
                 f"error in speech frames: {df.speech_err_db.mean():.1f} dB re speech energy; AGC excursion p50 {df.agc_min_db.median():.1f} dB\n")
    by_snr = df.assign(bin=pd.cut(df.snr, [-9, -4, 0, 4, 8, 12, 16])).groupby("bin", observed=True)[["in_pesq", "model_pesq", "oracle_crm_pesq", "oracle512_pesq", "model_stoi", "oracle_crm_stoi"]].mean().round(2)
    lines.append("## P1 by drawn SNR\n\n" + by_snr.to_markdown() + "\n")

    # P6 breakdown of the full eval csv joined with meta
    ev = pd.read_csv(out / "eval_test.csv", dtype={"id": str})
    meta = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data" / "battlefield_v3" / "test" / "meta.jsonl").open(encoding="utf-8")}
    ev["corpus"] = ev.id.map(lambda i: meta[i]["speech_corpus"])
    ev["effort"] = ev.id.map(lambda i: meta[i]["speech_effort"])
    ev["talker"] = ev.id.map(lambda i: meta[i]["speech_speaker_id"])
    ev["bed_pool"] = ev.id.map(lambda i: meta[i]["layers"][0]["pool"])
    ev["speech_level"] = ev.id.map(lambda i: meta[i]["speech_level_db"])
    cols = [c for c in ("si_snri", "stoi", "pesq", "blackout") if c in ev.columns]
    for key in ("corpus", "effort", "talker", "bed_pool", "gunfire"):
        g = ev.groupby(key)[cols].mean().round(3)
        g["n"] = ev.groupby(key).size()
        lines.append(f"## P6 by {key}\n\n" + g.sort_values("pesq").to_markdown() + "\n")
    ev["lvl_bin"] = pd.cut(ev.speech_level, [-33, -29, -25, -21, -17])
    lines.append("## P6 by speech level (dBFS)\n\n" + ev.groupby("lvl_bin", observed=True)[cols].mean().round(3).to_markdown() + "\n")

    # P7 speech-source quality
    from ancdata.snippets import load_snippets
    import soundfile as sf
    sn = load_snippets(ROOT / "data" / "snippets" / "lombard6s")
    q = []
    for corpus in ("lombardgrid", "avid"):
        sub = sn[(sn.corpus == corpus)].sample(60, random_state=0)
        for _, r in sub.iterrows():
            x, _ = sf.read(r.abs_path, dtype="float32")
            fr = x[: len(x) // HOP * HOP].reshape(-1, HOP)
            e = 10 * np.log10((fr ** 2).mean(-1) + 1e-12)
            act = e > e.max() - 35
            floor = np.percentile(e[act], 5) if act.sum() > 10 else np.nan
            X = np.abs(np.fft.rfft(x)) ** 2
            f = np.fft.rfftfreq(len(x), 1 / 16000)
            hf = 10 * np.log10(X[f > 4000].sum() / (X[(f > 300) & (f <= 4000)].sum() + 1e-12))
            q.append({"corpus": corpus, "floor_rel_peak_db": floor - e.max(), "hf_4k_8k_rel_db": hf, "peak": float(np.abs(x).max()),
                      "speech_fraction": r.speech_fraction})
    qd = pd.DataFrame(q).groupby("corpus").mean().round(2)
    lines.append("## P7 Speech source quality (60 snippets per corpus)\n\n" + qd.to_markdown() +
                 "\n\nfloor_rel_peak_db: 5th-percentile active-frame energy re the loudest frame (in-utterance floor / reverb tail); "
                 "hf_4k_8k_rel_db: energy above 4 kHz re 0.3-4 kHz (bandwidth).\n")
    (out / "probes.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
